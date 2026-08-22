"""VLM backend clients with a single generate() interface.

Key design:
- `--api_key` arg (passed in ctor) wins over env var.
- Each backend hides SDK details; agents just call .generate(system, user, images=[paths]).
- ClaudeCodeBackend uses a file-based request/response exchange so the current
  Claude Code chat session can fulfil each step manually.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path


def _caller_agent() -> str:
    """Best-effort tag of which agent function triggered this VLM call.

    Walks up the stack to the first frame defined in agents.py (or, failing
    that, run_pipeline.py) and returns its function name.
    """
    try:
        frame = sys._getframe(2)
        fallback = ""
        while frame is not None:
            fn = frame.f_code.co_filename
            name = frame.f_code.co_name
            # Skip private helpers (e.g. _screen_then_escalate) so calls are
            # attributed to the agent function that used them.
            if fn.endswith("agents.py") and not name.startswith("_"):
                return name
            if not fallback and fn.endswith("run_pipeline.py"):
                fallback = name
            frame = frame.f_back
        return fallback or "unknown"
    except Exception:
        return "unknown"


def log_usage(record: dict) -> None:
    """Append one JSON line to the usage log if VLM_USAGE_LOG is set.

    Never raises: instrumentation must not break the pipeline.
    """
    path = os.environ.get("VLM_USAGE_LOG")
    if not path:
        return
    try:
        record.setdefault("ts", time.time())
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def _guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "image/png"


def _read_image_b64(path: str) -> tuple[str, str]:
    mime = _guess_mime(path)
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("ascii"), mime


def _resolve_key(explicit: str | None, env_var: str) -> str | None:
    if explicit:
        return explicit
    return os.environ.get(env_var)


class VLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, system: str, user: str, images: list[str] | None = None) -> str:
        ...


class OpenAIBackend(VLMBackend):
    name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o"):
        from openai import OpenAI  # lazy
        key = _resolve_key(api_key, "OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAI: no api key (pass --api_key or set OPENAI_API_KEY).")
        self.client = OpenAI(api_key=key)
        self.model = model
        # Running token-usage tallies for the lifetime of this backend.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

    def generate(self, system, user, images=None):
        content = [{"type": "text", "text": user}]
        for p in images or []:
            b64, mime = _read_image_b64(p)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        kwargs = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            response_format={"type": "json_object"},
        )
        # gpt-5 / o-series reasoning models only allow temperature=1 (default); omit.
        if not (self.model.startswith("gpt-5") or self.model.startswith("o")):
            kwargs["temperature"] = 0
        t0 = time.time()
        resp = self.client.chat.completions.create(**kwargs)
        latency = time.time() - t0
        usage = getattr(resp, "usage", None)
        log_usage({
            "backend": self.name,
            "model": self.model,
            "agent": _caller_agent(),
            "latency_sec": round(latency, 3),
            "n_images": len(images or []),
            "system_chars": len(system or ""),
            "user_chars": len(user or ""),
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        })
        if usage is not None:
            pt = int(getattr(usage, "prompt_tokens", 0) or 0)
            ct = int(getattr(usage, "completion_tokens", 0) or 0)
            self.total_prompt_tokens += pt
            self.total_completion_tokens += ct
            self.total_calls += 1
            print(
                f"[openai-usage] call#{self.total_calls} prompt={pt} "
                f"completion={ct}  cumulative: prompt={self.total_prompt_tokens} "
                f"completion={self.total_completion_tokens}"
            )
        return resp.choices[0].message.content or ""


class GeminiBackend(VLMBackend):
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-pro"):
        import google.generativeai as genai  # lazy
        key = _resolve_key(api_key, "GEMINI_API_KEY") or _resolve_key(None, "GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Gemini: no api key (pass --api_key or set GEMINI_API_KEY/GOOGLE_API_KEY).")
        genai.configure(api_key=key)
        self._genai = genai
        self.model_name = model

    def generate(self, system, user, images=None):
        parts = [user]
        for p in images or []:
            with open(p, "rb") as f:
                parts.append({"mime_type": _guess_mime(p), "data": f.read()})
        model = self._genai.GenerativeModel(
            self.model_name,
            system_instruction=system,
            generation_config={"response_mime_type": "application/json", "temperature": 0},
        )
        resp = model.generate_content(parts)
        return resp.text or ""


class ClaudeAPIBackend(VLMBackend):
    name = "claude"

    def __init__(self, api_key: str | None = None, model: str = "claude-opus-4-7"):
        import anthropic  # lazy
        key = _resolve_key(api_key, "ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Claude API: no api key (pass --api_key or set ANTHROPIC_API_KEY).")
        self.client = anthropic.Anthropic(api_key=key)
        self.model = model

    def generate(self, system, user, images=None):
        # Prompt-caching modes (opt-in via env VLM_PROMPT_CACHE):
        #   unset/'' -> payload identical to the original code (no caching)
        #   "1"      -> H1 vanilla: system + first image breakpoints (REFUTED:
        #               unique crop-image writes cost more than the reads earn)
        #   "system" -> H1.1 arm A: system-prompt breakpoint ONLY, images never
        #               marked — removes H1's wasted writes, keeps its only wins
        mode = os.environ.get("VLM_PROMPT_CACHE", "")
        cache_images = mode == "1"
        cache_system = mode in ("1", "system")
        content = []
        for i, p in enumerate(images or []):
            b64, mime = _read_image_b64(p)
            block = {
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            }
            if cache_images and i == 0:
                block["cache_control"] = {"type": "ephemeral"}
            content.append(block)
        content.append({"type": "text", "text": user})
        if cache_system:
            system_param = [{"type": "text", "text": system,
                            "cache_control": {"type": "ephemeral"}}]
        else:
            system_param = system
        t0 = time.time()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            # temperature=0,
            system=system_param,
            messages=[{"role": "user", "content": content}],
        )
        latency = time.time() - t0
        usage = getattr(resp, "usage", None)
        log_usage({
            "backend": self.name,
            "model": self.model,
            "agent": _caller_agent(),
            "latency_sec": round(latency, 3),
            "n_images": len(images or []),
            "system_chars": len(system or ""),
            "user_chars": len(user or ""),
            "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            "cache_creation_input_tokens": int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
            "cache_read_input_tokens": int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        })
        # resp.content is a list of blocks; join text blocks.
        parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
        return "".join(parts)


class ClaudeCodeBackend(VLMBackend):
    """Semi-interactive backend: writes a request JSON + copies image, then
    polls for a response JSON. The current Claude Code chat session (me)
    fulfils each request by writing the response file.
    """
    name = "claude_code"

    def __init__(
        self,
        exchange_dir: str | Path,
        poll_sec: float = 1.5,
        timeout_sec: float = 3600,
        api_key: str | None = None,  # accepted for uniformity; unused
        model: str | None = None,     # accepted for uniformity; unused
    ):
        self.exchange_dir = Path(exchange_dir)
        self.exchange_dir.mkdir(parents=True, exist_ok=True)
        self.poll_sec = poll_sec
        self.timeout_sec = timeout_sec

    def generate(self, system, user, images=None):
        req_id = uuid.uuid4().hex[:8]
        req_path = self.exchange_dir / f"request_{req_id}.json"
        resp_path = self.exchange_dir / f"response_{req_id}.json"
        payload = {
            "id": req_id,
            "system": system,
            "user": user,
            "images": [str(Path(p).resolve()) for p in (images or [])],
            "response_path": str(resp_path),
        }
        with req_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(
            f"[claude_code] Wrote request {req_path}. "
            f"Waiting for response at {resp_path} ... (timeout {self.timeout_sec}s)"
        )
        start = time.time()
        while not resp_path.exists():
            if time.time() - start > self.timeout_sec:
                raise TimeoutError(f"No response at {resp_path} within {self.timeout_sec}s")
            time.sleep(self.poll_sec)
        with resp_path.open("r") as f:
            data = json.load(f)
        if "text" not in data:
            raise ValueError(f"Response file {resp_path} missing 'text' field")
        log_usage({
            "backend": self.name,
            "model": "claude_code",
            "agent": _caller_agent(),
            "latency_sec": round(time.time() - start, 3),
            "n_images": len(images or []),
            "system_chars": len(system or ""),
            "user_chars": len(user or ""),
            "input_tokens": 0,
            "output_tokens": 0,
        })
        return data["text"]


class ClaudeCLIBackend(VLMBackend):
    """Non-interactive `claude -p` subprocess backend (Claude Code subscription,
    no API key / credit needed). Mirrors eval/eval_lib/vlm_clients.py
    ClaudeCodeClient: images are referenced as absolute paths and read by the
    CLI's Read tool; the reply text comes back in the JSON envelope's `result`
    field and token usage in its `usage` field.
    """
    name = "claude_cli"

    def __init__(self, model: str | None = None, binary: str = "claude"):
        self.model = model or "claude-opus-4-7"
        self.binary = binary

    def generate(self, system, user, images=None):
        if images:
            image_block = "\n".join(
                f"  - image {i+1}: {os.path.abspath(p)}" for i, p in enumerate(images)
            )
            user_full = (
                f"{user}\n\nImages to inspect (read each with the Read tool):\n{image_block}"
            )
        else:
            user_full = user
        cmd = [
            self.binary,
            "-p",
            "--model", self.model,
            "--output-format", "json",
            "--append-system-prompt", system,
            "--tools", "Read",
            "--permission-mode", "bypassPermissions",
            user_full,
        ]
        t0 = time.time()
        # 7 attempts, exponential backoff capped at 300 s (~16 min total
        # coverage): a 2026-08-21 03:48 transient CLI outage outlasted the old
        # 4-attempt/2-min window and killed a 2.5 h run mid-flight.
        max_attempts = 7
        last_error: Exception | None = None
        envelope = None
        for attempt in range(max_attempts):
            try:
                # Never let the CLI fall back to API-key billing: this backend
                # exists precisely to use the subscription account.
                env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600, check=False,
                    env=env,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"claude CLI exit {proc.returncode}: stderr={proc.stderr[:600]!r}"
                    )
                envelope = json.loads(proc.stdout)
                break
            except (subprocess.TimeoutExpired, RuntimeError, json.JSONDecodeError) as e:
                last_error = e
                if attempt < max_attempts - 1:
                    sleep_s = min(300, 15 * (2 ** attempt))  # 15..300 s
                    print(
                        f"[claude_cli] call failed (attempt {attempt+1}/{max_attempts}): "
                        f"{e}. Retrying in {sleep_s}s...",
                        flush=True,
                    )
                    time.sleep(sleep_s)
        if envelope is None:
            raise RuntimeError(
                f"claude CLI failed after {max_attempts} attempts: {last_error}"
            )
        usage = envelope.get("usage") or {}
        log_usage({
            "backend": self.name,
            "model": self.model,
            "agent": _caller_agent(),
            "latency_sec": round(time.time() - t0, 3),
            "n_images": len(images or []),
            "system_chars": len(system or ""),
            "user_chars": len(user or ""),
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "cache_creation_input_tokens": int(usage.get("cache_creation_input_tokens", 0) or 0),
            "cache_read_input_tokens": int(usage.get("cache_read_input_tokens", 0) or 0),
            "cli_num_turns": envelope.get("num_turns"),
            "cli_total_cost_usd": envelope.get("total_cost_usd"),
        })
        result = envelope.get("result")
        if not isinstance(result, str) or not result.strip():
            raise ValueError(
                f"claude CLI returned no usable text. keys={list(envelope.keys())}"
            )
        return result


def build_backend(name: str, api_key: str | None, exchange_dir: str | None, model: str | None = None) -> VLMBackend:
    name = name.lower()
    if name == "openai":
        return OpenAIBackend(api_key=api_key, model=model or "gpt-4o")
    if name == "gemini":
        return GeminiBackend(api_key=api_key, model=model or "gemini-2.5-pro")
    if name == "claude":
        return ClaudeAPIBackend(api_key=api_key, model=model or "claude-opus-4-7")
    if name == "claude_cli":
        return ClaudeCLIBackend(model=model)
    if name == "claude_code":
        if not exchange_dir:
            raise ValueError("claude_code backend requires exchange_dir")
        return ClaudeCodeBackend(exchange_dir=exchange_dir)
    raise ValueError(f"Unknown backend: {name}")
