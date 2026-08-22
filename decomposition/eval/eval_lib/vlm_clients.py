"""Unified VLM client for the per-component (M3) and holistic (M4) judges.

Providers:
  - claude-code : non-interactive `claude -p` subprocess (uses your Claude Code account)
  - claude-api  : direct Anthropic SDK calls (requires ANTHROPIC_API_KEY)
  - openai      : OpenAI SDK (requires OPENAI_API_KEY)
  - gemini      : Google GenAI SDK (requires GEMINI_API_KEY or GOOGLE_API_KEY)

Every provider implements:

    judge_json(system: str, user: str, images: list[str], schema: dict) -> dict

The schema is a JSON Schema describing the expected response. Each backend uses
the most native structured-output mechanism it has; if none, we fall back to a
"respond with raw JSON only" prompt and best-effort parse.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
from typing import Any


class VLMError(RuntimeError):
    pass


def _record_usage(resp: Any) -> None:
    """Append per-call token usage to CLAUDE_API_USAGE_LOG dir (opt-in, best-effort).

    One JSONL file per pid to avoid cross-process write contention. Never raises:
    usage accounting must not be able to break an eval run.
    """
    log_dir = os.environ.get("CLAUDE_API_USAGE_LOG")
    if not log_dir:
        return
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    try:
        rec = {
            "i": getattr(usage, "input_tokens", 0) or 0,
            "o": getattr(usage, "output_tokens", 0) or 0,
            "cr": getattr(usage, "cache_read_input_tokens", 0) or 0,
            "cc": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        }
        path = os.path.join(log_dir, f"usage_{os.getpid()}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


def _record_openai_usage(resp: Any) -> None:
    """Append per-call OpenAI token usage to OPENAI_USAGE_LOG dir (opt-in, best-effort)."""
    log_dir = os.environ.get("OPENAI_USAGE_LOG")
    if not log_dir:
        return
    usage = getattr(resp, "usage", None)
    if usage is None:
        return
    try:
        details = getattr(usage, "completion_tokens_details", None)
        rec = {
            "i": getattr(usage, "prompt_tokens", 0) or 0,
            "o": getattr(usage, "completion_tokens", 0) or 0,
            "r": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
        }
        path = os.path.join(log_dir, f"usage_{os.getpid()}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception:
        pass


# ---------- helpers -----------------------------------------------------------

def _b64_image(path: str) -> tuple[str, str]:
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    if ext in ("jpg",):
        ext = "jpeg"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii"), f"image/{ext}"


def _extract_json(text: str) -> dict:
    """Best-effort JSON parse from a possibly chatty response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first {...} balanced block.
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise VLMError(f"Could not parse JSON from VLM response: {text[:400]!r}")


# ---------- claude-code (CLI) -------------------------------------------------

class ClaudeCodeClient:
    """Talks to the local `claude` CLI in non-interactive mode.
    Uses your logged-in Claude Code account — no API key required."""

    def __init__(self, model: str = "sonnet", binary: str = "claude"):
        self.model = model
        self.binary = binary

    def judge_json(self, system: str, user: str, images: list[str], schema: dict) -> dict:
        # claude CLI can read local files via the Read tool, so we reference paths
        # inline and instruct the model to inspect them.
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
            "--json-schema", json.dumps(schema),
            user_full,
        ]
        # Retry with backoff so transient failures (rate limits, timeouts) don't
        # get baked into results.json as permanent per-slide errors.
        import time
        max_attempts = 4
        last_error: Exception | None = None
        envelope = None
        for attempt in range(max_attempts):
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300, check=False
                )
                if proc.returncode != 0:
                    raise VLMError(
                        f"claude CLI exit {proc.returncode}: stderr={proc.stderr[:600]!r}"
                    )
                envelope = json.loads(proc.stdout)
                break
            except (subprocess.TimeoutExpired, VLMError, json.JSONDecodeError) as e:
                last_error = e
                if attempt < max_attempts - 1:
                    sleep_time = 15 * (2 ** attempt)  # 15s, 30s, 60s
                    print(
                        f"  [claude-code-client] call failed "
                        f"(attempt {attempt+1}/{max_attempts}): {e}. "
                        f"Retrying in {sleep_time}s...",
                        flush=True,
                    )
                    time.sleep(sleep_time)
        if envelope is None:
            raise VLMError(
                f"claude CLI failed after {max_attempts} attempts: {last_error}"
            ) from last_error
        # With --json-schema the validated payload lives at top-level
        # `structured_output`; the plain text reply is in `result`.
        struct = envelope.get("structured_output")
        if isinstance(struct, dict):
            return struct
        if isinstance(struct, str) and struct:
            return _extract_json(struct)
        result = envelope.get("result")
        if isinstance(result, dict):
            return result
        if isinstance(result, str) and result.strip():
            return _extract_json(result)
        raise VLMError(
            "claude CLI returned no usable output. "
            f"keys={list(envelope.keys())} result={result!r} structured_output={struct!r}"
        )


# ---------- claude-api (Anthropic SDK) ----------------------------------------

# JSON Schema keywords the structured-outputs endpoint does not accept.
_SO_UNSUPPORTED_KEYS = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "pattern", "minItems", "maxItems",
}


def _schema_for_structured_output(schema: dict) -> dict:
    """Sanitize a judge schema for the API's output_config.format:
    strip numeric/string constraints and force additionalProperties=false."""
    def walk(node):
        if isinstance(node, dict):
            out = {k: walk(v) for k, v in node.items() if k not in _SO_UNSUPPORTED_KEYS}
            if out.get("type") == "object":
                out["additionalProperties"] = False
                props = out.get("properties")
                if isinstance(props, dict) and "required" not in out:
                    out["required"] = list(props)
            return out
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node
    return walk(schema)


class ClaudeAPIClient:
    """Direct Anthropic API judge (requires ANTHROPIC_API_KEY).

    - Prefers structured outputs (output_config.format json_schema) so the
      response is guaranteed-valid JSON; permanently falls back to
      "raw JSON only" prompting if the API rejects the parameter/schema.
    - Retries rate limits (honoring retry-after), 5xx/529, network errors,
      timeouts, and JSON parse failures with backoff — on top of the SDK's
      own built-in retries — so transient failures don't get baked into
      results.json as permanent per-slide errors.
    """

    def __init__(self, model: str = "claude-sonnet-4-6"):
        try:
            import anthropic
        except ImportError as e:
            raise VLMError("anthropic SDK not installed (pip install anthropic)") from e
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise VLMError(
                "ANTHROPIC_API_KEY not set. Export it in the shell, or put "
                "`export ANTHROPIC_API_KEY=sk-ant-...` in "
                "components/tianhu_code_rebuttal/.anthropic_key "
                "(sourced automatically by the *_claudeapi.sh scripts)."
            )
        self.model = model
        self._anthropic = anthropic
        # 3-minute per-request timeout; SDK retries connection/429/5xx twice
        # itself, our outer loop adds longer backoff on top.
        self._client = anthropic.Anthropic(timeout=180.0)
        self._use_structured = True

    def _request(self, system: str, image_blocks: list[dict], user: str, schema: dict) -> dict:
        """One API round trip. Returns the parsed JSON verdict."""
        if self._use_structured:
            content = image_blocks + [{"type": "text", "text": user}]
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=system,
                    messages=[{"role": "user", "content": content}],
                    extra_body={"output_config": {"format": {
                        "type": "json_schema",
                        "schema": _schema_for_structured_output(schema),
                    }}},
                )
            except self._anthropic.BadRequestError as e:
                # output_config unsupported for this model/schema -> switch this
                # process to prompt-based JSON mode for all subsequent calls.
                print(
                    f"  [claude-api-client] structured output rejected "
                    f"({str(e)[:200]}); falling back to prompt JSON mode.",
                    flush=True,
                )
                self._use_structured = False
                return self._request(system, image_blocks, user, schema)
        else:
            content = image_blocks + [{
                "type": "text",
                "text": user + "\n\nRespond with raw JSON only, matching this schema:\n"
                        + json.dumps(schema),
            }]
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=2000,
                system=system,
                messages=[{"role": "user", "content": content}],
            )
        _record_usage(resp)
        if resp.stop_reason == "refusal":
            raise VLMError(f"model refused the request (stop_details={getattr(resp, 'stop_details', None)!r})")
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        text = "\n".join(text_parts)
        if resp.stop_reason == "max_tokens":
            raise VLMError(f"response truncated at max_tokens: {text[:200]!r}")
        return _extract_json(text)

    def judge_json(self, system: str, user: str, images: list[str], schema: dict) -> dict:
        import time
        image_blocks: list[dict] = []
        for path in images:
            b64, mime = _b64_image(path)
            image_blocks.append({
                "type": "image",
                "source": {"type": "base64", "media_type": mime, "data": b64},
            })

        a = self._anthropic
        max_attempts = 5
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            sleep_time = 15 * (2 ** attempt)  # 15s, 30s, 60s, 120s
            try:
                return self._request(system, image_blocks, user, schema)
            except a.RateLimitError as e:
                last_error = e
                retry_after = None
                try:
                    retry_after = float(e.response.headers.get("retry-after") or 0)
                except (TypeError, ValueError, AttributeError):
                    pass
                sleep_time = max(sleep_time, retry_after or 0)
            except (a.APIConnectionError, a.InternalServerError) as e:
                # covers network errors, timeouts, 500s and 529 overloaded
                last_error = e
            except a.APIStatusError as e:
                # remaining 4xx (bad request in prompt mode, auth, ...) are not
                # retryable — surface immediately.
                raise VLMError(f"Anthropic API error {e.status_code}: {e}") from e
            except VLMError as e:
                # parse failure / truncation / refusal — retry (except refusal)
                if "refused" in str(e):
                    raise
                last_error = e
            if attempt < max_attempts - 1:
                print(
                    f"  [claude-api-client] call failed "
                    f"(attempt {attempt+1}/{max_attempts}): "
                    f"{type(last_error).__name__}: {str(last_error)[:200]}. "
                    f"Retrying in {sleep_time:.0f}s...",
                    flush=True,
                )
                time.sleep(sleep_time)
        raise VLMError(
            f"Anthropic API failed after {max_attempts} attempts: {last_error}"
        ) from last_error


# ---------- openai ------------------------------------------------------------

class OpenAIClient:
    def __init__(self, model: str = "gpt-4o-mini"):
        try:
            from openai import OpenAI  # noqa
        except ImportError as e:
            raise VLMError("openai SDK not installed (pip install openai)") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise VLMError("OPENAI_API_KEY not set")
        from openai import OpenAI
        self._client = OpenAI()
        self.model = model

    def judge_json(self, system: str, user: str, images: list[str], schema: dict) -> dict:
        parts: list[dict] = [{"type": "text", "text": user}]
        for path in images:
            b64, mime = _b64_image(path)
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": parts},
            ],
            response_format={"type": "json_object"},
        )
        _record_openai_usage(resp)
        return _extract_json(resp.choices[0].message.content or "")


# ---------- gemini ------------------------------------------------------------

class GeminiClient:
    def __init__(self, model: str = "gemini-2.0-flash"):
        try:
            from google import genai  # noqa
        except ImportError as e:
            raise VLMError(
                "google-genai SDK not installed (pip install google-genai)"
            ) from e
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise VLMError("GEMINI_API_KEY (or GOOGLE_API_KEY) not set")
        from google import genai as _genai
        from google.genai import types as gtypes
        # Set a client-level timeout of 60 seconds (60,000 milliseconds) to prevent indefinite socket hangs
        self._client = _genai.Client(
            api_key=api_key,
            http_options=gtypes.HttpOptions(timeout=60000)
        )
        self.model = model

    def judge_json(self, system: str, user: str, images: list[str], schema: dict) -> dict:
        import time
        from google.genai import types as gtypes
        parts: list = [gtypes.Part.from_text(text=user)]
        for path in images:
            with open(path, "rb") as f:
                data = f.read()
            mime = "image/png"
            if path.lower().endswith((".jpg", ".jpeg")):
                mime = "image/jpeg"
            parts.append(gtypes.Part.from_bytes(data=data, mime_type=mime))

        max_attempts = 3
        last_error = None
        for attempt in range(max_attempts):
            try:
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=parts,
                    config=gtypes.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                    ),
                )
                return _extract_json(resp.text)
            except Exception as e:
                last_error = e
                if attempt < max_attempts - 1:
                    sleep_time = 2 ** (attempt + 1)
                    print(f"  [gemini-client] API call failed (attempt {attempt+1}/{max_attempts}): {e}. Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
        raise VLMError(f"Gemini API failed after {max_attempts} attempts: {last_error}") from last_error




# ---------- factory -----------------------------------------------------------

def make_client(provider: str, model: str | None = None):
    p = provider.lower()
    if p in ("claude-code", "claudecode", "cc"):
        return ClaudeCodeClient(model=model or "sonnet")
    if p in ("claude-api", "claude", "anthropic"):
        return ClaudeAPIClient(model=model or "claude-sonnet-4-6")
    if p in ("openai", "gpt"):
        return OpenAIClient(model=model or "gpt-4o-mini")
    if p in ("gemini", "google"):
        return GeminiClient(model=model or "gemini-2.0-flash")
    raise VLMError(f"Unknown VLM provider: {provider}")
