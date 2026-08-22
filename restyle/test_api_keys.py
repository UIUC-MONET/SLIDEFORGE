"""Low-cost reachability check for the two API keys in ``api_keys.txt``.

  * Anthropic: one Claude Haiku 4.5 call with a 5-token max_tokens cap
    (a few hundredths of a cent).
  * OpenAI:   one ``client.models.list()`` call (free; just lists models).
              We then verify ``gpt-image-2`` is visible in the listing so
              you know the image-edit calls used by step1 / step3_class5
              will be reachable later.

No image generation is triggered, so total cost is well under $0.01.
"""

from __future__ import annotations

import sys

from common import OPENAI_IMAGE_MODEL, anthropic_client, load_api_keys, openai_client


def test_anthropic() -> tuple[bool, str]:
    try:
        client = anthropic_client()
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
        usage = getattr(resp, "usage", None)
        msg = f"OK (model=claude-haiku-4-5, usage={usage})"
        return True, msg
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_openai() -> tuple[bool, str]:
    try:
        client = openai_client()
        models = list(client.models.list())
        ids = {m.id for m in models}
        if OPENAI_IMAGE_MODEL in ids:
            return True, f"OK (models.list -> {len(ids)} model(s); '{OPENAI_IMAGE_MODEL}' visible)"
        # Listing worked but the image model isn't visible. The key is
        # valid; the image-edit calls might still fail later for org/access
        # reasons. Flag it but don't call it a hard fail.
        return True, (
            f"PARTIAL (models.list -> {len(ids)} model(s); "
            f"'{OPENAI_IMAGE_MODEL}' NOT in listing - image-edit calls may 404)"
        )
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    try:
        keys = load_api_keys()
    except Exception as e:
        print(f"[api_keys.txt] FAIL: {e}", file=sys.stderr)
        return 2
    print(f"[api_keys.txt] loaded keys: {sorted(keys.keys())}")

    a_ok, a_msg = test_anthropic()
    print(f"[Anthropic] {'OK ' if a_ok else 'FAIL'} {a_msg}")

    o_ok, o_msg = test_openai()
    print(f"[OpenAI]    {'OK ' if o_ok else 'FAIL'} {o_msg}")

    return 0 if (a_ok and o_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
