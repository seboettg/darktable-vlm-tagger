"""Minimal Ollama HTTP client for vision-language tagging.

Talks to /api/chat directly so that Ollama handles model loading/swapping
itself. Stdlib only (urllib), no requests dependency.
"""

import json
import time
import urllib.error
import urllib.request


def _post(host: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def unload(host: str, model: str, timeout: int = 30) -> None:
    """Drop the model from VRAM immediately.

    Without this it lingers per OLLAMA_KEEP_ALIVE, and the next model no
    longer fits the card fully - Ollama then silently offloads layers to the
    CPU, which cost a 3x slowdown the first time this was missed.
    """
    try:
        _post(host, {"model": model, "keep_alive": 0}, timeout)
    except Exception:
        pass


def ask(host: str, model: str, prompt: str, image_b64: str, schema: dict,
        timeout: int, num_ctx: int, num_predict: int, temperature: float = 0.2,
        top_p: float = 0.9):
    """One tagging call. Returns (raw_content, elapsed_seconds, diagnostics).

    Always binds the response to `schema` and disables the thinking trace:
    the -instruct model variants answer in ~2s with schema binding versus
    20-28s for the -thinking variants (whose reasoning trace consumes
    thousands of output tokens that never surface in message.content).
    Requires an -instruct tag; the bare Ollama tags resolve to -thinking.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "format": schema,
        "think": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }

    think_rejected = False
    started = time.monotonic()
    try:
        body = _post(host, payload, timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 400 and "think" in payload:
            # This Ollama/model combination doesn't know the "think" field.
            # Fall back to Qwen3's own /no_think prompt instruction.
            think_rejected = True
            payload.pop("think")
            payload["messages"][0]["content"] = prompt + "\n\n/no_think"
            body = _post(host, payload, timeout)
        else:
            raise
    elapsed = time.monotonic() - started

    message = body.get("message", {})
    thinking = message.get("thinking") or ""

    # done_reason: stop | length | load. "length" means num_predict was too
    # small - not necessarily the context window. Compare prompt_tokens
    # against num_ctx to tell the two apart.
    diagnostics = {
        "done_reason": body.get("done_reason"),
        "prompt_tokens": body.get("prompt_eval_count"),
        "output_tokens": body.get("eval_count"),
        "thinking_chars": len(thinking),
        "think_rejected": think_rejected,
    }
    return message.get("content", ""), elapsed, diagnostics
