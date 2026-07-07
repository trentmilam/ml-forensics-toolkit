"""Thin, fail-safe client for the local OpenAI-compatible gateway.

Every method has a SHORT timeout and returns None on any failure so the
experiment can always complete against the deterministic in-code oracle.
The caller records which source was actually used.
"""
from __future__ import annotations
import json
import os
import urllib.request
import urllib.error

BASE = os.environ.get("BDAPP_GATEWAY_BASE", "http://127.0.0.1:8000")
# Some local OpenAI-compatible gateways expose a custom, non-standard field for selecting an
# inference profile/persona. Not part of the OpenAI API -- override to match your own gateway.
PROFILE_FIELD = os.environ.get("BDAPP_GATEWAY_PROFILE_FIELD", "profile")


def _post(path: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        BASE + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def list_models(timeout: float = 8.0):
    try:
        req = urllib.request.Request(BASE + "/v1/models")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            j = json.loads(resp.read().decode("utf-8"))
        return [m["id"] for m in j.get("data", [])]
    except Exception:
        return None


def chat(model: str, prompt: str, timeout: float = 15.0, max_tokens: int = 24,
         profile: str | None = None):
    """Return the assistant text, or None on ANY failure.

    `profile` is sent under PROFILE_FIELD -- a custom extension some local
    OpenAI-compatible gateways expose for persona/sampler selection. A
    thinking-enabled model under its default persona may spend its whole token
    budget on hidden reasoning and return EMPTY assistant content; if that
    happens with your gateway, select a thinking-off profile via `profile=` to
    get a usable answer. An empty/whitespace content is treated as a non-answer
    (returns None) so the fail-safe fallback path is entered honestly rather
    than on a blank string.
    """
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if profile:
            payload[PROFILE_FIELD] = profile
        j = _post("/v1/chat/completions", payload, timeout)
        content = j["choices"][0]["message"].get("content")
        if content is None:
            return None
        content = content.strip()
        return content if content else None
    except Exception:
        return None


def embed(model: str, inputs, timeout: float = 15.0):
    """Return list of embedding vectors, or None on ANY failure."""
    try:
        j = _post("/v1/embeddings", {"model": model, "input": list(inputs)}, timeout)
        return [d["embedding"] for d in j["data"]]
    except Exception:
        return None
