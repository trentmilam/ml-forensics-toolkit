"""VecStamp -- re-embedding reproduction certificate with failure-typing.

Problem: a RAG index is only meaningful if the vectors in it were produced by
the SAME embedder configuration that a query will be embedded with at serve
time. Silent embedder drift (a model swap, a quantization change, a dropped
L2-normalization step, a truncated dimension) corrupts retrieval *without any
error* -- cosine scores just quietly degrade.

VecStamp writes a small "reproduction certificate" at index-build time: a set of
anchor probe strings, the vectors the build-time embedder produced for them, and
a canonical float hash. At load/serve time it re-embeds the same probes with the
LIVE embedder and checks reproduction within tolerance. When reproduction FAILS,
it does not just say "mismatch" -- it TYPES the failure via a decision tree so an
operator knows *what* changed:

    - dim mismatch        : vector length differs
    - lost L2 norm        : direction preserved but magnitude != 1 (norm step dropped)
    - quant/dtype drift    : same direction, tiny per-element jitter (fp16 -> q8 etc.)
    - wrong weights       : direction materially different (a different model)

The wedge is the TYPING: benign quant/dtype jitter must NOT be reported as
"wrong model", or every quantized redeploy would false-alarm.

Pure Python stdlib. Deterministic in-file fake embedders stand in for real ones.
"""

import hashlib
import json
import math

# --------------------------------------------------------------------------
# Deterministic fake embedders (stand-ins for real embedding models).
# Each maps a string -> a fixed-dim vector. They are deterministic so a
# certificate built now reproduces exactly later under the same embedder.
# --------------------------------------------------------------------------

BASE_DIM = 8


def _raw_features(text, dim=BASE_DIM):
    """Deterministic pseudo-features from text (no randomness, no libs)."""
    feats = []
    for i in range(dim):
        h = hashlib.sha256(("%d|%s" % (i, text)).encode("utf-8")).digest()
        # map first 4 bytes to a float in roughly [-1, 1]
        val = int.from_bytes(h[:4], "big") / 0xFFFFFFFF
        feats.append(val * 2.0 - 1.0)
    return feats


def _l2_normalize(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def embed_fp16(text):
    """Baseline embedder: raw features, L2-normalized (the 'fp16' reference)."""
    return _l2_normalize(_raw_features(text))


def embed_q8(text):
    """Quantized embedder: same model, values rounded to low precision.

    This models fp16 -> int8-ish quantization: direction is essentially
    preserved, only tiny per-element jitter is introduced.
    """
    base = embed_fp16(text)
    # round to ~2 decimals to emulate an 8-bit-ish quantization grid,
    # then RE-normalize as a real quantized pipeline would.
    quant = [round(x, 2) for x in base]
    return _l2_normalize(quant)


def embed_other_model(text):
    """A different model entirely: different feature function -> different dir."""
    feats = []
    for i in range(BASE_DIM):
        h = hashlib.sha256(("model2::%d::%s" % (i, text)).encode("utf-8")).digest()
        val = int.from_bytes(h[4:8], "big") / 0xFFFFFFFF
        feats.append(val * 2.0 - 1.0)
    return _l2_normalize(feats)


def embed_unnormalized(text):
    """Same model as fp16 but the L2-normalization step was dropped."""
    return _raw_features(text)


def embed_wrong_dim(text):
    """Same model as fp16 but truncated to fewer dimensions."""
    return _l2_normalize(_raw_features(text)[: BASE_DIM - 3])


# --------------------------------------------------------------------------
# Canonical float hashing
# --------------------------------------------------------------------------

def _canonical_floats(vec, places=6):
    """Round floats to a canonical string form for stable hashing."""
    return [format(round(x, places), ".%df" % places) for x in vec]


def canonical_hash(vectors, places=6):
    """Order-sensitive hash over a list of probe vectors."""
    payload = json.dumps(
        [_canonical_floats(v, places) for v in vectors], separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Certificate build / load
# --------------------------------------------------------------------------

DEFAULT_PROBES = [
    "the quick brown fox",
    "retrieval augmented generation",
    "vector database certificate",
    "reproduction under tolerance",
    "quantization drift detection",
]


def build_manifest(embed_fn, probes=None, model_id="fp16", hash_places=6):
    """Persist anchor-probe vectors + canonical float hash into a manifest dict."""
    probes = list(probes if probes is not None else DEFAULT_PROBES)
    vectors = [list(embed_fn(p)) for p in probes]
    return {
        "version": 1,
        "model_id": model_id,
        "dim": len(vectors[0]) if vectors else 0,
        "hash_places": hash_places,
        "probes": probes,
        "vectors": vectors,
        "canonical_hash": canonical_hash(vectors, hash_places),
    }


def save_manifest(manifest, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Divergence typing (the novelty wedge)
# --------------------------------------------------------------------------

def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _max_abs_elem_diff(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def type_divergence(stored, live,
                    elem_tol=0.05,
                    cosine_ok=0.9995,
                    cosine_related=0.90,
                    norm_tol=0.02):
    """Classify why a live probe vector diverges from the stored one.

    Decision tree (order matters):
      1. length differs                          -> "dim mismatch"
      2. cosine ~ 1 AND per-elem jitter small    -> reproduction OK (return None-ish "ok")
      3. direction preserved (cosine ~ 1) but
         stored/live norms disagree with each
         other's normalization state             -> "lost L2 norm"
      4. cosine high but not ~1, small jitter     -> "quant/dtype drift"
      5. otherwise (direction changed)            -> "wrong weights"
    """
    if len(stored) != len(live):
        return "dim mismatch"

    cos = _cosine(stored, live)
    stored_norm = math.sqrt(sum(x * x for x in stored))
    live_norm = math.sqrt(sum(x * x for x in live))

    # Direction is essentially identical.
    if cos >= cosine_ok:
        # Same direction. Is it a magnitude/normalization problem?
        stored_normed = abs(stored_norm - 1.0) <= norm_tol
        live_normed = abs(live_norm - 1.0) <= norm_tol
        if stored_normed != live_normed:
            return "lost L2 norm"
        # Both normalized (or both not) + same direction: check jitter.
        if _max_abs_elem_diff(stored, live) <= elem_tol:
            return "ok"
        return "quant/dtype drift"

    # Direction changed but still fairly aligned: quant on a coarse grid can
    # nudge cosine below the strict "ok" bar while staying clearly the same
    # model. Distinguish that from a genuinely different model by jitter size.
    if cos >= cosine_related:
        stored_normed = abs(stored_norm - 1.0) <= norm_tol
        live_normed = abs(live_norm - 1.0) <= norm_tol
        if stored_normed != live_normed:
            return "lost L2 norm"
        if _max_abs_elem_diff(stored, live) <= elem_tol:
            return "quant/dtype drift"
        return "wrong weights"

    # Direction materially different -> a different model's weights.
    return "wrong weights"


def verify(manifest, live_embed_fn, **type_kwargs):
    """Re-embed the manifest probes with the live embedder and certify.

    Returns a dict:
      {"ok": bool, "failure_type": str|None, "per_probe": [types...]}
    A single dominant non-ok type is reported (majority vote across probes).
    """
    stored_vectors = manifest["vectors"]
    probes = manifest["probes"]
    per_probe = []
    for probe, stored in zip(probes, stored_vectors):
        live = list(live_embed_fn(probe))
        per_probe.append(type_divergence(stored, live, **type_kwargs))

    non_ok = [t for t in per_probe if t != "ok"]
    if not non_ok:
        return {"ok": True, "failure_type": None, "per_probe": per_probe}

    # majority vote among non-ok types
    counts = {}
    for t in non_ok:
        counts[t] = counts.get(t, 0) + 1
    dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    return {"ok": False, "failure_type": dominant, "per_probe": per_probe}
