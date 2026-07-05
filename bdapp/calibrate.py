"""Monotone calibration of the per-case displacement score d = s*r*dtok*kappa
to an empirical flip probability P(flip), via isotonic regression (PAVA).

No sklearn/scipy: PAVA is implemented directly in numpy.
"""
from __future__ import annotations
import numpy as np


def isotonic_fit(x, y):
    """Non-decreasing isotonic regression of y on x (Pool Adjacent Violators).
    Returns (xs_sorted, ys_fitted)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    # PAVA
    vals = list(ys)
    wts = [1.0] * len(ys)
    i = 0
    blocks = [[v, w] for v, w in zip(vals, wts)]
    merged = []
    for v, w in blocks:
        merged.append([v, w])
        while len(merged) > 1 and merged[-2][0] > merged[-1][0]:
            v2, w2 = merged.pop()
            v1, w1 = merged.pop()
            nw = w1 + w2
            merged.append([(v1 * w1 + v2 * w2) / nw, nw])
    # expand back
    fitted = []
    for v, w in merged:
        fitted.extend([v] * int(round(w)))
    fitted = np.asarray(fitted[: len(ys)], float)
    return xs, fitted


def predict(xs, ys_fitted, q):
    """Step interpolation of the isotonic map at query point q."""
    idx = np.searchsorted(xs, q, side="right") - 1
    idx = int(np.clip(idx, 0, len(ys_fitted) - 1))
    return float(ys_fitted[idx])


def auc(scores, labels):
    """Rank-based AUC (Mann-Whitney). Ties get 0.5 credit."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    wins = 0.0
    for p in pos:
        wins += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return wins / (len(pos) * len(neg))
