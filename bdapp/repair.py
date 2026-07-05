"""Displacement-weighted min-cut repair vs. centrality-cut and dedup-cut.

All three strategies remove exactly K foreign-routing edges (a fixed edit budget).
They differ only in HOW they choose which edges to cut:

  * displacement : edge weight = per-unit load-bearing displacement (dtok * kappa);
                   cut the K highest -> a min-cut of the poisoning flow.
  * centrality   : cut the K edges whose merge sits at the most central anchor.
  * dedup        : cut the K edges the dedup system is LEAST confident about
                   (lowest native/foreign token similarity).

After cutting, we re-measure the answer-flip count and the on-entity
truncated-context precision over the at-risk seeds.
"""
from __future__ import annotations
import numpy as np

import method
from corpus import Corpus


def _all_edges(corpus: Corpus):
    return list(corpus.edges.keys())


def displacement_weights(corpus, base_active, kappa_fn):
    w = {}
    for eid, e in corpus.edges.items():
        d = method.dtok(corpus, e.seed, e.merge, base_active)
        k = kappa_fn(corpus, e.seed, e.merge)
        w[eid] = d * k
    return w


def choose_cut(corpus, base_active, K, strategy, kappa_fn):
    edges = _all_edges(corpus)
    if strategy == "displacement":
        w = displacement_weights(corpus, base_active, kappa_fn)
        key = lambda eid: (-w[eid], eid)          # highest load-bearing displacement first
    elif strategy == "centrality":
        cent = method.degree_centrality(corpus)
        key = lambda eid: (-cent[corpus.edges[eid].seed], eid)   # score real seed site
    elif strategy == "dedup":
        key = lambda eid: (method.dedup_confidence_edge(corpus, eid), eid)  # least confident first
    else:
        raise ValueError(strategy)
    return set(sorted(edges, key=key)[:K])


def _relevant_tokens(chunk, seed):
    kw = method.query_keywords(seed)
    return {t for t in chunk.tokens if t in kw or t.startswith("v")}


def on_entity_precision(corpus, seed, ctx):
    num = 0
    den = 0
    for c in ctx:
        rel = _relevant_tokens(c, seed)
        den += len(rel)
        if c.authentic and c.owner == seed:
            num += len(rel)
    return (num / den) if den else 0.0


def evaluate(corpus, active_edges, at_risk_seeds, answer_fn):
    """Return (residual_flip_count, mean_on_entity_precision) over at_risk_seeds."""
    flips = 0
    precs = []
    for seed in at_risk_seeds:
        ctx = method.assemble(corpus, seed, active_edges)
        ans = answer_fn(seed, ctx)
        correct = corpus.native[seed].value
        if ans not in (correct, "unknown"):
            flips += 1
        precs.append(on_entity_precision(corpus, seed, ctx))
    return flips, float(np.mean(precs)) if precs else 0.0
