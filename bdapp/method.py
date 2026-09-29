"""BD-APP core: token-budget assembler + generator-free displacement predictor.

Independent claim (method): a poisoning-risk score for a GraphRAG false-merge M
computed WITHOUT any generator, as

    PR(M) = sum_e  s(e) * r(e->M) * dtok(e,M) * kappa(e,M)

where dtok(e,M) is the count of on-entity, query-relevant tokens that are present
in the counterfactual split-M truncated context but EVICTED, in the with-M
truncated context, by foreign content routed through M under a fixed token budget,
and kappa(e,M) is a deterministic directional-contradiction proxy.  The score is
obtained by running the assembler TWICE per affected seed (with-M vs split-M) and
diffing the two truncated contexts. No model call is required.
"""
from __future__ import annotations
from collections import deque
import numpy as np

from corpus import Corpus, Chunk, STATUS


# ---------------------------------------------------------------------------
# relevance + budget assembler
# ---------------------------------------------------------------------------
def query_keywords(seed: int):
    return {f"E{seed}", STATUS}


def relevance(chunk: Chunk, seed: int) -> float:
    kw = query_keywords(seed)
    overlap = len(kw & set(chunk.tokens))
    return overlap + chunk.boost


def candidate_chunks(corpus: Corpus, seed: int, active_edges) -> list:
    cands = [corpus.native[seed]] + list(corpus.neighbors.get(seed, []))
    for eid in active_edges:
        e = corpus.edges[eid]
        if e.seed == seed:
            cands.append(e.foreign)
    return cands


def assemble(corpus: Corpus, seed: int, active_edges) -> list:
    """Relevance-greedy truncation under the seed's fixed token budget.

    Returns the list of INCLUDED chunks (the truncated context).
    """
    budget = corpus.budget[seed]
    cands = candidate_chunks(corpus, seed, active_edges)
    # sort by relevance desc, deterministic tiebreak by cid
    order = sorted(cands, key=lambda c: (-relevance(c, seed), c.cid))
    included, used = [], 0
    for c in order:
        if used + c.size <= budget:
            included.append(c)
            used += c.size
    return included


# ---------------------------------------------------------------------------
# on-entity query-relevant tokens
# ---------------------------------------------------------------------------
def on_entity_relevant_tokens(corpus: Corpus, seed: int) -> set:
    """The authentic, query-relevant tokens the native chunk load-bears."""
    nat = corpus.native[seed]
    kw = query_keywords(seed)
    toks = (kw & set(nat.tokens)) | {nat.value}
    return toks


# ---------------------------------------------------------------------------
# displacement dtok(e, M): generator-free, by diffing two contexts
# ---------------------------------------------------------------------------
def dtok(corpus: Corpus, seed: int, merge: str, base_active) -> int:
    """Query-relevant on-entity tokens present under split-M but evicted with-M."""
    merge_edges = set(corpus.merges[merge])
    with_m = set(base_active) | merge_edges
    split_m = set(base_active) - merge_edges

    ctx_split = assemble(corpus, seed, split_m)
    ctx_with = assemble(corpus, seed, with_m)

    want = on_entity_relevant_tokens(corpus, seed)
    present_split = {t for t in want if any(t in c.tokens for c in ctx_split)}
    present_with = {t for t in want if any(t in c.tokens for c in ctx_with)}
    evicted = present_split - present_with
    return len(evicted)


# ---------------------------------------------------------------------------
# kappa(e, M): directional contradiction proxy (lexical fallback)
# ---------------------------------------------------------------------------
def kappa_lexical(corpus: Corpus, seed: int, merge: str) -> float:
    """1.0 if a foreign chunk of M asserts a DIFFERENT value for the SAME
    (queried) attribute of the seed; else 0.0."""
    nat = corpus.native[seed]
    for eid in corpus.merges[merge]:
        e = corpus.edges[eid]
        if e.seed != seed:
            continue
        f = e.foreign
        if f.attribute == nat.attribute and f.value != nat.value:
            return 1.0
    return 0.0


def kappa_embedding(corpus, seed, merge, gw, emb_model, cache):
    """Optional embedding-based directional contradiction. Returns None if the
    gateway embeddings endpoint is unavailable so the caller can fall back."""
    nat = corpus.native[seed]
    foreign = None
    for eid in corpus.merges[merge]:
        e = corpus.edges[eid]
        if e.seed == seed and e.foreign.attribute == nat.attribute:
            foreign = e.foreign
            break
    if foreign is None:
        return 0.0
    texts = [" ".join(nat.tokens), " ".join(foreign.tokens)]
    key = tuple(texts)
    if key in cache:
        vecs = cache[key]
    else:
        vecs = gw.embed(emb_model, texts)
        cache[key] = vecs
    if not vecs:
        return None
    a = np.asarray(vecs[0], float)
    b = np.asarray(vecs[1], float)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    # high textual similarity + differing value token => directional contradiction
    same_attr_diff_val = (foreign.attribute == nat.attribute and foreign.value != nat.value)
    return float(cos * (1.0 if same_attr_diff_val else 0.0))


# ---------------------------------------------------------------------------
# seeding weight s(e) and expansion reach r(e->M)
# ---------------------------------------------------------------------------
def s_weight(corpus: Corpus, seed: int) -> float:
    """Query-log-free seeding: each chunk is a surrogate query. s(e) is the
    number of surrogate queries that name e (its own native chunk + any chunk
    labelled about e)."""
    n = 1  # own native chunk
    for c in corpus.neighbors.get(seed, []):
        if c.owner == seed:
            n += 1
    return float(n)


def _bfs_dist(adj, src, dst):
    if src == dst:
        return 0
    seen = {src}
    q = deque([(src, 0)])
    while q:
        u, d = q.popleft()
        for v in adj.get(u, ()):
            if v == dst:
                return d + 1
            if v not in seen:
                seen.add(v)
                q.append((v, d + 1))
    return None


def r_reach(corpus: Corpus, seed: int, merge: str, decay: float = 0.8) -> float:
    """Expansion reach from seed to the merge anchor: decay ** graph-distance."""
    anchor = corpus.edges[corpus.merges[merge][0]].anchor
    d = _bfs_dist(corpus.adj, seed, anchor)
    if d is None:
        d = 6  # unreachable in legit graph -> heavy decay
    return decay ** d


# ---------------------------------------------------------------------------
# poisoning-risk PR(M)  (the predictor)  +  affected-seed set
# ---------------------------------------------------------------------------
def affected_seeds(corpus: Corpus, merge: str):
    return [corpus.edges[eid].seed for eid in corpus.merges[merge]]


def per_seed_displacement(corpus, seed, merge, base_active, kappa_fn):
    d = dtok(corpus, seed, merge, base_active)
    k = kappa_fn(corpus, seed, merge)
    s = s_weight(corpus, seed)
    r = r_reach(corpus, seed, merge)
    return s * r * d * k, d, k, s, r


def PR(corpus: Corpus, merge: str, base_active, kappa_fn) -> float:
    total = 0.0
    for seed in affected_seeds(corpus, merge):
        contrib, *_ = per_seed_displacement(corpus, seed, merge, base_active, kappa_fn)
        total += contrib
    return total


# ---------------------------------------------------------------------------
# baseline signals the defender might use instead
# ---------------------------------------------------------------------------
def degree_centrality(corpus: Corpus) -> dict:
    n = len(corpus.entities)
    return {i: len(corpus.adj[i]) / (n - 1) for i in corpus.entities}


def centrality_of_merge(corpus: Corpus, merge: str) -> float:
    """Realistic centrality defence: score the merge's actual injection/seed
    sites (mean degree-centrality over the poisoned seeds), NOT a decorative
    anchor label."""
    dc = degree_centrality(corpus)
    seeds = [corpus.edges[eid].seed for eid in corpus.merges[merge]]
    return sum(dc[s] for s in seeds) / len(seeds)


def dedup_confidence_edge(corpus: Corpus, eid: str) -> float:
    """A dedup system's confidence that the merged pair is truly one entity:
    Jaccard token similarity between the seed's native chunk and the foreign
    chunk.  Identity-conflated foreign chunks score HIGH (dedup trusts them)."""
    e = corpus.edges[eid]
    nat = set(corpus.native[e.seed].tokens)
    fgn = set(e.foreign.tokens)
    return len(nat & fgn) / len(nat | fgn)
