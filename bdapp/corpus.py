"""Synthetic GraphRAG corpus with injected false-merges.

The corpus is a set of ENTITIES, each owning a native CHUNK that load-bears the
answer to a status query.  Each entity's true status value is *corroborated* by a
second small legitimate chunk (a real corpus rarely states a fact exactly once),
plus two lower-relevance neighbour chunks and a legitimate knowledge GRAPH.

Three false-merges are injected, each modelled as a bundle of foreign-content-
routing EDGES:

  * M_peripheral : anchored/seeded at LOW-centrality (leaf) entities, but each of
                   its edges routes a HEAVY, retrieval-optimized (high-relevance),
                   identity-conflated, CONTRADICTORY foreign chunk.  On tight
                   budgets it is assembled first and EVICTS the native + its
                   corroborator, so only the wrong value survives.  On a few roomy
                   budgets it co-locates with the correct evidence and is
                   out-voted -- same merge, flip only where displacement happens.
  * M_central    : anchored/seeded at HIGH-centrality (hub) entities, but its
                   edges route a LIGHT contradictory chunk that fits WITHOUT
                   evicting the native+corroborator, so the correct value keeps
                   the plurality -> no flip despite the contradiction co-locating.
  * M_distract   : routes HEAVY foreign content that EVICTS on-entity tokens but
                   asserts a DIFFERENT attribute (location, kappa == 0) -->
                   genuine displacement with no contradictory status value, so the
                   reader answers "unknown", not a flip.  Separates the
                   displacement-only signal from the contradiction signal.

Ground truth is produced by a label-free plurality reader (see oracle.py); the
`authentic` flag below is NEVER read by the reader -- it is used only for the
on-entity-precision metric in repair.py.  Everything is deterministic given SEED.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

N_ENTITIES = 40
STATUS = "status"


@dataclass
class Chunk:
    cid: str
    owner: int            # entity id this chunk is *labelled* about
    attribute: str        # "status" or "location"
    value: str            # asserted value token
    authentic: bool       # True only for an entity's real native chunk (metric-only)
    tokens: list          # whitespace token list
    boost: float          # retrieval-relevance boost (identity-conflation inflation)

    @property
    def size(self) -> int:
        return len(self.tokens)


@dataclass
class Edge:
    eid: str
    merge: str            # which merge this edge belongs to
    anchor: int           # merge's graph position (for the reach weight r only)
    seed: int             # affected seed entity (whose query is poisoned)
    foreign: Chunk        # foreign chunk routed to the seed


@dataclass
class Corpus:
    entities: list
    native: dict          # entity id -> Chunk
    neighbors: dict       # seed id -> [Chunk, ...]  corroborator + legit context
    edges: dict           # eid -> Edge
    adj: dict             # entity id -> set(entity id)  legitimate graph
    budget: dict          # seed id -> token budget
    merges: dict          # merge name -> [eid, ...]


def _tok(owner, attr, value, filler_n):
    return [f"E{owner}", attr, "is", value] + [f"f{k}" for k in range(filler_n)]


def build_corpus(seed: int = 20260703) -> Corpus:
    rng = np.random.default_rng(seed)

    # --- entities + native chunks ---------------------------------------
    entities = list(range(N_ENTITIES))
    native = {}
    for i in entities:
        native[i] = Chunk(
            cid=f"nat{i}", owner=i, attribute=STATUS, value=f"v{i}",
            authentic=True, tokens=_tok(i, STATUS, f"v{i}", 0), boost=0.0,
        )

    # --- legitimate knowledge graph -------------------------------------
    # Built so that the SEED SITES (what a realistic centrality defence scores)
    # invert against harm: the harmful merge sits on leaves, the harmless one on
    # hubs.  degree_centrality is measured on this real graph.
    adj = {i: set() for i in entities}

    def link(a, b):
        adj[a].add(b)
        adj[b].add(a)

    # central seeds 1..12: hub-connected (degree 3) -> HIGH centrality
    for s in range(1, 13):
        link(0, s)
        link(37, s)
        link(38, s)
    # distract seeds 13..20: a chain (degree ~2) -> MID centrality
    for i in range(13, 21):
        link(i, i - 1 if i > 13 else 39)
    link(20, 21)
    link(21, 22)
    # peripheral seeds 25..36: pure leaves off a single non-seed hub 23 ->
    # each SEED has degree 1 -> LOW centrality (23 itself is not a seed)
    for s in range(25, 37):
        link(23, s)
    # a sparse random backbone among the remaining structural nodes so degrees
    # of non-seed nodes vary (does not touch seed centrality)
    for i in (24, 39):
        k = int(rng.integers(0, 24))
        link(i, k)

    # --- seed sets ------------------------------------------------------
    central_seeds = list(range(1, 13))               # 12, high centrality
    distract_seeds = list(range(13, 21))             # 8,  mid centrality
    peripheral_seeds = list(range(25, 37))           # 12, low centrality
    peri_roomy = {25, 26, 27}                        # co-location, no eviction
    # tight (evicting) peripheral seeds = the other 9

    # --- seed budgets ---------------------------------------------------
    budget = {i: 8 for i in entities}
    for s in central_seeds:
        budget[s] = 10                               # fits native+corr+light foreign
    for s in peripheral_seeds:
        budget[s] = 8                                # tight: heavy foreign evicts
    for s in peri_roomy:
        budget[s] = 15                               # roomy: heavy foreign co-locates

    neighbors = {i: [] for i in entities}
    edges = {}
    merges = {"M_peripheral": [], "M_central": [], "M_distract": []}

    def add_edge(merge, anchor, seed, foreign):
        eid = f"{merge}:{seed}"
        edges[eid] = Edge(eid=eid, merge=merge, anchor=anchor, seed=seed, foreign=foreign)
        merges[merge].append(eid)

    # legit context for a seed: a same-value CORROBORATOR (the fact stated twice
    # in the corpus) + two lower-relevance neighbour chunks about other entities.
    def seed_context(seed):
        corr = Chunk(
            cid=f"corr{seed}", owner=seed, attribute=STATUS, value=f"v{seed}",
            authentic=False, tokens=[f"E{seed}", STATUS, f"v{seed}"], boost=0.0,
        )
        nbs = sorted(adj[seed])[:2]
        while len(nbs) < 2:
            nbs.append((seed + len(nbs) + 1) % N_ENTITIES)
        chunks = [corr]
        for t, nb in enumerate(nbs):
            chunks.append(Chunk(
                cid=f"nb{seed}_{t}", owner=nb, attribute=STATUS, value=f"v{nb}",
                authentic=False, tokens=[f"E{nb}", STATUS, "is"], boost=0.0,
            ))
        return chunks

    # --- M_peripheral: leaf seeds, heavy conflated contradiction --------
    for s in peripheral_seeds:
        neighbors[s] = seed_context(s)
        j = (s + 17) % N_ENTITIES         # a different entity's (wrong) value
        foreign = Chunk(
            cid=f"peri_f{s}", owner=s, attribute=STATUS, value=f"v{j}",
            authentic=False, tokens=_tok(s, STATUS, f"v{j}", 4), boost=0.5,
        )
        add_edge("M_peripheral", 23, s, foreign)

    # --- M_central: hub seeds, light non-displacing contradiction -------
    for s in central_seeds:
        neighbors[s] = seed_context(s)
        j = (s + 17) % N_ENTITIES
        foreign = Chunk(
            cid=f"cent_f{s}", owner=s, attribute=STATUS, value=f"v{j}",
            authentic=False, tokens=[f"E{s}", STATUS, f"v{j}"], boost=0.0,
        )
        add_edge("M_central", 0, s, foreign)

    # --- M_distract: heavy eviction, but off-attribute (kappa == 0) -----
    for s in distract_seeds:
        neighbors[s] = seed_context(s)
        foreign = Chunk(
            cid=f"dist_f{s}", owner=s, attribute="location", value=f"loc{s}",
            authentic=False, tokens=_tok(s, "location", f"loc{s}", 4), boost=1.5,
        )
        add_edge("M_distract", 20, s, foreign)

    return Corpus(entities, native, neighbors, edges, adj, budget, merges)
