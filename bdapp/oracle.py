"""Answer labelling: does the truncated context flip the answer?

Preferred path: ask a REAL generator on the local gateway the SAME question over
the with-M context vs the split-M context and compare the parsed values.

Fallback path: a deterministic, LABEL-FREE answer-oracle f(context).  It reads
ONLY observable features of the assembled context -- it never consults the hidden
`authentic` flag or the true native value.  It models a self-consistent reader:
among the status chunks that name the seed, it returns the value with the most
support (plurality of chunks asserting it), breaking ties toward the higher
retrieval-relevance value.  If no status chunk names the seed, it answers
"unknown".

A flip is: correct value under split-M, but a *different, definite* value under
with-M.  Whether that happens is decided by the vote among whatever survives the
budget -- an event computed independently of the predictor's dtok/kappa.  The
predictor is therefore tested against a ground truth it does not itself define.
"""
from __future__ import annotations

import method
from corpus import Corpus, STATUS


def oracle_answer(corpus: Corpus, seed: int, ctx: list) -> str:
    """Label-free plurality reader over the assembled context.

    Uses only tokens/relevance of the present chunks -- never the `authentic`
    flag nor corpus.native[seed].value.
    """
    votes = {}
    rel = {}
    for c in ctx:
        if STATUS in c.tokens and f"E{seed}" in c.tokens:
            votes[c.value] = votes.get(c.value, 0) + 1
            rel[c.value] = max(rel.get(c.value, 0.0), method.relevance(c, seed))
    if not votes:
        return "unknown"
    # plurality; ties -> higher retrieval relevance, then value token (determinism)
    return max(votes, key=lambda v: (votes[v], rel[v], v))


def _ctx_text(ctx: list) -> str:
    return "\n".join("- " + " ".join(c.tokens) for c in ctx)


def _gen_answer(corpus, seed, ctx, gw, model, profile=None):
    """Ask the real generator; return parsed value token or None on failure.

    Strict grammar: the parsed answer is one of {v<digits>, "unknown"}. Anything
    else (prose, hedging) is treated as an abstain ("unknown") rather than being
    coerced into a definite value -- coercing arbitrary prose into a value token
    is a fail-open that would inflate the measured flip rate.
    """
    prompt = (
        "Context facts (each line is a retrieved chunk):\n"
        + _ctx_text(ctx)
        + f"\n\nQuestion: What is the {STATUS} of E{seed}? "
        "Reply with ONLY the single value token (e.g. v7). "
        "If the context does not state it, reply exactly: unknown"
    )
    if profile is not None:
        out = gw.chat(model, prompt, timeout=15.0, max_tokens=12, profile=profile)
    else:
        out = gw.chat(model, prompt, timeout=15.0, max_tokens=12)
    if out is None:
        return None
    toks = out.strip().replace(",", " ").split()
    for t in toks:
        t = t.strip(".:`'\"")
        if t == "unknown" or (t.startswith("v") and t[1:].isdigit()):
            return t
    return "unknown"


class FlipLabeler:
    """Labels flips, preferring the real gateway, degrading gracefully."""

    def __init__(self, corpus, gw=None, model=None, max_calls=40, profile=None):
        self.corpus = corpus
        self.gw = gw
        self.model = model
        self.profile = profile              # gateway profile-selection hint (e.g. a thinking-off variant)
        self.max_calls = max_calls          # cap on generator calls
        self.calls = 0
        self.gateway_ok = gw is not None and model is not None
        self.n_real = 0
        self.n_oracle = 0

    def _answer(self, seed, ctx):
        """Return (value, source)."""
        if self.gateway_ok and self.calls < self.max_calls:
            self.calls += 1
            v = _gen_answer(self.corpus, seed, ctx, self.gw, self.model, self.profile)
            if v is not None:
                self.n_real += 1
                return v, "real-gateway"
            # first failure: trip the circuit for the rest of the run
            self.gateway_ok = False
        self.n_oracle += 1
        return oracle_answer(self.corpus, seed, ctx), "fallback-oracle"

    def flip(self, seed, merge, base_active):
        """1 if the merge flips the answer for this seed, else 0. Also the source."""
        merge_edges = set(self.corpus.merges[merge])
        with_m = set(base_active) | merge_edges
        split_m = set(base_active) - merge_edges
        ctx_split = method.assemble(self.corpus, seed, split_m)
        ctx_with = method.assemble(self.corpus, seed, with_m)

        correct = self.corpus.native[seed].value
        a_split, s1 = self._answer(seed, ctx_split)
        a_with, s2 = self._answer(seed, ctx_with)
        if s1 != s2:
            # Gateway tripped BETWEEN the two answers (a_split real, a_with fell
            # back). Comparing a real-model token against an oracle plurality
            # vote is a meaningless cross-source label -- never emit it. Discard
            # the real answer and re-label the pair from the single committed
            # fallback oracle so both contexts share one source. (The only
            # possible mismatch is real->oracle: once the circuit trips it stays
            # tripped, so s2 can never be "real-gateway".)
            self.n_real -= 1        # the real a_split is discarded
            self.n_oracle += 1      # ...replaced by an oracle answer
            a_split = oracle_answer(self.corpus, seed, ctx_split)
            a_with = oracle_answer(self.corpus, seed, ctx_with)
            s1 = s2 = "fallback-oracle"
        flipped = int(a_split == correct and a_with not in (correct, "unknown"))
        return flipped, s1

    def source_summary(self):
        if self.n_real and not self.n_oracle:
            return "real-gateway"
        if self.n_oracle and not self.n_real:
            return "fallback-oracle"
        if self.n_real and self.n_oracle:
            return "mixed"
        return "unknown"
