"""Deadstage - a judge-free RAG liveness gate that names the dead stage.

A RAG pipeline can silently die at exactly one stage while every other stage
happily forwards garbage: an empty index still "retrieves" [], zero-norm
embeddings still "score" (as NaN or a flat 0), and the final answer is
plausible-looking mush. Most eval harnesses only tell you the *end* score is
bad. Deadstage instead asserts per-stage STRUCTURAL invariants on the actual
intermediate outputs and prints the SINGLE named stage that is dead.

It is judge-free: no LLM, no embedding model, no ground-truth labels. It only
inspects the shape/statistics of what each stage emitted. It asserts and
reports; it never masks, repairs, or substitutes a "safe" default.

Pipeline stages (fixed order):
    ingest  -> index -> embed -> retrieve -> score

Public API:
    PipelineOutputs(...)   - a plain container of per-stage outputs
    check(outputs)         - returns a Report; does not exit
    main(...)              - CLI-style: prints the report, exits nonzero on death
"""

import math
import sys


STAGES = ("ingest", "index", "embed", "retrieve", "score")


class PipelineOutputs:
    """Container for the observed output of each pipeline stage.

    Every field is exactly what the corresponding stage produced. Deadstage
    inspects these; it never fabricates them. Fields left as None are treated
    as "stage did not run" and are reported as such.

    - ingest:   list of ingested document chunks (list of str)
    - index:    list of indexed entries (any objects; emptiness is what matters)
    - embed:    list of embedding vectors (list of list[float])
    - retrieve: list of retrieved hits for the query (any objects)
    - score:    list of relevance scores (list of float), one per retrieved hit
    """

    def __init__(self, ingest=None, index=None, embed=None,
                 retrieve=None, score=None):
        self.ingest = ingest
        self.index = index
        self.embed = embed
        self.retrieve = retrieve
        self.score = score


class Report:
    """Result of a check. `dead_stage` is None iff the pipeline is alive."""

    def __init__(self, dead_stage=None, reason=None, detail=None):
        self.dead_stage = dead_stage
        self.reason = reason      # short machine-ish tag, e.g. "collapsed/zero-norm"
        self.detail = detail      # human-readable one-liner

    @property
    def alive(self):
        return self.dead_stage is None

    def label(self):
        """The canonical 'stage: reason' string, e.g. 'embed: collapsed/zero-norm'."""
        if self.alive:
            return "alive"
        return "{}: {}".format(self.dead_stage, self.reason)

    def render(self):
        if self.alive:
            return "LIVE: all stages structurally alive"
        return "DEAD STAGE -> {}\n  {}".format(self.label(), self.detail or "")


# --- per-stage structural invariants -------------------------------------
#
# Each checker returns (reason, detail) if the stage is DEAD, else None.
# They inspect only structure/statistics; they never mutate the outputs.


def _vec_norm(vec):
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def _check_ingest(out):
    if out is None:
        return ("not-run", "ingest produced no output (None)")
    if len(out) == 0:
        return ("empty", "ingest returned 0 chunks; nothing entered the pipeline")
    return None


def _check_index(out):
    if out is None:
        return ("not-run", "index produced no output (None)")
    if len(out) == 0:
        return ("empty", "index is empty; retrieval can only ever return []")
    return None


def _check_embed(out):
    if out is None:
        return ("not-run", "embed produced no output (None)")
    if len(out) == 0:
        return ("empty", "embed returned 0 vectors")

    # Reject NaN/inf components outright.
    for i, vec in enumerate(out):
        if len(vec) == 0:
            return ("empty-vector",
                    "embedding {} has zero dimensions".format(i))
        for x in vec:
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return ("nan",
                        "embedding {} contains NaN/inf component".format(i))

    # Zero-norm collapse: every vector has (near) zero magnitude.
    norms = [_vec_norm(v) for v in out]
    if all(n <= 1e-12 for n in norms):
        return ("collapsed/zero-norm",
                "all {} embeddings have ~zero norm; the embedder emitted "
                "zero-vectors".format(len(out)))

    # All-identical collapse: distinct inputs mapped to one point => no signal.
    if len(out) > 1:
        first = [float(x) for x in out[0]]
        same = True
        for vec in out[1:]:
            if len(vec) != len(first):
                same = False
                break
            for a, b in zip(first, vec):
                if abs(float(a) - float(b)) > 1e-12:
                    same = False
                    break
            if not same:
                break
        if same:
            return ("collapsed/all-identical",
                    "all {} embeddings are identical; the embedder carries no "
                    "signal".format(len(out)))
    return None


def _check_retrieve(out):
    if out is None:
        return ("not-run", "retrieve produced no output (None)")
    if len(out) == 0:
        return ("empty", "retrieve returned []; no candidates for scoring")
    return None


def _check_score(out):
    if out is None:
        return ("not-run", "score produced no output (None)")
    if len(out) == 0:
        return ("empty", "score returned []; nothing was ranked")

    vals = []
    for i, s in enumerate(out):
        sf = float(s)
        if math.isnan(sf) or math.isinf(sf):
            return ("nan", "score {} is NaN/inf".format(i))
        vals.append(sf)

    # Zero-spread: every score identical => ranking is meaningless.
    if len(vals) > 1:
        spread = max(vals) - min(vals)
        if spread <= 1e-12:
            return ("zero-spread",
                    "all {} scores are identical ({}); ranking carries no "
                    "signal".format(len(vals), vals[0]))
    return None


_CHECKERS = {
    "ingest": _check_ingest,
    "index": _check_index,
    "embed": _check_embed,
    "retrieve": _check_retrieve,
    "score": _check_score,
}


def check(outputs):
    """Return a Report naming the FIRST dead stage (upstream wins).

    Upstream-first is deliberate: a dead upstream stage causes downstream
    deadness, so the root cause is the earliest failing stage. Deadstage
    reports exactly one stage - the one to fix.
    """
    if not isinstance(outputs, PipelineOutputs):
        raise TypeError("check() expects a PipelineOutputs instance")
    for stage in STAGES:
        result = _CHECKERS[stage](getattr(outputs, stage))
        if result is not None:
            reason, detail = result
            return Report(dead_stage=stage, reason=reason, detail=detail)
    return Report()


def main(outputs, stream=None):
    """Print the report and return an exit code (0 alive, 1 dead)."""
    stream = stream or sys.stdout
    report = check(outputs)
    stream.write(report.render() + "\n")
    return 0 if report.alive else 1


if __name__ == "__main__":
    # Demo self-run with a healthy pipeline.
    demo = PipelineOutputs(
        ingest=["doc a", "doc b"],
        index=["e0", "e1"],
        embed=[[0.1, 0.9], [0.8, 0.2]],
        retrieve=["e1", "e0"],
        score=[0.91, 0.42],
    )
    sys.exit(main(demo))
