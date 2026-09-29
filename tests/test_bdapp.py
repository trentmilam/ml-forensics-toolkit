"""Direct-import tests for BD-APP's claimed invariants (no subprocess, no network).

Imports corpus.py/method.py/calibrate.py/repair.py/oracle.py directly and
reconstructs the same fixed-seed pipeline run_experiment.py prints, asserting
the concrete numbers the README claims as measured: the centrality inversion,
the calibration AUC, the displacement-repair win, and the mixed-source-flip
invariant under a flaky gateway.

method.py is loaded via importlib under a unique module name ("bdapp_method")
rather than a bare `import method`, because qslm/method.py has the same filename, and
both test modules load in the same pytest process, so a bare import would risk
one package's tests silently picking up the other package's module from the
shared sys.modules cache.
"""
import importlib.util
import os
import sys

BDAPP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bdapp")
if BDAPP_DIR not in sys.path:
    sys.path.insert(0, BDAPP_DIR)


def _load_module(unique_name, filename):
    path = os.path.join(BDAPP_DIR, filename)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


from corpus import build_corpus  # noqa: E402  (unique name)

method = _load_module("bdapp_method", "method.py")
calibrate = _load_module("bdapp_calibrate", "calibrate.py")
repair = _load_module("bdapp_repair", "repair.py")
repair.method = method  # repair.py does `import method` internally; point it at ours
oracle_mod = _load_module("bdapp_oracle", "oracle.py")
oracle_mod.method = method  # oracle.py does `import method` internally; point it at ours
FlipLabeler = oracle_mod.FlipLabeler

SEED = 20260703
EDIT_BUDGET_K = 6
MERGES = ["M_peripheral", "M_central", "M_distract"]


def _label_free_labeler(corpus):
    """The deterministic label-free reader: gw=None never touches the network."""
    return FlipLabeler(corpus, gw=None, model=None, max_calls=40)


def test_centrality_inversion():
    corpus = build_corpus(SEED)
    ALL = set(corpus.edges.keys())
    labeler = _label_free_labeler(corpus)

    rows = {}
    for m in MERGES:
        pr = method.PR(corpus, m, ALL, method.kappa_lexical)
        cent = method.centrality_of_merge(corpus, m)
        seeds = method.affected_seeds(corpus, m)
        flips = sum(labeler.flip(s, m, ALL)[0] for s in seeds)
        harm = flips / len(seeds)
        rows[m] = (pr, cent, harm)

    pr_p = {m: rows[m][0] for m in MERGES}
    cent_p = {m: rows[m][1] for m in MERGES}
    harm_p = {m: rows[m][2] for m in MERGES}
    assert pr_p["M_peripheral"] > pr_p["M_central"]
    assert cent_p["M_peripheral"] < cent_p["M_central"]
    assert harm_p["M_peripheral"] > harm_p["M_central"]


def test_calibration_auc_near_perfect():
    corpus = build_corpus(SEED)
    ALL = set(corpus.edges.keys())
    labeler = _label_free_labeler(corpus)

    cal_seeds = []
    for m in MERGES:
        cs = method.affected_seeds(corpus, m)
        take = cs if m == "M_peripheral" else cs[:4]
        cal_seeds += [(s, m) for s in take]
    cal_seeds = cal_seeds[:20]

    ds, ys = [], []
    for s, m in cal_seeds:
        contrib, *_ = method.per_seed_displacement(corpus, s, m, ALL, method.kappa_lexical)
        f, _ = labeler.flip(s, m, ALL)
        ds.append(contrib)
        ys.append(f)

    auc = calibrate.auc(ds, ys)
    assert auc > 0.99, f"calibration AUC {auc} not > 0.99"


def test_displacement_repair_beats_centrality_and_dedup():
    corpus = build_corpus(SEED)
    ALL = set(corpus.edges.keys())
    labeler = _label_free_labeler(corpus)

    def answer_fn(seed, ctx):
        v, _ = labeler._answer(seed, ctx)
        return v

    at_risk = [s for s in method.affected_seeds(corpus, "M_peripheral")
               if labeler.flip(s, "M_peripheral", ALL)[0] == 1]
    assert at_risk, "baseline flips must exist for the repair comparison to mean anything"

    results = {}
    for strat in ("displacement", "centrality", "dedup"):
        cut = repair.choose_cut(corpus, ALL, EDIT_BUDGET_K, strat, method.kappa_lexical)
        active = ALL - cut
        flips, prec = repair.evaluate(corpus, active, at_risk, answer_fn)
        results[strat] = (flips, prec)

    d_flips, d_prec = results["displacement"]
    c_flips, c_prec = results["centrality"]
    dd_flips, _dd_prec = results["dedup"]

    assert d_flips < c_flips
    assert d_flips < dd_flips
    assert d_prec > c_prec


def test_redcase_no_mixed_source_flip_on_flaky_gateway():
    """A gateway that answers once then trips mid-pair must never emit a
    cross-source ('mixed') comparison. This uses an in-process fake gateway, offline."""
    class FlakyGW:
        def __init__(self):
            self.n = 0

        def chat(self, model, prompt, timeout=15.0, max_tokens=12):
            self.n += 1
            return "v0" if self.n == 1 else None

    corpus = build_corpus(SEED)
    lab = FlipLabeler(corpus, gw=FlakyGW(), model="flaky", max_calls=40)
    m = "M_peripheral"
    s = method.affected_seeds(corpus, m)[0]
    _flipped, src = lab.flip(s, m, set(corpus.edges.keys()))
    assert src != "mixed"
