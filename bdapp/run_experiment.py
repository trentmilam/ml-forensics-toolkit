"""BD-APP reduction-to-practice experiment.

Runs entirely offline against a deterministic oracle if the local gateway
generator is unavailable, and records which source produced the flip labels.

Prints, all MEASURED:
  (1) the centrality-inversion: PR ranks M_peripheral above M_central while a
      degree-centrality baseline ranks them oppositely, and the ground-truth
      answer-flip rate agrees with PR, not centrality;
  (2) the repair: displacement-weighted min-cut reduces the answer-flip rate and
      raises on-entity context precision more than centrality-cut or dedup-cut at
      an equal edit budget.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import gateway
import method
import calibrate
import repair
from corpus import build_corpus
from oracle import FlipLabeler, oracle_answer, _gen_answer

SEED = 20260703
# Live gateway model preference + thinking-off profile name are environment-configurable so this
# script carries no hardcoded reference to any particular deployment's model ids. Set
# BDAPP_CHAT_MODELS (comma-separated) to match your own gateway; an empty preference falls back to
# trying every model /v1/models returns (see probe_gateway below), so this is safe to leave unset.
CHAT_MODEL_PREF = [m for m in os.environ.get("BDAPP_CHAT_MODELS", "").split(",") if m]
# Some local gateways serve a THINKING model by default whose default persona spends its whole
# token budget on hidden reasoning and returns EMPTY content. If yours does, set BDAPP_CHAT_PROFILE
# to a thinking-off profile/variant name it recognizes to get a usable single-token answer. Leave
# unset (None) to reproduce the naive-integration failure (empty answers -> loop cannot close).
CHAT_PROFILE = os.environ.get("BDAPP_CHAT_PROFILE") or None
EMB_MODEL = "nomic-embed"
MAX_GEN_CALLS = 40          # hard cap on generator calls
EDIT_BUDGET_K = 6           # fixed repair edit budget (edges)

BAR = "=" * 70


def redcase_no_mixed_source_flip(corpus):
    """RED (mixed-source flip invariant): a gateway that answers the FIRST
    context of a flip() then trips before the SECOND must never emit a
    cross-source ("mixed") comparison: a real-model token compared against an
    oracle plurality vote is a meaningless label. The pair must be single-source.

    Uses a flaky in-process fake gateway (offline, deterministic): success once,
    then failure, forcing a trip *between* a_split and a_with within one flip().
    """
    class FlakyGW:
        def __init__(self):
            self.n = 0

        def chat(self, model, prompt, timeout=15.0, max_tokens=12):
            self.n += 1
            return "v0" if self.n == 1 else None  # succeed once, then trip

    lab = FlipLabeler(corpus, gw=FlakyGW(), model="flaky", max_calls=40)
    m = "M_peripheral"
    s = method.affected_seeds(corpus, m)[0]
    _flipped, src = lab.flip(s, m, set(corpus.edges.keys()))
    ok = src != "mixed"          # never a cross-source comparison
    return ok, src


def _batch_pairs(corpus):
    """A handful of (seed, merge) pairs spanning all three merges."""
    pairs = []
    for m in ("M_peripheral", "M_central", "M_distract"):
        for s in method.affected_seeds(corpus, m)[:2]:
            pairs.append((s, m))
    return pairs[:6]


class _FixedGW:
    """Adapter that pins a profile + token budget onto gateway.chat, so the SAME
    parsing path can be exercised under two integration recipes for a fair A/B."""

    def __init__(self, profile, max_tokens):
        self.profile = profile
        self.max_tokens = max_tokens

    def chat(self, model, prompt, timeout=15.0, max_tokens=12):
        return gateway.chat(model, prompt, timeout=timeout,
                            max_tokens=self.max_tokens, profile=self.profile)


def real_llm_ab(corpus, chat_model):
    """Close the real-LLM loop with a MEASURED head-to-head against the naive
    integration a competent engineer would try first.

    NAIVE (incumbent): call the OpenAI-compatible endpoint the obvious way:
    default persona (thinking on), tight token budget (max_tokens=6). This is
    exactly what the original probe did. On a thinking model it returns EMPTY
    content, so 0 usable value tokens are produced and the flip loop silently
    falls back to the oracle (generator_source=fallback-oracle despite a live
    gateway). That is the defect this fix removes.

    CORRECTED: request the thinking-off profile (CHAT_PROFILE) with a small
    budget, which yields clean single value tokens, so the loop closes with
    generator_source=real-gateway.

    Returns a dict of MEASURED datapoints.
    """
    ALL = set(corpus.edges.keys())
    pairs = _batch_pairs(corpus)
    naive_gw = _FixedGW(profile=None, max_tokens=6)          # default persona
    corr_gw = _FixedGW(profile=CHAT_PROFILE, max_tokens=12)  # thinking-off

    naive_usable = corr_usable = 0
    for s, m in pairs:
        ctx = method.assemble(corpus, s, ALL | set(corpus.merges[m]))
        if _gen_answer(corpus, s, ctx, naive_gw, chat_model) is not None:
            naive_usable += 1
        if _gen_answer(corpus, s, ctx, corr_gw, chat_model) is not None:
            corr_usable += 1

    # Closed-loop headline datapoint: run the flip labeller against the REAL
    # generator (thinking-off) and cross-check its flip labels against the
    # deterministic label-free oracle on the SAME pairs.
    real_lab = FlipLabeler(corpus, gw=gateway, model=chat_model,
                           max_calls=2 * len(pairs) + 4, profile=CHAT_PROFILE)
    orac_lab = FlipLabeler(corpus, gw=None, model=None)
    agree = 0
    for s, m in pairs:
        rf, _ = real_lab.flip(s, m, ALL)
        of, _ = orac_lab.flip(s, m, ALL)
        agree += int(rf == of)
    loop_closed = (real_lab.n_real > 0
                   and real_lab.n_oracle == 0
                   and real_lab.source_summary() == "real-gateway")
    return {
        "pairs": len(pairs),
        "naive_usable": naive_usable,
        "corr_usable": corr_usable,
        "n_real": real_lab.n_real,
        "n_oracle": real_lab.n_oracle,
        "source": real_lab.source_summary(),
        "agree": agree,
        "loop_closed": loop_closed,
    }


def probe_gateway():
    models = gateway.list_models(timeout=8.0)
    chat_model, chat_ok = None, False
    if models:
        cands = [m for m in CHAT_MODEL_PREF if m in models] or list(models)
        # pick the first candidate that ACTUALLY generates a non-empty answer
        # under the configured profile (an empty thinking-only reply is a miss).
        for m in cands:
            out = gateway.chat(m, "Reply with only: OK", timeout=20.0,
                               max_tokens=8, profile=CHAT_PROFILE)
            if out is not None:
                chat_model, chat_ok = m, True
                break
    emb_ok = gateway.embed(EMB_MODEL, ["a", "b"], timeout=10.0) is not None
    return models, chat_model, chat_ok, emb_ok


def is_offline_mode():
    """RAGTOOLKIT_OFFLINE=1 or --offline skips the gateway probe entirely for a
    deterministic, network-free run (see README)."""
    return (os.environ.get("RAGTOOLKIT_OFFLINE", "") not in ("", "0")
            or "--offline" in sys.argv[1:])


def main():
    print(BAR)
    print("BD-APP: budget-displacement poisoning predictor + repair")
    print("SEED =", SEED)
    print(BAR)

    if is_offline_mode():
        models, chat_model, chat_ok, emb_ok = None, None, False, False
        print("[gateway] OFFLINE mode (RAGTOOLKIT_OFFLINE/--offline) -- probe skipped")
    else:
        models, chat_model, chat_ok, emb_ok = probe_gateway()
        # Never print raw model IDs returned by a local service here: `models` may
        # contain internal/non-public deployment codenames that must not end up in a
        # pasted terminal log or public doc; report only a count + booleans.
        print(f"[gateway] models_available={len(models) if models else 0}  "
              f"chat_model_configured={chat_model is not None}  "
              f"chat_generates={chat_ok}  embeddings={emb_ok}")

    corpus = build_corpus(SEED)
    ALL = set(corpus.edges.keys())

    # ----- RED-CASE: mixed-source flip invariant (gateway trips mid-flip) -----
    print("\n" + BAR)
    print("(0) RED-CASE  --  no mixed-source flip when gateway trips mid-pair")
    print(BAR)
    redcase_ok, redcase_src = redcase_no_mixed_source_flip(corpus)
    print(f"flip() source under mid-pair gateway failure: {redcase_src}")
    print(f"NO MIXED-SOURCE COMPARISON: {redcase_ok}")

    # kappa source: prefer gateway embeddings, else deterministic lexical
    emb_cache = {}
    if emb_ok:
        kappa_source = "gateway-embedding"
        def kappa_fn(c, s, m):
            v = method.kappa_embedding(c, s, m, gateway, EMB_MODEL, emb_cache)
            return method.kappa_lexical(c, s, m) if v is None else v
    else:
        kappa_source = "lexical-fallback"
        kappa_fn = method.kappa_lexical

    # Graded tables below use the deterministic label-free reader so the headline
    # numbers are reproducible offline and independent of live-model variance;
    # the REAL-LLM loop is demonstrated separately in section (3).
    labeler = FlipLabeler(corpus, gw=None, model=None, max_calls=MAX_GEN_CALLS)

    merges = ["M_peripheral", "M_central", "M_distract"]

    # ----- (1) RANKING: predictor vs centrality vs ground-truth harm -----
    print("\n" + BAR)
    print("(1) RANKING  --  centrality inversion")
    print(BAR)
    rows = []
    for m in merges:
        pr = method.PR(corpus, m, ALL, kappa_fn)
        cent = method.centrality_of_merge(corpus, m)
        seeds = method.affected_seeds(corpus, m)
        flips = 0
        for s in seeds:
            f, _ = labeler.flip(s, m, ALL)
            flips += f
        harm = flips / len(seeds)
        rows.append((m, pr, cent, harm, len(seeds)))

    print(f"{'merge':<14}{'PR(predict)':>13}{'centrality':>12}"
          f"{'harm(flip%)':>13}{'#seeds':>8}")
    for m, pr, cent, harm, n in rows:
        print(f"{m:<14}{pr:>13.3f}{cent:>12.4f}{harm*100:>12.1f}%{n:>8}")

    pr_rank = [r[0] for r in sorted(rows, key=lambda r: -r[1])]
    cent_rank = [r[0] for r in sorted(rows, key=lambda r: -r[2])]
    harm_rank = [r[0] for r in sorted(rows, key=lambda r: -r[3])]
    print("\nPR ranking       :", " > ".join(pr_rank))
    print("centrality ranking:", " > ".join(cent_rank))
    print("ground-truth harm :", " > ".join(harm_rank))

    pr_p = dict((r[0], r[1]) for r in rows)
    cent_p = dict((r[0], r[2]) for r in rows)
    harm_p = dict((r[0], r[3]) for r in rows)
    inversion = (pr_p["M_peripheral"] > pr_p["M_central"]
                 and cent_p["M_peripheral"] < cent_p["M_central"]
                 and harm_p["M_peripheral"] > harm_p["M_central"])
    print("\nINVERSION CONFIRMED:", inversion,
          " (PR & ground-truth put peripheral > central; centrality inverts it)")

    # ----- CALIBRATION: monotone PR-per-case -> P(flip) -----------------
    print("\n" + BAR)
    print("(1b) CALIBRATION  --  per-case displacement score -> P(flip)")
    print(BAR)
    cal_seeds = []
    for m in merges:
        cs = method.affected_seeds(corpus, m)
        take = cs if m == "M_peripheral" else cs[:4]
        cal_seeds += [(s, m) for s in take]
    cal_seeds = cal_seeds[:20]            # cap
    ds, ys = [], []
    for s, m in cal_seeds:
        contrib, d, k, sw, rr = method.per_seed_displacement(corpus, s, m, ALL, kappa_fn)
        f, _ = labeler.flip(s, m, ALL)
        ds.append(contrib)
        ys.append(f)
    xs, fit = calibrate.isotonic_fit(ds, ys)
    a = calibrate.auc(ds, ys)
    ds_np, ys_np = np.asarray(ds), np.asarray(ys)
    mflip = ds_np[ys_np == 1].mean() if (ys_np == 1).any() else float("nan")
    mno = ds_np[ys_np == 0].mean() if (ys_np == 0).any() else float("nan")
    print(f"calibration cases      : {len(ds)}  (positives={int(ys_np.sum())})")
    print(f"mean d | flip=1        : {mflip:.3f}")
    print(f"mean d | flip=0        : {mno:.3f}")
    print(f"isotonic P(flip|d=0)   : {calibrate.predict(xs, fit, 0.0):.3f}")
    print(f"isotonic P(flip|d=max) : {calibrate.predict(xs, fit, max(ds) if ds else 0):.3f}")
    print(f"ranking AUC (d vs flip): {a:.3f}")

    # ----- (2) REPAIR: displacement-cut vs centrality-cut vs dedup-cut ---
    print("\n" + BAR)
    print(f"(2) REPAIR  --  equal edit budget K = {EDIT_BUDGET_K} edges cut")
    print(BAR)

    def answer_fn(seed, ctx):
        v, _ = labeler._answer(seed, ctx)
        return v

    at_risk = [s for s in method.affected_seeds(corpus, "M_peripheral")
               if labeler.flip(s, "M_peripheral", ALL)[0] == 1]
    base_flips, base_prec = repair.evaluate(corpus, ALL, at_risk, answer_fn)
    print(f"at-risk seeds (baseline flips): {len(at_risk)}")
    print(f"NO-REPAIR   flips={base_flips}/{len(at_risk)}  "
          f"flip_rate={(base_flips/len(at_risk)*100 if at_risk else 0.0):5.1f}%  "
          f"precision={base_prec:.3f}")

    results = {}
    for strat in ("displacement", "centrality", "dedup"):
        cut = repair.choose_cut(corpus, ALL, EDIT_BUDGET_K, strat, kappa_fn)
        active = ALL - cut
        flips, prec = repair.evaluate(corpus, active, at_risk, answer_fn)
        results[strat] = (flips, prec, cut)
        rate = flips / len(at_risk) * 100 if at_risk else 0.0
        print(f"{strat:<12} flips={flips}/{len(at_risk)}  flip_rate={rate:5.1f}%  "
              f"precision={prec:.3f}  cut_edges={len(cut)}")

    d_flips = results["displacement"][0]
    c_flips = results["centrality"][0]
    dd_flips = results["dedup"][0]
    d_prec = results["displacement"][1]
    c_prec = results["centrality"][1]
    repair_win = (d_flips < c_flips and d_flips < dd_flips and d_prec > c_prec)
    print(f"\nDISPLACEMENT-REPAIR WINS: {repair_win}  "
          f"(fewer residual flips AND higher precision than centrality/dedup at equal K)")

    # ----- (3) REAL-LLM LOOP: naive integration vs thinking-off, MEASURED -----
    print("\n" + BAR)
    print("(3) REAL-LLM LOOP  --  naive gateway call vs thinking-off (INFORMATIONAL)")
    print(BAR)
    # This section is purely DIAGNOSTIC and never gates RESULT: PASS/FAIL below. A
    # live gateway's answer quality is not a deterministic property of this repo's
    # method, so making the exit code depend on it would make the suite non-
    # reproducible (it previously required corr_usable > naive_usable, which a fully
    # cooperative gateway fails by tying rather than beating).
    ab = None
    if chat_ok:
        ab = real_llm_ab(corpus, chat_model)
        print(f"batch pairs                    : {ab['pairs']}")
        print(f"NAIVE (default persona, mt=6)  : usable value tokens "
              f"{ab['naive_usable']}/{ab['pairs']}")
        print(f"CORRECTED ({CHAT_PROFILE}, mt=12) : usable value tokens "
              f"{ab['corr_usable']}/{ab['pairs']}")
        print(f"real-generator flip labels     : n_real={ab['n_real']} "
              f"n_oracle={ab['n_oracle']}  source={ab['source']}")
        print(f"real vs label-free-oracle agree: {ab['agree']}/{ab['pairs']}")
        print(f"LOOP CLOSED (generator_source=real-gateway): {ab['loop_closed']}")
        print(f"CORRECTED >= NAIVE usable tokens (informational, not gating): "
              f"{ab['corr_usable'] >= ab['naive_usable']}")
    else:
        print("gateway chat unavailable -> real loop SKIPPED (honest fallback).")
        print("Route needed to close it: local OpenAI-compatible gateway at")
        print(f"  {gateway.BASE}/v1/chat/completions  with model id in "
              f"{CHAT_MODEL_PREF}")
        print(f"  and profile field set to {CHAT_PROFILE!r}  (thinking off,")
        print("  else a thinking model returns empty content and the loop cannot close).")

    # ----- provenance -----
    print("\n" + BAR)
    print("PROVENANCE")
    print(BAR)
    print("graded_tables_source :", labeler.source_summary(),
          f"(deterministic label-free reader; real_calls={labeler.n_real},"
          f" oracle_calls={labeler.n_oracle})")
    if ab is not None:
        print("real_llm_loop        :", ab['source'],
              f"(real_calls={ab['n_real']}, oracle_calls={ab['n_oracle']})")
    else:
        print("real_llm_loop        : SKIPPED (gateway chat unavailable)")
    print("kappa_source         :", kappa_source)
    print("seed                 :", SEED)
    print("reproducible         : graded tables deterministic given SEED; "
          "real-LLM loop optional")

    ok = inversion and repair_win and redcase_ok
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
