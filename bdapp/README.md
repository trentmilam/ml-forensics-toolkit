# BD-APP — Budget-Displacement Poisoning Predictor + Displacement-Weighted Repair

A **generator-free** method to predict which GraphRAG false-merge will poison
answers, and to **repair** the graph under a fixed edit budget — by measuring
*token-budget displacement* instead of graph centrality or dedup confidence.

## Installation / Prerequisites

Tested on Python 3.12. From the repo root:

```
pip install -r requirements.txt
python bdapp/run_experiment.py
```

Section (3) additionally probes a local OpenAI-compatible gateway at `127.0.0.1:8000`
(informational only — see "Honest scope" below; it never gates the PASS/FAIL verdict).
Set `RAGTOOLKIT_OFFLINE=1` (or pass `--offline`) to skip the probe entirely for a
deterministic, network-free run; override the probed URL with `BDAPP_GATEWAY_BASE`.

## Independent claim (method)

A computer-implemented method for scoring the poisoning risk of a false-merge `M`
in a graph-based retrieval corpus that assembles retrieved chunks into a
fixed-token-budget context, comprising:

1. for each affected seed `e`, assembling the truncated context **twice** — once
   with `M` present and once for the counterfactual **split** of `M` — under the
   seed's fixed token budget;
2. computing a **displacement** `dtok(e, M)` = the number of on-entity,
   query-relevant tokens that are present in the split-`M` truncated context but
   **evicted** in the with-`M` truncated context by foreign content routed
   through `M`;
3. computing a deterministic **directional-contradiction proxy** `kappa(e, M)`;
4. computing a poisoning-risk score
   **`PR(M) = Σ_e s(e) · r(e→M) · dtok(e, M) · κ(e, M)`**
   using a query-log-free seeding weight `s(e)` and an expansion-reach weight
   `r(e→M)` — **without invoking any generative model**; and
5. **repairing** the corpus by removing the `K` foreign-routing edges of highest
   per-unit load-bearing displacement (`dtok·κ`) — a displacement-weighted
   min-cut — within a fixed edit budget `K`.

**Inventive effect (what the baselines miss):** poisoning damage is driven by
*budget displacement*, not by where a merge sits in the graph nor by how
confident a dedup system is about it. A peripheral merge that routes heavy,
identity-conflated content can evict the load-bearing on-entity chunk and flip
the answer, while a highly-central merge that routes light content flips nothing.
`PR` captures this; a centrality baseline **inverts** the true ranking, and a
dedup-confidence baseline trusts the very identity-conflated edges that do the
damage.

## Reduction-to-practice (measured, this repo)

Run: `python bdapp/run_experiment.py`

### (1) Centrality inversion — measured

Centrality here is scored on the merge's **actual injection/seed sites** (mean
degree-centrality over the poisoned seeds) — the sites a realistic centrality
defence would guard — not a decorative anchor label.

| merge          | PR (predict) | degree centrality (seed sites) | ground-truth harm (flip rate) |
|----------------|-------------:|-------------------------------:|------------------------------:|
| M_peripheral   | **14.400**   | 0.0256                         | **75.0 %**                    |
| M_central      | 0.000        | **0.0769**                     | 0.0 %                         |
| M_distract     | 0.000        | 0.0545                         | 0.0 %                         |

- **PR ranking:** `M_peripheral > M_central` — matches ground truth.
- **Centrality ranking:** `M_central > M_distract > M_peripheral` — **inverted**:
  the harmful merge sits on the *lowest*-centrality (leaf) seeds while the harmless
  one sits on the *highest*-centrality (hub) seeds.
- Ground-truth harm (independently measured answer-flip rate, from a **label-free
  plurality reader** — see below) agrees with **PR**, not centrality.
- The 75 % (not 100 %) harm is itself the causal control: the *same* merge with the
  *same* contradictory foreign chunk flips only the 9 tight-budget seeds and not
  the 3 roomy-budget seeds, where the correct evidence survives and out-votes the
  poison — isolating **budget displacement**, not merge identity, as the driver.
- `M_distract` evicts on-entity tokens (`dtok > 0`) but is non-contradictory on
  the queried attribute (`κ = 0`), so it flips nothing — showing `PR` needs
  **both** displacement and contradiction, and that a displacement-only signal
  would over-predict.

### (1b) Calibration `PR → P(flip)` — measured

Isotonic (PAVA) fit over 20 capped (query, M) cases: `P(flip | d=0) = 0.000`,
`P(flip | d=max) = 1.000`, ranking **AUC = 1.000**, mean per-case score
`d = 1.600` for flipped vs `0.000` for non-flipped.

The perfect separation is a property of this controlled corpus, **not** of the
labelling: the ground-truth reader is now independent of the predictor (see Honest
scope). On a real corpus, out-of-threat-model flips (e.g. corroboration-stripping
that lets a higher-relevance poison win a co-location vote — a case `dtok` scores 0)
would pull the AUC below 1.

### (2) Repair at equal edit budget `K = 6` edges — measured

| strategy               | residual flip rate | on-entity precision |
|------------------------|-------------------:|--------------------:|
| no repair              | 100.0 %            | 0.000               |
| **displacement-cut**   | **33.3 %**         | **0.333**           |
| centrality-cut         | 100.0 %            | 0.000               |
| dedup-confidence-cut   | 100.0 %            | 0.000               |

The displacement-weighted min-cut spends the same budget but **reduces the
answer-flip rate from 100 % to 33 %** and raises on-entity truncated-context
precision, while both baselines spend their budget on harmless edges and leave
every flip in place. The baseline failures are **emergent, not constructed**:
centrality-cut spends its budget on the highest-centrality (hub) seeds — which
happen to carry the harmless light merge — and dedup-cut spends its budget on the
lowest-Jaccard edges — which are the off-attribute distractors — because the
identity-conflated poison scores *higher* dedup confidence than the distractors it
is measured against. Neither baseline is handicapped; each is given its genuine
signal and misranks on it.

## Honest scope

- **Controlled synthetic corpus.** The corpus, budgets, merges, and foreign
  chunks are constructed to *exhibit* the budget-displacement / centrality-
  inversion regime. The numbers above are **measured** on that controlled corpus,
  not on production data. Real corpora would show a noisier `PR→flip` relation
  than the perfect separation (AUC = 1.0) seen here.
- **The ground truth does not reuse the predictor (no label leakage).** The
  fallback reader is a **label-free plurality reader**: among the status chunks
  that name the seed in the *assembled* context, it returns the value with the
  most support, ties broken toward higher retrieval relevance. It never consults
  the hidden `authentic` flag nor the true native value — verified by a
  discriminating probe: with the native chunk **absent** but two corroborators
  present, it still returns the correct value by majority even though the poison
  out-ranks them on relevance. So a "flip" is decided by a vote over whatever
  survives the budget — an event computed independently of the predictor's
  `dtok`/`κ`. (An earlier version *did* leak: it preferred the `authentic` chunk
  whenever present, which made the label a restatement of "was the authentic
  chunk evicted" — the same event `dtok` measures. That has been removed.)
- **What is genuinely measured vs. assumed.** `dtok` is measured by running the
  budgeted assembler twice and diffing the two truncated contexts; the
  ground-truth harm is measured by the independent reader above; **centrality is
  the real graph degree of the poisoned seed sites** (not a decorative anchor);
  dedup confidence is the real native/foreign token Jaccard. The naive baselines
  genuinely fail on their genuine signals (inverted ranking; 100 % residual flips).
- **Threat model + known blind spot.** A counted "flip" requires the correct
  value to *lose the plurality* — in this corpus that happens when the correct
  evidence is *evicted from the budget* (displacement) and a contradictory foreign
  value survives. Mere co-location of correct and contradictory chunks does **not**
  flip, because corroboration keeps the correct value's majority — a principled,
  label-free reason, not a thesis-favouring rule. The invention concerns budget
  displacement specifically, and it therefore **misses** a different attack:
  *corroboration-stripping*, where the poison out-ranks a thinned-out correct set
  in a co-location vote. `dtok` scores that 0, so the predictor would not flag it.
  A real corpus would contain such cases and the AUC would fall below 1.
- **Generator source (real-LLM loop CLOSED — measured 2026-07-04).** The graded
  tables above use the deterministic **label-free reader** so the headline numbers
  are reproducible offline and independent of live-model variance. Section (3) of
  the experiment *additionally* closes the loop against the **real** local gateway
  generator at `temperature=0` and reports a measured head-to-head:

  | integration (same endpoint, same contexts)         | usable value tokens | flip loop |
  |----------------------------------------------------|--------------------:|-----------|
  | **naive** — default persona, `max_tokens=6`        | **0 / 6**           | cannot close (empty thinking-only replies → silent fallback to oracle) |
  | **corrected** — thinking-off profile selected via the gateway's profile field, `mt=12` | **6 / 6** | closes: `generator_source = real-gateway`, `real_calls = 12`, `oracle_calls = 0` |

  This is why earlier runs reported `generator_source = fallback-oracle` *despite a
  live gateway*: the configured chat model is a thinking model whose default persona
  spends its whole token budget on hidden reasoning and returns empty content, which
  the loop correctly treats as a non-answer. Selecting the thinking-off profile
  closes the loop. **The real model agrees with the label-free oracle on 5 / 6 batch pairs.**
  The one disagreement is honest and instructive: on `M_central` seed 1 the real
  model *flips* on a **co-located** contradiction (`dtok = 0`, `κ = 1` — the correct
  value is corroborated but not evicted) that the budget-displacement predictor does
  **not** flag. That is a live instance of the documented co-location / corroboration
  blind spot (see "Threat model + known blind spot"), not a failure of the
  displacement claim — the predictor concerns budget displacement specifically.
  **`kappa_source = lexical-fallback`**: the gateway's embeddings endpoint returned
  404 in this run, so `κ` uses the deterministic lexical proxy. The experiment is
  fully reproducible offline (graded tables deterministic given `SEED = 20260703`);
  the real-LLM loop is optional and skipped with a documented route when the gateway
  is unreachable.

## Files

- `corpus.py` — synthetic GraphRAG corpus + injected false-merges (peripheral /
  central / distractor).
- `method.py` — budget assembler, `dtok`, `κ`, `s`, `r`, `PR`, and the baseline
  signals (degree centrality, dedup confidence).
- `oracle.py` — real-gateway flip labelling with a deterministic, **label-free
  plurality-reader** fallback (ignores the `authentic` flag and native value).
- `calibrate.py` — isotonic (PAVA) `PR→P(flip)` calibration + AUC, numpy-only.
- `repair.py` — displacement / centrality / dedup edge-cut strategies + eval.
- `gateway.py` — fail-safe client for the local OpenAI-compatible gateway.
- `run_experiment.py` — runs everything and prints the measured results.
