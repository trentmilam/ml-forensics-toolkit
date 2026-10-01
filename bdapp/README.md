# BD-APP: budget-displacement poisoning predictor and repair

Research prototype. Controlled synthetic corpus. Not run against a production GraphRAG pipeline.

Predicts which GraphRAG entity-merge (two different entities collapsed into one node) will poison an answer. Generator-free: no language model in the prediction. Scores by token-budget displacement, not graph centrality or dedup confidence. Repairs the graph under a fixed edit budget.

## Install and run

Python 3.12, from the repo root:

```
pip install -r requirements.txt
python bdapp/run_experiment.py
```

- Section (3) probes a local OpenAI-compatible gateway at `127.0.0.1:8000` (informational, never gates PASS/FAIL).
- Offline: `RAGTOOLKIT_OFFLINE=1` or `--offline`.
- Gateway URL override: `BDAPP_GATEWAY_BASE`.

## How it works

Scores a false-merge `M` in a corpus that assembles retrieved chunks into a fixed-token-budget context.

1. Assemble the context twice per affected seed `e`: with `M`, and with the counterfactual split of `M`.
2. Displacement: `dtok(e, M)` = on-entity, query-relevant tokens present in the split context but evicted in the with-`M` context by foreign content.
3. Contradiction: deterministic proxy `kappa(e, M)` flags whether the foreign content conflicts with the answer.
4. Score: `PR(M) = Σ_e s(e) · r(e→M) · dtok(e, M) · κ(e, M)`, with query-log-free seeding weight `s(e)` and expansion-reach weight `r(e→M)`.
5. Repair: remove the `K` foreign-routing edges with the highest `dtok·κ` per unit (displacement-weighted min-cut).

## Measured results (this repo)

Run: `python bdapp/run_experiment.py`

Sample: 3 injected merges, 12 poisoned seeds, repair budget `K = 6`. Three-decimal values are raw printed output. Directional evidence only.

### (1) Centrality inversion

Centrality: mean degree-centrality over the poisoned seed sites.

| merge          | PR (predict) | degree centrality (seed sites) | ground-truth harm (flip rate) |
|----------------|-------------:|-------------------------------:|------------------------------:|
| M_peripheral   | 14.400       | 0.0256                         | 75.0 %                        |
| M_central      | 0.000        | 0.0769                         | 0.0 %                         |
| M_distract     | 0.000        | 0.0545                         | 0.0 %                         |

- PR ranking: `M_peripheral > M_central`, matches ground truth.
- Centrality ranking: `M_central > M_distract > M_peripheral`, inverted.
- Ground truth (label-free plurality reader) agrees with PR.
- 75 % harm: the same merge flips the 9 tight-budget seeds and not the 3 roomy-budget seeds.
- `M_distract`: `dtok > 0`, `κ = 0`, flips nothing.

### (1b) Calibration `PR → P(flip)`

Isotonic (PAVA) fit over 20 capped (query, M) cases: `P(flip | d=0) = 0.000`, `P(flip | d=max) = 1.000`, ranking AUC = 1.000, mean per-case score `d = 1.600` for flipped vs `0.000` for non-flipped.

On a real corpus, out-of-threat-model flips (corroboration-stripping, where `dtok` scores 0) would pull AUC below 1.

### (2) Repair at equal edit budget `K = 6` edges

| strategy               | residual flip rate | on-entity precision |
|------------------------|-------------------:|--------------------:|
| no repair              | 100.0 %            | 0.000               |
| displacement-cut       | 33.3 %              | 0.333               |
| centrality-cut         | 100.0 %            | 0.000               |
| dedup-confidence-cut   | 100.0 %            | 0.000               |

- Displacement-cut: flip rate 100 % to 33 %.

## Prior art

Closest prior work:

- PoisonedRAG (Zou et al., 2024, [arXiv:2402.07867](https://arxiv.org/abs/2402.07867)): external adversary injecting passages. BD-APP: internal entity-resolution error, predicted and repaired.
- Lost in the Middle (Liu et al., 2023, [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)): context budget and position govern which evidence a model uses. `dtok` measures that eviction.
- GraphRAG (Edge et al., 2024, [arXiv:2404.16130](https://arxiv.org/abs/2404.16130)): the entity-graph-plus-community pipeline assumed here.
- Isotonic regression / PAVA: monotone score-to-probability calibrator (Niculescu-Mizil & Caruana, 2005; also `sklearn.isotonic.IsotonicRegression`).

Reused, not claimed as new: poisoning as a RAG framing, the fixed-token-budget mechanism, GraphRAG's pipeline, isotonic/PAVA calibration (plain numpy), budgeted edge-removal.

Combination: diff two budget-truncated context assemblies (merged graph vs counterfactual split), count the on-entity tokens evicted (`dtok`), combine with `kappa` into one generator-free score `PR(M)`, use the same per-edge weight for repair. Baselines: [Measured results](#measured-results-this-repo).

Search coverage: arXiv, GitHub, PyPI. No implementation of this signal found. Code search limited to repository-level.

## Scope and limitations

- Controlled synthetic corpus. Real corpora would show a noisier `PR→flip` relation than AUC = 1.0.
- Corpus, budgets, merges and foreign chunks are constructed to exhibit the regime.
- Ground truth does not reuse the predictor. The fallback reader is a label-free plurality reader: among status chunks naming the seed, it returns the value with the most support, ties broken toward higher retrieval relevance. It never reads the hidden `authentic` flag or the true native value.
- Probe: native chunk absent, two corroborators present, reader still returns the correct value.
- An earlier version leaked the label (it preferred the `authentic` chunk). Removed.
- `dtok`: two budgeted assemblies, diffed. Harm: independent reader. Centrality: real graph degree of the poisoned seed sites. Dedup confidence: real native/foreign token Jaccard.
- Known blind spot: corroboration-stripping, where the poison out-ranks a thinned-out correct set in a co-location vote. `dtok` scores it 0.
- Generator source (real-LLM loop, measured 2026-07-04): graded tables use the deterministic reader. Section (3) also runs the real local gateway generator at `temperature=0`. Anecdotal: one gateway/model configuration. Not reproducible; re-runs differed.

  | integration (same endpoint, same contexts)         | usable value tokens | flip loop |
  |----------------------------------------------------|--------------------:|-----------|
  | naive, default persona, `max_tokens=6`             | 0 / 6                | cannot close (empty thinking-only replies, silent fallback to oracle) |
  | corrected, thinking-off profile selected via the gateway's profile field, `mt=12` | 6 / 6 | closes: `generator_source = real-gateway`, `real_calls = 12`, `oracle_calls = 0` |

  - Earlier runs reported `generator_source = fallback-oracle` despite a live gateway.
  - Real model agrees with the oracle on 5 / 6 batch pairs. The disagreement: `M_central` seed 1, where the real model flips on a co-located contradiction (`dtok = 0`, `κ = 1`).
  - `kappa_source = lexical-fallback`: the gateway's embeddings endpoint returned 404.
  - Graded tables deterministic given `SEED = 20260703`. The real-LLM loop is optional and skipped when the gateway is unreachable.

## Files

- `corpus.py`: synthetic GraphRAG corpus + injected false-merges (peripheral /
  central / distractor).
- `method.py`: budget assembler, `dtok`, `κ`, `s`, `r`, `PR`, and the baseline
  signals (degree centrality, dedup confidence).
- `oracle.py`: real-gateway flip labelling with a deterministic, label-free
  plurality-reader fallback (ignores the `authentic` flag and native value).
- `calibrate.py`: isotonic (PAVA) `PR→P(flip)` calibration + AUC, numpy-only.
- `repair.py`: displacement / centrality / dedup edge-cut strategies + eval.
- `gateway.py`: fail-safe client for the local OpenAI-compatible gateway.
- `run_experiment.py`: runs everything and prints the measured results.
