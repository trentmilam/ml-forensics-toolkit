# rag-toolkit — original ML methods (novelty-vetted pair)

Two original ML / model-management methods, built as open proof of inventive novelty.

Both survived a rigorous, multi-round **adversarial novelty review by independent AI examiner
agents**: invent → examine against prior art (Google Patents + arXiv, on the anticipation /
obviousness / abstract-idea axes) → amend → two independent re-examiners. Across five such rounds
the yield was deliberately harsh (2 of 5, then 0/10, 0/3, 0/10); only **QSLM** and **BD-APP**
survived. This is a **differentiation check — not a legal patentability opinion**: the examiners
were AI agents, and every result is measured on **controlled / synthetic** setups. Both are built
and proven with a runnable, measured reduction-to-practice.

---

## QSLM — quantization-vs-weights typing  (`qslm/`)

Decides whether a candidate model differs from a white-box reference because it was **quantized**
(and recovers the bit-width `b*`) or because its **weights were changed** — even at *identical
residual magnitude*, the regime where every energy/magnitude test provably fails. Mechanism: a
scheme-locked self-quantization family → energy-vs-bit curve + cross-bit-invariant residual
subspace + effective-rank null → tri-condition classifier.

```
python qslm/run_experiment.py
```

Measured (controlled numpy-embedder validation, fixed seed, exit 0):

| signal | value |
|---|---|
| tri-condition accuracy | 0.917 (0 false positives) |
| AUC — subspace-overlap / effective-rank (the invention) | 1.000 / 1.000 |
| AUC — energy-only baseline (magnitude) | 0.491 (≈ chance) ← non-obviousness evidence |
| bit-width recovery error | MAE 0.107 bits |

*Scope:* validates the **method** on a numpy MLP with a real per-block k-quant quantizer;
a real-GGUF model is the documented upgrade path.

## BD-APP — GraphRAG budget-displacement poisoning predictor + repair  (`bdapp/`)

**Generator-free** prediction of which GraphRAG false-merge will poison answers, driven by
token-budget **displacement** (not centrality or dedup confidence), plus a displacement-weighted
repair under a fixed edit budget.

```
python bdapp/run_experiment.py
```

Measured (controlled synthetic corpus, fixed seed, exit 0):

| signal | value |
|---|---|
| centrality inversion | confirmed — PR ranks the peripheral merge (14.400, 75% real harm) above the central one (0.000, 0% harm); centrality ranks them backwards |
| calibration AUC | 1.000 |
| repair @ equal budget K=6 | displacement-cut **33.3% flips / 0.333 precision** vs centrality & dedup both **100% / 0.000** |

*Scope:* synthetic corpus; flip-labels come from a **deterministic, label-free reader (no live LLM
required)**; the AUC=1.0 is a clean-corpus property measured against an *independent* reader (an
earlier label-leak was caught and fixed in adversarial review). An optional real-LLM labelling pass
is a documented upgrade.

---

## Honesty

Every number above was **observed in real experiment output** (exit 0), not asserted; both
experiments are deterministic (fixed seed, `numpy.random.default_rng`). "Novelty-vetted" means an
adversarial **AI-examiner** review — not a patent-office, legal, or third-party assessment — and the
perfect scores are on **self-authored synthetic corpora**, so treat them as differentiation
evidence, not external validation. Nothing here is investment, legal, or financial advice. MIT-licensed.
