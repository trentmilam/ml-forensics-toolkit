# QSLM — Quantization-vs-weights typing via a scheme-locked self-quantization family

QSLM decides whether an observed model `C` differs from a reference model `W_ref`
because it was **quantized** (and if so recovers the bit-width `b*`) or because its
**weights were changed** (finetune / edit) — even when the two changes have the
**same residual magnitude**, the case where every magnitude/energy test provably fails.

## Installation / Prerequisites

Tested on Python 3.12. From the repo root:

```
pip install -r requirements.txt
python qslm/run_experiment.py
```

`run_experiment.py` also pings a local OpenAI-compatible gateway at `127.0.0.1:8000`
for an informational status line only — the method never uses it (see "What is
controlled vs real" below). Set `RAGTOOLKIT_OFFLINE=1` (or pass `--offline`) to skip
that probe entirely for a deterministic, network-free run; override the probed URL
with `QSLM_GATEWAY_BASE`.

## Independent claim (method)

A method for typing a candidate model relative to a white-box reference model, comprising:

1. **synthesizing a scheme-locked self-quantization family** `{Q_b(W_ref)}` by applying,
   to the reference weights, the same block-wise affine (k-quant-style) quantizer at a
   plurality of bit-widths `b`, and embedding a fixed probe set with the reference and
   with each family member to obtain per-member residuals `r_b = emb(Q_b) − emb(ref)`;
2. **deriving three scheme-locked signatures** from the family: (a) an **energy-vs-bit
   curve** `eps(b)`; (b) a **cross-bit invariant residual subspace `U`** — the residual
   directions that are consistent across bit-widths, obtained from the top eigenvectors of
   the scale-normalized mean residual covariance; and (c) an **empirical residual null**
   over a same-scheme residual statistic (the effective rank of the residual matrix);
3. **typing a candidate** by its residual to the reference as *quantized* iff (i) its
   residual energy lands on the energy-vs-bit curve at a recoverable `b*`, **and**
   (ii) the principal-angle overlap of its residual subspace with `U` exceeds a bound
   calibrated from the family, **and** (iii) its residual effective rank lies in the null
   band; otherwise typing it as *weight-changed*; and, when quantized, **recovering `b*`**
   by inverting the energy-vs-bit curve.

The inventive step over a magnitude/energy test: quantization error is **dense and
near-isotropic in weight space**, so — propagated through the fixed network on the fixed
probes — it consistently excites the network's dominant response subspace `U` and produces
a high-rank residual; a **naive finetune delta is a random low-rank object**, so even when
scaled to the identical residual energy it lands in a random subspace and is low-rank, failing
(ii) and (iii). Energy alone therefore cannot separate the two; the scheme-locked subspace/rank
signatures can.

**Honest boundary — the load-bearing condition is (iii), not (ii).** An *adaptive* adversary
who knows the method can steer its low-rank delta directly **into `U`** (a second-layer delta
`dW2 = U·M`), producing a residual whose subspace overlap with `U` is ~1.0 — **higher than a
real quant member** — which **defeats condition (ii)** (measured subspace-overlap AUC on this
adversary collapses to **0.000**). It does **not** break the method, because `U` is only
`k`-dimensional: confining the residual to `span(U)` **structurally caps its effective rank at
`k`** (measured effective rank ≈ 8.7 vs quant ≈ 39.7), so condition **(iii) still separates it
perfectly (AUC 1.000, 0 false positives)**. The non-obviousness therefore rests on the
**conjunction**: no weight-change can be simultaneously *`U`-aligned* (energy inside a
`k`-dimensional subspace) **and** *high-effective-rank* — the two adversarial objectives are
mutually exclusive by construction, whereas dense quantization error satisfies both. Condition
(ii) alone is not robust to an adaptive adversary; the tri-condition (specifically the rank
null) is.

## What is controlled vs real (honest scope)

- **CONTROLLED (this reduction-to-practice):** the "model" is a fixed-seed numpy MLP
  (2 layers, hidden 256, output 128-d L2-normalized). We own its **reference weights**, so
  we can synthesize the full self-quantization family. The quantizer is a **real** per-block
  asymmetric affine quantizer (block size 32, bit-widths 8/6/5/4/3, fractional via mixed
  per-block bit assignment) — a faithful numpy model of GGUF k-quant. The finetune
  candidates are `W_ref + low-rank ΔW` (LoRA-style), each **scaled so its residual energy
  equals a quant member's** (matched-magnitude, the hard case).
- **REAL / upgrade path:** replace the numpy MLP with an actual GGUF model and its k-quant
  variants (`Q8_0…Q3_K`), the probes with real tokens, and `emb()` with the model's
  embedding/logit readout. The **method is unchanged** — family synthesis, the three
  signatures, and the tri-condition classifier apply verbatim. This module validates the
  method; it does not claim numbers on a production LLM.
- The local gateway (`http://127.0.0.1:8000`) is pinged and its status recorded, but is
  **not used**: the method fundamentally requires white-box reference weights that a black-box
  embeddings endpoint cannot provide. The run is fully reproducible offline.

## Measured results (seed 20260703; verbatim from `run_experiment.py`)

Family energy-vs-bit fit: `ln(eps) = 1.489 − 1.435·b` (max family fit residual 0.065).

Matched-magnitude candidates (24 quant vs 24 finetune), finetune energy-match relative
error ≈ 1.1e-09 (energy sets are identical):

| metric | value |
|---|---|
| tri-condition confusion accuracy | **0.917** (20/24 quant, 24/24 finetune; **0 false positives**) |
| AUC — tri-condition subspace overlap | **1.000** |
| AUC — tri-condition effective rank | **1.000** |
| AUC — **baseline energy-only (magnitude)** | **0.491 (≈ chance)** |
| quant vs finetune subspace overlap | mean 0.424 vs 0.060 |
| quant vs finetune effective rank | mean 39.7 vs 13.3 |
| b* recovery error (24 quant, b*∈[3.2,7.8]) | MAE **0.107 bits**, max 0.204 bits |

Interpretation: the matched-magnitude design forces the **energy-only baseline to chance
(AUC 0.491)** — it provably cannot tell quant from finetune. The scheme-locked subspace and
effective-rank signatures separate them **perfectly (AUC 1.000)**, and `b*` is recovered to
~0.1 bit. The 4 false negatives are high-bit quant candidates (tiny residual, noisier
subspace) conservatively typed as weight-changed; the classifier produces **no** false
positives (no finetune is ever mistaken for a quantization).

### Hard adaptive adversary (U-aligned finetune) — measured

A negative control stronger than a random delta: 24 finetune deltas **steered into the
invariant subspace `U`** (`dW2 = U·M`), each energy-matched to a quant member.

| metric (quant vs U-aligned) | value |
|---|---|
| U-aligned subspace overlap vs `U` | mean **0.998** (vs quant 0.424 — adversary exceeds real quant) |
| U-aligned effective rank | mean **8.73** (vs quant 39.71; `k=8` caps a `U`-confined residual) |
| AUC — subspace overlap | **0.000 — DEFEATED** (condition (ii) alone is not robust) |
| AUC — effective rank | **1.000** (condition (iii) is the load-bearing defense) |
| tri-condition false positives (U-aligned typed as quant) | **0 / 24** |

This is the honest non-obviousness result: an adversary *can* beat the subspace test, but the
effective-rank null — and therefore the tri-condition conjunction — still rejects every
U-aligned finetune, because U-alignment and high effective rank are mutually exclusive.

### Fair-baseline head-to-head (smart magnitude-free detectors) — measured

The energy-only baseline above is chance **by construction** (energies are matched), so beating
it proves little. The honest question is whether the tri-condition beats the detector a competent
engineer would *actually* build. Without the scheme-locked family, the two strongest magnitude-free
signals are a residual **effective-rank threshold** and a residual **subspace-overlap-with-`U`
threshold** — each a *reasonable* single-signal detector, not a strawman. To expose the second
adversary a rank test misses, we add a **dense full-rank finetune** (real full finetuning, *not*
LoRA), energy-matched: its residual is **high-rank** (mean effective rank **35.7** vs quant 39.7 —
fools a rank threshold) but **random-subspace** (overlap with `U` mean **0.055** — fails the
subspace test). False positives (a weight-change typed as *quant*), out of 24 each:

| adversary | rank-only detector | overlap-only detector | **tri-condition (QSLM)** |
|---|---|---|---|
| random LoRA | 0 | 0 | **0** |
| U-aligned LoRA | 0 | **24 — DEFEATED** | **0** |
| dense full-rank finetune | **24 — DEFEATED** | 0 | **0** |
| **total false positives** | **24** | **24** | **0** |
| quant recall (of 24) | 24 | 20 | 20 |

Each single smart detector is defeated by exactly one realistic weight-change adversary
(rank-only by dense full finetuning; overlap-only by the U-aligned attack); **only the
conjunction rejects all three with 0 false positives.** This is a fair A/B — QSLM beats the
*smart* magnitude-free detector, not merely the tautological energy-only one.

## Files & reproduce

- `quantizer.py` — per-block affine (k-quant-style) quant/dequant; fractional mixed-bit.
- `embedder.py` — controlled MLP embedder, reference weights, fixed probes.
- `method.py` — family synthesis, energy curve, invariant subspace `U`, null, tri-condition
  classifier, `b*` recovery, rank-based AUC.
- `run_experiment.py` — runs the experiment and prints all measured numbers.

```
python qslm/run_experiment.py
```

Deterministic (fixed `SEED`, `numpy.random.default_rng`); exits 0 on the pass criteria
(subspace AUC > 0.85, baseline AUC within 0.15 of 0.5, accuracy > 0.85, **and** the U-aligned
adaptive adversary is still separated by effective rank: erank AUC > 0.85 with 0 false positives).
