# QSLM: quantization-vs-weights typing via a scheme-locked self-quantization family

Research prototype. Controlled synthetic model. Real GGUF model not done.

Decides whether an observed model `C` differs from a reference `W_ref` because it was quantized (recovers bit-width `b*`) or because its weights were changed (finetune / edit). Covers matched residual magnitude, where energy/magnitude tests fail.

## Install and run

Python 3.12, from the repo root:

```
pip install -r requirements.txt
python qslm/run_experiment.py
```

- Pings a local OpenAI-compatible gateway at `127.0.0.1:8000` for a status line only. The method never uses it.
- Offline: `RAGTOOLKIT_OFFLINE=1` or `--offline`.
- Gateway URL override: `QSLM_GATEWAY_BASE`.

## How it works

1. Self-quantization family: apply the same block-wise affine (k-quant-style) quantizer to the reference weights at several bit-widths `b`, giving `{Q_b(W_ref)}`. Embed a fixed probe set with the reference and each member: residuals `r_b = emb(Q_b) − emb(ref)`.
2. Three signatures from the family:
   - energy-vs-bit curve `eps(b)`
   - cross-bit invariant residual subspace `U`: top eigenvectors of the scale-normalized mean residual covariance
   - empirical null over the residual's effective rank
3. Type `C` as quantized only if all three hold:
   - (i) residual energy lands on the energy-vs-bit curve at a recoverable `b*`
   - (ii) principal-angle overlap of its residual subspace with `U` exceeds a bound calibrated from the family
   - (iii) residual effective rank lies in the null band

   Otherwise `C` is weight-changed. For quantized `C`, `b*` comes from inverting the energy curve.

Adaptive adversary: a low-rank delta steered into `U` (`dW2 = U·M`) gets subspace overlap ~1.0, above a real quant member. Subspace-overlap AUC drops to 0.000. Effective rank ≈ 8.7 vs quant ≈ 39.7, rank AUC 1.000, 0 false positives.

## Controlled vs real

- Controlled: fixed-seed numpy MLP (2 layers, hidden 256, output 128-d L2-normalized). Quantizer: per-block asymmetric affine, block size 32, bit-widths 8/6/5/4/3, fractional via mixed per-block bit assignment. Finetune candidates: `W_ref + low-rank ΔW` (LoRA-style), scaled to match a quant member's residual energy.
- Upgrade path: real GGUF model and its k-quant variants (`Q8_0…Q3_K`), real-token probes, the model's embedding/logit readout for `emb()`. No numbers on a production LLM.
- Gateway: pinged and recorded, not used.

## Measured results (seed 20260703; verbatim from `run_experiment.py`)

Family energy-vs-bit fit: `ln(eps) = 1.489 − 1.435·b` (max family fit residual 0.065).

Matched-magnitude candidates (24 quant vs 24 finetune), finetune energy-match relative
error ≈ 1.1e-09 (energy sets are identical):

| metric | value |
|---|---|
| tri-condition confusion accuracy | 0.917 (20/24 quant, 24/24 finetune; 0 false positives) |
| AUC, tri-condition subspace overlap | 1.000 |
| AUC, tri-condition effective rank | 1.000 |
| AUC, baseline energy-only (magnitude) | 0.491 (≈ chance) |
| quant vs finetune subspace overlap | mean 0.424 vs 0.060 |
| quant vs finetune effective rank | mean 39.7 vs 13.3 |
| b* recovery error (24 quant, b*∈[3.2,7.8]) | MAE 0.107 bits, max 0.204 bits |

- The 4 false negatives are high-bit quant candidates, typed as weight-changed.
- No finetune is typed as quantization.

### U-aligned finetune adversary

24 finetune deltas steered into `U` (`dW2 = U·M`), each energy-matched to a quant member.

| metric (quant vs U-aligned) | value |
|---|---|
| U-aligned subspace overlap vs `U` | mean 0.998 (vs quant 0.424) |
| U-aligned effective rank | mean 8.73 (vs quant 39.71; `k=8` caps a `U`-confined residual) |
| AUC, subspace overlap | 0.000 |
| AUC, effective rank | 1.000 |
| tri-condition false positives (U-aligned typed as quant) | 0 / 24 |

### Magnitude-free single-signal detectors

Baselines: residual effective-rank threshold, and residual subspace-overlap-with-`U` threshold. Added a dense full-rank finetune, energy-matched: effective rank mean 35.7 vs quant 39.7, overlap with `U` mean 0.055. False positives (a weight-change typed as quant), out of 24 each:

| adversary | rank-only detector | overlap-only detector | tri-condition (QSLM) |
|---|---|---|---|
| random LoRA | 0 | 0 | 0 |
| U-aligned LoRA | 0 | 24 (defeated) | 0 |
| dense full-rank finetune | 24 (defeated) | 0 | 0 |
| total false positives | 24 | 24 | 0 |
| quant recall (of 24) | 24 | 20 | 20 |

## Files and reproduce

- `quantizer.py`: per-block affine (k-quant-style) quant/dequant; fractional mixed-bit.
- `embedder.py`: controlled MLP embedder, reference weights, fixed probes.
- `method.py`: family synthesis, energy curve, invariant subspace `U`, null, tri-condition
  classifier, `b*` recovery, rank-based AUC.
- `run_experiment.py`: runs the experiment and prints all measured numbers.

```
python qslm/run_experiment.py
```

Deterministic (fixed `SEED`, `numpy.random.default_rng`). Exits 0 on the pass criteria: subspace AUC > 0.85, baseline AUC within 0.15 of 0.5, accuracy > 0.85, and the U-aligned adversary still separated by effective rank (erank AUC > 0.85, 0 false positives).
