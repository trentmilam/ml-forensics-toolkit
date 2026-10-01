# ml-forensics-toolkit

[![verify](https://github.com/trentmilam/ml-forensics-toolkit/actions/workflows/verify.yml/badge.svg)](https://github.com/trentmilam/ml-forensics-toolkit/actions/workflows/verify.yml)

Two independent research prototypes. No shared code.

- [QSLM](qslm/README.md): quantized vs altered model weights
- [BD-APP](bdapp/README.md): which GraphRAG entity-merges corrupt an answer, and repair under an edit budget

Validated on controlled or synthetic data. Not production libraries.

## Setup

Python 3.12.

```
pip install -r requirements.txt
```

- Deterministic, fixed seed. Prints PASS/FAIL.
- Probes a local OpenAI-compatible gateway at `127.0.0.1:8000` (optional, non-gating).
- Offline: `RAGTOOLKIT_OFFLINE=1` or `--offline`.
- `python verify.py` runs both experiments and the tests.
- Tests need `pytest`: `pip install -r requirements-dev.txt`. Without it, `verify.py` skips them and says so.

## QSLM

Quantized vs weight-changed model, including matched residual magnitude. Recovers bit-width `b*`.

```
python qslm/run_experiment.py
```

Numpy embedder, fixed seed, exit 0:

| signal | value |
|---|---|
| tri-condition accuracy | 0.917 (0 false positives) |
| AUC, subspace-overlap / effective-rank | 1.000 / 1.000 |
| AUC, energy-only baseline (magnitude) | 0.491 (≈ chance) |
| bit-width recovery error | MAE 0.107 bits |

Scope: numpy MLP with a per-block k-quant quantizer. Real GGUF model not done.

## BD-APP

Predicts which GraphRAG false-merge poisons answers, by token-budget displacement. Displacement-weighted repair under a fixed edit budget.

```
python bdapp/run_experiment.py
```

Synthetic corpus, fixed seed, exit 0:

| signal | value (n≈12 seeds, 3 merges, K=6; small sample) |
|---|---|
| centrality inversion | PR ranks the peripheral merge (14.4, 75% real harm) above the central one (0.0, 0% harm); centrality ranks them backwards |
| calibration AUC | ~1.0 |
| repair @ equal budget K=6 | displacement-cut ~33% flips / ~0.33 precision vs centrality & dedup both 100% / 0.0 |

Rounded; exact output in [bdapp/README.md](bdapp/README.md).

Scope:
- Synthetic corpus.
- Deterministic label-free reader, no live LLM.
- AUC 1.0 on clean synthetic corpus, independent reader.
- Optional real-LLM labelling pass not done.

[MIT-licensed](LICENSE).
