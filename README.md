# ml-forensics-toolkit: two research-prototype ML methods

This repo holds two small research experiments about catching problems in machine learning
systems. **QSLM** looks at a model's weights and tries to tell whether they were compressed as
expected or secretly altered. **BD-APP** looks at a retrieval system built on a knowledge graph
and predicts which of its entity-merging mistakes will actually corrupt an answer.

The two methods are independent: they don't share code, an API, or even a problem domain. One is
about the integrity of a model's weights, the other about the correctness of a retrieval graph.
They're published together as a two-method portfolio pair, not as a single product; each has its
own directory and its own README, so read the one you care about and ignore the other if it's not
relevant to you.

- QSLM is for anyone who deploys quantized models and needs to know whether an observed model
  differs from a known-good reference because it was quantized (compressed to a smaller
  bit-width, which is expected and changes the weights in a predictable way) or because its
  weights were changed outright (a stealth edit), including the hard case where both changes
  have the same residual magnitude, where simple magnitude/energy checks can't tell them apart.
- BD-APP is for anyone running a GraphRAG pipeline (a retrieval system that organizes source
  documents into a knowledge graph before answering questions) who has to worry about it silently
  merging two different real-world entities into one node, for example combining two different
  people who share a name. It predicts which such entity-merges will actually poison (corrupt) an
  answer, rather than just which ones look risky by graph centrality or dedup confidence, and
  repairs the graph under a fixed edit budget.

Both are research prototypes validated on controlled or synthetic data, not production libraries;
see each method's "scope" section below for exactly what is and isn't shown. Both were informally
checked against existing methods before writing this up. That check wasn't recorded step by step
and isn't presented as a reproducible result, unlike every measured number below, which is.

## Setup

Tested on Python 3.12. From the repo root:

```
pip install -r requirements.txt
```

This is all you need to run either experiment below. Each is deterministic (fixed seed) and
prints a measured PASS/FAIL verdict. Both scripts also probe a local OpenAI-compatible gateway at
`127.0.0.1:8000` by default as an optional, non-gating enrichment; set `RAGTOOLKIT_OFFLINE=1`
(or pass `--offline`) for a fully network-free run. See [qslm/README.md](qslm/README.md) and
[bdapp/README.md](bdapp/README.md) for details.

`python verify.py` runs both experiments plus the test suite in `tests/` and prints one aggregate
PASS/FAIL. The test suite needs `pytest`, which is a separate, optional dev dependency; install
it with `pip install -r requirements-dev.txt` first if you want `verify.py` to also run the tests.
If `pytest` isn't installed, `verify.py` skips the test suite and says so explicitly rather than
reporting a toolkit failure.

---

## QSLM: quantization-vs-weights typing ([qslm/README.md](qslm/README.md))

Decides whether a candidate model differs from a white-box reference because it was quantized
(and recovers the bit-width `b*`) or because its weights were changed, even at identical residual
magnitude, the regime where every energy/magnitude test provably fails. Mechanism: a
scheme-locked self-quantization family produces an energy-vs-bit curve, a cross-bit-invariant
residual subspace, and an effective-rank null, which together feed a tri-condition classifier.

```
python qslm/run_experiment.py
```

Measured (controlled numpy-embedder validation, fixed seed, exit 0):

| signal | value |
|---|---|
| tri-condition accuracy | 0.917 (0 false positives) |
| AUC, subspace-overlap / effective-rank | 1.000 / 1.000 |
| AUC, energy-only baseline (magnitude) | 0.491 (≈ chance; confirms magnitude alone can't tell the two apart) |
| bit-width recovery error | MAE 0.107 bits |

*Scope:* validates the method on a numpy MLP with a real per-block k-quant quantizer;
a real-GGUF model is the documented upgrade path.

## BD-APP: GraphRAG budget-displacement poisoning predictor and repair ([bdapp/README.md](bdapp/README.md))

Generator-free prediction of which GraphRAG false-merge will poison answers, driven by
token-budget displacement rather than centrality or dedup confidence, plus a
displacement-weighted repair under a fixed edit budget.

```
python bdapp/run_experiment.py
```

Measured (controlled synthetic corpus, fixed seed, exit 0):

| signal | value (n≈12 seeds, 3 merges, K=6; small sample, see note) |
|---|---|
| centrality inversion | confirmed: PR ranks the peripheral merge (14.4, 75% real harm) above the central one (0.0, 0% harm); centrality ranks them backwards |
| calibration AUC | ~1.0 |
| repair @ equal budget K=6 | displacement-cut ~33% flips / ~0.33 precision vs centrality & dedup both 100% / 0.0 |

*Small-sample note:* these are computed over 3 injected merges and ~12 seeds with a K=6 edit
budget; the numbers above are rounded because three-decimal precision (as printed verbatim by
`run_experiment.py`, e.g. `33.3%`/`0.333`/`1.000`) isn't statistically meaningful at this sample
size, so treat them as directional evidence of the effect, not stable estimates. See
[bdapp/README.md](bdapp/README.md) for the exact unrounded output.

*Scope:* synthetic corpus; flip-labels come from a **deterministic, label-free reader (no live LLM
required)**; the AUC=1.0 is a clean-corpus property measured against an *independent* reader (an
earlier label-leak was caught and fixed in adversarial review). An optional real-LLM labelling pass
is a documented upgrade.

---

## Honesty

Every number above was observed in real experiment output (exit 0), not asserted; both
experiments are deterministic (fixed seed, `numpy.random.default_rng`). The "informally checked
against existing methods" note above is exactly that: an informal check, not a formal or
third-party assessment, and not something this repo presents as a reproducible result. The perfect scores are on self-authored synthetic corpora, so treat them as
differentiation evidence, not external validation. Nothing here is investment, legal, or financial
advice. [MIT-licensed](LICENSE).
