"""Direct-import tests for QSLM's claimed invariants (no subprocess, no network).

Imports embedder.py/quantizer.py/method.py directly and reconstructs the same
fixed-seed pipeline run_experiment.py prints, asserting the concrete numbers the
README claims as measured (accuracy, subspace/rank AUC, energy-baseline chance).

method.py is loaded via importlib under a unique module name ("qslm_method")
rather than a bare `import method` -- bdapp/method.py has the same filename, and
both test modules load in the same pytest process, so a bare import would risk
one package's tests silently picking up the other package's module from the
shared sys.modules cache.
"""
import importlib.util
import os
import sys

import numpy as np

QSLM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qslm")
if QSLM_DIR not in sys.path:
    sys.path.insert(0, QSLM_DIR)


def _load_module(unique_name, filename):
    path = os.path.join(QSLM_DIR, filename)
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[unique_name] = mod
    spec.loader.exec_module(mod)
    return mod


from embedder import make_reference_weights, make_probes  # noqa: E402  (unique name)
from quantizer import quantize_matrix_affine  # noqa: E402  (unique name)

M = _load_module("qslm_method", "method.py")

SEED = 20260703
N_PROBES = 400
BLOCK_SIZE = 32
K = 8
FAMILY_BITS = [8, 6, 5, 4, 3]
N_QUANT = 24
N_FINETUNE = 24
FT_RANK = 16


def _build_candidates():
    W_ref = make_reference_weights(SEED)
    X = make_probes(SEED, N_PROBES)
    members, emb_ref = M.build_family(W_ref, X, FAMILY_BITS, BLOCK_SIZE, K)
    eps_by_bit = {b: members[b]["eps"] for b in FAMILY_BITS}
    _, _, b_of_eps, resid_of = M.fit_energy_curve(FAMILY_BITS, eps_by_bit)
    U = M.invariant_subspace(members, FAMILY_BITS, K)

    fam_U_ov = [M.subspace_overlap(members[b]["V"], U) for b in FAMILY_BITS]
    tau = 0.7 * min(fam_U_ov)
    fit_res = [resid_of(eps_by_bit[b], b) for b in FAMILY_BITS]
    curve_tol = 3.0 * max(fit_res) + 0.75
    fam_erank = [members[b]["erank"] for b in FAMILY_BITS]
    erank_lo = 0.7 * min(fam_erank)

    rng = np.random.default_rng(SEED + 1)
    b_stars = np.round(np.linspace(3.2, 7.8, N_QUANT), 3)
    quant_cands = [M.make_quant_candidate(W_ref, X, float(b), BLOCK_SIZE, K, emb_ref)
                   for b in b_stars]
    ft_targets = [quant_cands[i % N_QUANT]["eps"] for i in range(N_FINETUNE)]
    ft_cands = [M.make_finetune_candidate(W_ref, X, t, FT_RANK, rng, BLOCK_SIZE, K, emb_ref)
                for t in ft_targets]

    cands = quant_cands + ft_cands
    labels = np.array([c["label"] for c in cands])
    return cands, labels, U, b_of_eps, resid_of, tau, curve_tol, erank_lo


def test_tri_condition_accuracy_and_aucs():
    cands, labels, U, b_of_eps, resid_of, tau, curve_tol, erank_lo = _build_candidates()

    overlaps, eranks, energies, preds = [], [], [], []
    for c in cands:
        is_q, _b_rec, _ci, _cii, _ciii, ov = M.tri_condition_decision(
            c, U, b_of_eps, resid_of, tau, curve_tol, erank_lo)
        overlaps.append(ov)
        eranks.append(c["erank"])
        energies.append(c["eps"])
        preds.append(1 if is_q else 0)
    preds = np.array(preds)

    acc = float(np.mean(preds == labels))
    auc_overlap = M.auc(overlaps, labels)
    auc_erank = M.auc(eranks, labels)
    auc_energy = M.auc(energies, labels)  # magnitude-only baseline -- must be ~chance
    false_positives = int(np.sum((labels == 0) & (preds == 1)))

    assert acc > 0.85, f"accuracy {acc} not > 0.85"
    assert auc_overlap > 0.99, f"subspace-overlap AUC {auc_overlap} not > 0.99"
    assert auc_erank > 0.99, f"effective-rank AUC {auc_erank} not > 0.99"
    assert abs(auc_energy - 0.5) < 0.15, f"energy-only baseline AUC {auc_energy} not ~chance"
    assert false_positives == 0, f"{false_positives} finetune candidates mistyped as quant"


def test_quantizer_affine_error_shrinks_with_more_bits():
    """The renamed quantize_matrix_affine (was quantize_matrix_bfloat) must still
    behave as a real per-block affine quantizer: more bits -> less error."""
    rng = np.random.default_rng(SEED)
    W = rng.standard_normal((64, 64))
    err8 = np.mean((quantize_matrix_affine(W, 8, BLOCK_SIZE) - W) ** 2)
    err4 = np.mean((quantize_matrix_affine(W, 4, BLOCK_SIZE) - W) ** 2)
    err3 = np.mean((quantize_matrix_affine(W, 3, BLOCK_SIZE) - W) ** 2)
    assert err8 < err4 < err3
