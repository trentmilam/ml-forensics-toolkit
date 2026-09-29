"""QSLM method: quantization-vs-weights typing via a scheme-locked self-quantization family.

Given a white-box reference model (W_ref) and a fixed probe set, we synthesize the
self-quantization family {Q_b(W_ref)} and derive three scheme-locked signatures:

  (curve) energy-vs-bit    : eps_b as a function of bit-width b (log-linear).
  (U)     invariant subspace: the residual DIRECTIONS consistent across b=8..3.
  (null)  residual null    : the distribution of a same-scheme residual statistic
                             (effective rank of the residual matrix).

A candidate model C is typed by comparing its residual against W_ref to these
signatures. It is QUANT iff (i) its energy lands on the curve at a recoverable b*,
(ii) its residual subspace aligns with U (principal-angle overlap >= tau), and
(iii) its residual effective-rank lies in the null band; else it is WEIGHTS.

The inventive point: a finetune delta scaled to the SAME residual energy as a quant
member (matched magnitude) defeats any energy-only test, but its residual is a
random-subspace, low-rank object that fails (ii) and (iii).
"""
import numpy as np

from embedder import embed
from quantizer import quantize_matrix_affine


# ---------------------------------------------------------------- residual stats
def residual_matrix(W_cand, W_ref, X, emb_ref=None):
    """R = emb(W_cand) - emb(W_ref), shape (n_probes, d_out)."""
    if emb_ref is None:
        emb_ref = embed(W_ref, X)
    emb_c = embed(W_cand, X)
    return emb_c - emb_ref


def energy(R):
    """Mean per-probe squared residual energy."""
    return float(np.mean(np.sum(R * R, axis=1)))


def top_subspace(R, k):
    """Top-k right singular vectors of R (directions in embedding space), (d_out, k)."""
    # R = U S Vt ; rows are probes, columns are embedding dims.
    _, _, Vt = np.linalg.svd(R, full_matrices=False)
    return Vt[:k].T


def effective_rank(R):
    """exp(entropy) of the squared singular-value spectrum (participation ratio)."""
    s = np.linalg.svd(R, compute_uv=False)
    p = (s ** 2)
    tot = p.sum()
    if tot < 1e-30:
        return 0.0
    p = p / tot
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(np.exp(H))


def subspace_overlap(Va, Vb):
    """Mean cos^2 of principal angles between two orthonormal subspaces (in [0,1]).

    For orthonormal Va (d,k1), Vb (d,k2): ||Va^T Vb||_F^2 = sum cos^2(theta_i).
    Normalized by min(k1,k2) to give a mean overlap in [0,1].
    """
    M = Va.T @ Vb
    val = np.sum(M * M)
    return float(val / min(Va.shape[1], Vb.shape[1]))


# --------------------------------------------------------- family + signatures
def build_family(W_ref, X, family_bits, block_size, k):
    """Quantize W1 and W2 at each integer bit-width; return per-member residual info."""
    emb_ref = embed(W_ref, X)
    members = {}
    for b in family_bits:
        Wq = {
            "W1": quantize_matrix_affine(W_ref["W1"], b, block_size),
            "W2": quantize_matrix_affine(W_ref["W2"], b, block_size),
        }
        R = residual_matrix(Wq, W_ref, X, emb_ref=emb_ref)
        members[b] = {
            "R": R,
            "eps": energy(R),
            "V": top_subspace(R, k),
            "erank": effective_rank(R),
        }
    return members, emb_ref


def fit_energy_curve(family_bits, eps_by_bit):
    """Fit ln(eps) = a - c*b over the family; return (a, c) and b_of_eps()."""
    b = np.asarray(family_bits, dtype=float)
    y = np.log(np.asarray([eps_by_bit[bb] for bb in family_bits], dtype=float))
    # least squares: y = a - c*b  ->  y = [1, -b] @ [a, c]
    A = np.column_stack([np.ones_like(b), -b])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, c = float(coef[0]), float(coef[1])

    def b_of_eps(eps):
        return (a - np.log(eps)) / c

    def resid_of(eps, bit):
        # residual of ln(eps) from the fitted line at bit -> "on curve" measure
        return abs(np.log(eps) - (a - c * bit))

    return a, c, b_of_eps, resid_of


def invariant_subspace(members, family_bits, k):
    """U = top-k eigenvectors of the scale-normalized mean residual covariance.

    Each member's residual is normalized to unit Frobenius norm before accumulation
    so that U captures DIRECTION consistency (not the largest-energy member).
    """
    d_out = members[family_bits[0]]["R"].shape[1]
    C = np.zeros((d_out, d_out))
    for b in family_bits:
        R = members[b]["R"]
        fn = np.linalg.norm(R)
        Rn = R / (fn + 1e-12)
        C += Rn.T @ Rn
    w, V = np.linalg.eigh(C)
    U = V[:, ::-1][:, :k]  # top-k
    return U


# --------------------------------------------------------------- candidates
def make_quant_candidate(W_ref, X, b_star, block_size, k, emb_ref):
    Wq = {
        "W1": quantize_matrix_affine(W_ref["W1"], b_star, block_size),
        "W2": quantize_matrix_affine(W_ref["W2"], b_star, block_size),
    }
    R = residual_matrix(Wq, W_ref, X, emb_ref=emb_ref)
    return {"R": R, "eps": energy(R), "V": top_subspace(R, k),
            "erank": effective_rank(R), "b_true": b_star, "label": 1}


def make_finetune_candidate(W_ref, X, target_eps, rank, rng, block_size, k, emb_ref):
    """W_ref + low-rank deltaW (both layers), scaled so residual energy == target_eps.

    The scale is found by iterating (nonlinearity makes energy not exactly quadratic
    in the scale): eps ~ s^2 -> s <- s * sqrt(target/eps_current), a few passes.
    """
    A1 = rng.standard_normal((W_ref["W1"].shape[0], rank))
    B1 = rng.standard_normal((rank, W_ref["W1"].shape[1]))
    A2 = rng.standard_normal((W_ref["W2"].shape[0], rank))
    B2 = rng.standard_normal((rank, W_ref["W2"].shape[1]))
    d1 = A1 @ B1
    d2 = A2 @ B2
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    s = 0.05
    for _ in range(6):
        Wf = {"W1": W_ref["W1"] + s * d1, "W2": W_ref["W2"] + s * d2}
        R = residual_matrix(Wf, W_ref, X, emb_ref=emb_ref)
        eps = energy(R)
        if eps < 1e-30:
            s *= 2
            continue
        s = s * np.sqrt(target_eps / eps)
    Wf = {"W1": W_ref["W1"] + s * d1, "W2": W_ref["W2"] + s * d2}
    R = residual_matrix(Wf, W_ref, X, emb_ref=emb_ref)
    return {"R": R, "eps": energy(R), "V": top_subspace(R, k),
            "erank": effective_rank(R), "b_true": None, "label": 0}


def make_ualigned_finetune_candidate(W_ref, X, target_eps, U, rng, block_size, k,
                                     emb_ref):
    """HARD (adaptive) adversary: a low-rank finetune delta STEERED so its embedding-space
    residual lands in the method's OWN cross-bit invariant subspace U, defeating the
    subspace-overlap condition (ii) that a naive random-delta finetune fails.

    Construction: a second-layer delta dW2 = U @ M0 forces the delta's output rows into
    span(U); propagated through the fixed network on the fixed probes the residual therefore
    aligns with U (measured overlap ~1.0, i.e. HIGHER than a real quant member). The delta is
    energy-matched to target_eps exactly like the random finetune. This is the substantive
    attack on non-obviousness: it targets the invariant subspace directly. Its structural
    weakness (and why the tri-condition survives) is that U is only k-dimensional, so
    confining the residual to span(U) caps its effective rank at k, so it cannot also
    satisfy (iii).
    """
    d_hidden = W_ref["W2"].shape[1]
    M0 = rng.standard_normal((U.shape[1], d_hidden))
    dW2 = U @ M0                       # (d_out, d_hidden), rank <= k; output rows in span(U)
    dW2 = dW2 / np.linalg.norm(dW2)

    s = 0.05
    for _ in range(8):
        Wf = {"W1": W_ref["W1"], "W2": W_ref["W2"] + s * dW2}
        R = residual_matrix(Wf, W_ref, X, emb_ref=emb_ref)
        eps = energy(R)
        if eps < 1e-30:
            s *= 2
            continue
        s = s * np.sqrt(target_eps / eps)
    Wf = {"W1": W_ref["W1"], "W2": W_ref["W2"] + s * dW2}
    R = residual_matrix(Wf, W_ref, X, emb_ref=emb_ref)
    return {"R": R, "eps": energy(R), "V": top_subspace(R, k),
            "erank": effective_rank(R), "b_true": None, "label": 0}


# --------------------------------------------------------------- classifier
def tri_condition_decision(cand, U, b_of_eps, resid_of, tau, curve_tol,
                           erank_lo, b_lo=2.5, b_hi=8.5):
    """Return (is_quant_bool, b_star_recovered, cond_i, cond_ii, cond_iii)."""
    b_rec = float(b_of_eps(cand["eps"]))
    cond_i = (b_lo <= b_rec <= b_hi) and (resid_of(cand["eps"], b_rec) <= curve_tol)
    ov = subspace_overlap(cand["V"], U)
    cond_ii = ov >= tau
    cond_iii = cand["erank"] >= erank_lo
    is_quant = bool(cond_i and cond_ii and cond_iii)
    return is_quant, b_rec, cond_i, cond_ii, cond_iii, ov


def auc(scores, labels):
    """AUC via the Mann-Whitney U statistic (rank-based; handles ties)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = scores[order]
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank (1-based)
        ranks[order[i:j + 1]] = avg
        i = j + 1
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    sum_ranks_pos = ranks[pos].sum()
    U_stat = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(U_stat / (n_pos * n_neg))
