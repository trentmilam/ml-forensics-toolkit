"""Deterministic CONTROLLED embedder: a fixed-seed numpy MLP.

W_ref are the REFERENCE weights (W1, W2). Inputs are fixed synthetic probes, each
hashed (via a per-index seeded RNG) to input features. The forward pass is
2 linear layers with a tanh nonlinearity, output L2-normalized to a 128-d unit vector.

This is a white-box stand-in for a real GGUF model: because we own the reference
weights, we can synthesize the entire self-quantization family {Q_b(W_ref)}. The
real-model upgrade path (using an actual GGUF and its k-quant variants) is described
in the README; the METHOD is identical.
"""
import numpy as np

D_IN = 64
D_HIDDEN = 256
D_OUT = 128


def make_reference_weights(seed):
    """Build reference weights with a mildly anisotropic output spectrum.

    A mild singular-value decay on W2 gives the network a dominant response
    subspace (as real trained networks have). This is a stated controlled
    property, not a rig: the separation the method exploits is that DENSE
    isotropic quant error excites this dominant subspace consistently while a
    RANDOM low-rank finetune delta excites a random subspace.
    """
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((D_HIDDEN, D_IN)) / np.sqrt(D_IN)

    # W2 with an imposed, mild singular-value decay (exponent 0.5).
    A = rng.standard_normal((D_OUT, D_OUT))
    B = rng.standard_normal((D_HIDDEN, D_HIDDEN))
    Uo, _ = np.linalg.qr(A)
    Vo, _ = np.linalg.qr(B)
    r = min(D_OUT, D_HIDDEN)
    sv = (np.arange(1, r + 1)) ** (-0.5)
    S = np.zeros((D_OUT, D_HIDDEN))
    S[:r, :r] = np.diag(sv)
    W2 = Uo @ S @ Vo.T
    W2 = W2 / np.sqrt(D_HIDDEN)
    return {"W1": W1, "W2": W2}


def make_probes(seed, n_probes):
    """Fixed synthetic probes: probe i hashed via default_rng([seed, i]) to features."""
    X = np.empty((n_probes, D_IN))
    for i in range(n_probes):
        r = np.random.default_rng([seed, i])
        v = r.standard_normal(D_IN)
        X[i] = v / (np.linalg.norm(v) + 1e-12)
    return X


def embed(W, X):
    """Forward pass: tanh MLP, L2-normalized 128-d output. W = {'W1','W2'}."""
    A1 = np.tanh(X @ W["W1"].T)
    Z2 = A1 @ W["W2"].T
    norm = np.linalg.norm(Z2, axis=1, keepdims=True)
    return Z2 / (norm + 1e-12)
