"""Real per-block affine quantizer emulating GGUF k-quant style block quantization.

Each weight matrix is flattened, partitioned into fixed-size blocks; each block gets
its own affine (scale, zero-point) computed from that block's min/max, is quantized to
a bit-width b, then dequantized. This is a faithful (numpy) model of block-wise
integer weight quantization as used by GGUF k-quants.

Fractional average bit-widths are realized by MIXED per-block bit assignment (some
blocks at ceil(b), the rest at floor(b)), exactly as a k-quant mixed scheme would.
"""
import numpy as np


def assign_mixed_bits(b_float, n_blocks):
    """Return an integer array (len n_blocks) whose mean ~= b_float.

    Integer b -> all blocks at b. Fractional b -> first m blocks at ceil, rest floor.
    Bit-widths are clamped to the supported [3, 8] range.
    """
    b_float = float(np.clip(b_float, 3.0, 8.0))
    lo = int(np.floor(b_float))
    if lo == b_float:
        return np.full(n_blocks, lo, dtype=np.int64)
    hi = lo + 1
    frac = b_float - lo
    m = int(round(frac * n_blocks))
    bits = np.full(n_blocks, lo, dtype=np.int64)
    bits[:m] = hi
    return bits


def _quant_block(w, b):
    """Asymmetric affine quantize+dequantize a 1-D block to bit-width b."""
    levels = (1 << b) - 1  # 2^b - 1
    mn = w.min()
    mx = w.max()
    rng = mx - mn
    if rng < 1e-12:
        return w.copy()
    scale = rng / levels
    zp = np.round(-mn / scale)
    q = np.clip(np.round(w / scale + zp), 0, levels)
    return (q - zp) * scale


def quantize_dequant(W, block_size=32, bits=8):
    """Per-block affine quantize+dequantize a weight matrix.

    bits: either a scalar int (all blocks) or an int array of length n_blocks
    (mixed per-block assignment, for fractional average bit-widths).
    Returns a matrix of the same shape as W (the dequantized weights).
    """
    shape = W.shape
    flat = W.reshape(-1).astype(np.float64)
    n = flat.size
    pad = (-n) % block_size
    if pad:
        flat = np.concatenate([flat, np.zeros(pad)])
    blocks = flat.reshape(-1, block_size)
    n_blocks = blocks.shape[0]

    if np.isscalar(bits):
        bit_arr = np.full(n_blocks, int(bits), dtype=np.int64)
    else:
        bit_arr = np.asarray(bits, dtype=np.int64)
        assert bit_arr.size == n_blocks, (bit_arr.size, n_blocks)

    out = np.empty_like(blocks)
    for i in range(n_blocks):
        out[i] = _quant_block(blocks[i], int(bit_arr[i]))

    flat_out = out.reshape(-1)[:n]
    return flat_out.reshape(shape)


def n_blocks_for(W, block_size=32):
    n = W.size
    return (n + block_size - 1) // block_size


def quantize_matrix_bfloat(W, b_float, block_size=32):
    """Quantize a matrix to an average (possibly fractional) bit-width b_float
    using a mixed per-block assignment."""
    nb = n_blocks_for(W, block_size)
    bits = assign_mixed_bits(b_float, nb)
    return quantize_dequant(W, block_size=block_size, bits=bits)
