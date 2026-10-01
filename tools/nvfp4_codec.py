"""tools/nvfp4_codec.py - NVFP4 and Q8_0 for Strata's expert packs, on torch (CPU or CUDA).

NVFP4 (ModelOpt / ggml block_nvfp4): a value = E2M1 code x its 16-value block's UE4M3 scale x the tensor's FP32
global scale (ModelOpt's weight_scale_2, Strata's blk.N.ffn_*_exps.scale). ggml stores a row as blocks of 64 values:
4 scale bytes (UE4M3 = an E4M3 byte with the sign 0) then 32 code bytes, sub-block s holding values j (low nibble)
and j + 8 (high nibble) of its 16 in byte s*8 + j. ModelOpt packs a row's codes two per byte in order (value 2i low,
2i + 1 high). Q8_0 (ggml): blocks of 32 values, an FP16 scale amax/127 then 32 int8, round half away from zero.

Encoders return codes (int64 0..15, bit 3 = sign), scale bytes (uint8, E4M3) and the global scale; to_ggml() lays
them out as experts.bin rows. dequant() gives the float weights the engine computes with.
"""
from __future__ import annotations

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP8_MAX = 448.0
SUB = 16          # values per NVFP4 scale
QK = 64           # values per ggml block_nvfp4
QK8 = 32          # values per ggml block_q8_0


def e2m1_table(device) -> torch.Tensor:
    """The 16 code values, index = code (bit 3 = sign)."""
    t = E2M1.to(device)
    return torch.cat([t, -t])


def fp8_bytes(x: torch.Tensor) -> torch.Tensor:
    """Non-negative floats -> E4M3 bytes, round to nearest even, saturating at 448 (ModelOpt's cast)."""
    return x.clamp(min=0.0, max=FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def fp8_value(b: torch.Tensor) -> torch.Tensor:
    return b.view(torch.float8_e4m3fn).to(torch.float32)


def e2m1_round(y: torch.Tensor, ties: str = "even") -> torch.Tensor:
    """Scaled values -> codes. ties='even': a value halfway between two grid points takes the one with an even
    mantissa bit (IEEE round-to-nearest-even, as a cast to FP4 does); ties='first': the lower-magnitude point
    (ggml's best_index scans upward and keeps the first of two equal errors)."""
    a = y.abs().clamp(max=6.0)
    grid = E2M1.to(y.device)
    # distance to every grid point; argmin keeps the first (lower) index on exact ties
    d = (a.unsqueeze(-1) - grid).abs()
    idx = d.argmin(-1)
    if ties == "up":   # exact ties: the higher magnitude
        nxt = (idx + 1).clamp(max=7)
        tie = (d.gather(-1, nxt.unsqueeze(-1)).squeeze(-1) == d.gather(-1, idx.unsqueeze(-1)).squeeze(-1)) & (nxt != idx)
        idx = torch.where(tie, nxt, idx)
    if ties == "even":
        # exact ties between idx and idx+1: take the even code (codes 0,2,4,6 have mantissa bit 0)
        nxt = (idx + 1).clamp(max=7)
        tie = (d.gather(-1, nxt.unsqueeze(-1)).squeeze(-1) == d.gather(-1, idx.unsqueeze(-1)).squeeze(-1)) & (nxt != idx)
        idx = torch.where(tie & (idx % 2 == 1), nxt, idx)
    sign = (y < 0) & (idx != 0)
    return idx + 8 * sign.long()


def nvfp4_global_scale(w: torch.Tensor) -> float:
    """ModelOpt: weight_scale_2 = amax(W) / (6 * 448)."""
    return float(w.abs().max()) / (6.0 * FP8_MAX)


def nvfp4_rtn(w: torch.Tensor, s2: float | None = None, ties: str = "even"):
    """ModelOpt's NVFP4: block scale = E4M3(amax_block / (6 * s2)), codes = nearest E2M1 of w / (scale * s2)."""
    w = w.float()
    rows, cols = w.shape
    s2 = nvfp4_global_scale(w) if s2 is None else s2
    blk = w.view(rows, cols // SUB, SUB)
    sb = fp8_bytes(blk.abs().amax(-1) / (6.0 * s2))
    scale = fp8_value(sb) * s2
    y = blk / torch.where(scale == 0, torch.ones_like(scale), scale).unsqueeze(-1)
    codes = e2m1_round(y, ties).view(rows, cols)
    return codes, sb, s2


def nvfp4_dequant(codes: torch.Tensor, sb: torch.Tensor, s2: float) -> torch.Tensor:
    rows, cols = codes.shape
    v = e2m1_table(codes.device)[codes].view(rows, cols // SUB, SUB)
    return (v * (fp8_value(sb) * s2).unsqueeze(-1)).view(rows, cols)


def unpack_modelopt(packed: torch.Tensor) -> torch.Tensor:
    """ModelOpt's uint8 [rows, cols/2] (value 2i in the low nibble) -> codes [rows, cols]."""
    p = packed.to(torch.int64)
    return torch.stack([p & 0xF, p >> 4], dim=-1).view(packed.shape[0], -1)


def to_ggml(codes: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    """codes [rows, cols] + scale bytes [rows, cols/16] -> ggml block_nvfp4 rows, uint8 [rows, cols/64 * 36]."""
    rows, cols = codes.shape
    nb = cols // QK
    c = codes.view(rows, nb, 4, SUB)                       # block, sub-block, value
    qs = (c[..., :8] | (c[..., 8:] << 4)).to(torch.uint8)  # byte j: value j low, value j + 8 high
    d = sb.view(rows, nb, 4)
    return torch.cat([d, qs.view(rows, nb, 32)], dim=-1).view(rows, nb * 36)


def from_ggml(raw: torch.Tensor, cols: int):
    """ggml block_nvfp4 rows -> (codes, scale bytes); the inverse of to_ggml."""
    rows = raw.shape[0]
    nb = cols // QK
    b = raw.view(rows, nb, 36)
    sb = b[..., :4].reshape(rows, nb * 4)
    qs = b[..., 4:].to(torch.int64).view(rows, nb, 4, 8)
    codes = torch.cat([qs & 0xF, qs >> 4], dim=-1).view(rows, cols)
    return codes, sb


def q8_0(w: torch.Tensor) -> torch.Tensor:
    """ggml quantize_row_q8_0_ref: per 32 values d = amax / 127 (stored FP16), q = round(x / d) half away from zero.
    Returns uint8 rows [rows, cols/32 * 34]."""
    w = w.float()
    rows, cols = w.shape
    b = w.view(rows, cols // QK8, QK8)
    d = b.abs().amax(-1) / 127.0
    inv = torch.where(d > 0, 1.0 / d, torch.zeros_like(d))
    x = b * inv.unsqueeze(-1)
    q = (torch.sign(x) * torch.floor(x.abs() + 0.5)).clamp(-127, 127).to(torch.int8)
    dh = d.to(torch.float16).view(torch.uint8).view(rows, cols // QK8, 2)
    return torch.cat([dh, q.view(torch.uint8)], dim=-1).view(rows, (cols // QK8) * 34)


def q8_0_dequant(raw: torch.Tensor, cols: int) -> torch.Tensor:
    rows = raw.shape[0]
    b = raw.view(rows, cols // QK8, 34)
    d = b[..., :2].contiguous().view(torch.float16).float()
    q = b[..., 2:].contiguous().view(torch.int8).float()
    return (q * d).view(rows, cols)


def nvfp4_search(w: torch.Tensor, s2: float | None = None, h: torch.Tensor | None = None, targets=(6.0, 4.0),
                 neighbors: int = 2, ties: str = "even"):
    """NVFP4 with each 16-value block's scale chosen for the least (weighted) squared error: candidates are the E4M3
    codes for amax -> 6 and amax -> 4 ("Four Over Six") and `neighbors` codes on each side of both. h: per-column
    importance (e.g. E[x^2] of the inputs, activation-aware as AWQ weighs it); None = plain MSE."""
    w = w.float()
    rows, cols = w.shape
    s2 = nvfp4_global_scale(w) if s2 is None else s2
    blk = w.view(rows, cols // SUB, SUB)
    amax = blk.abs().amax(-1)
    cand = []
    for t in targets:
        b0 = fp8_bytes(amax / (t * s2)).to(torch.int64)
        for k in range(-neighbors, neighbors + 1):
            cand.append((b0 + k).clamp(0, 0x7E))
    cand = torch.stack(cand, dim=-1)                                     # [rows, nblk, C]
    scale = fp8_value(cand.to(torch.uint8)) * s2                         # [rows, nblk, C]
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    y = blk.unsqueeze(2) / safe.unsqueeze(-1)                            # [rows, nblk, C, 16]
    codes = e2m1_round(y, ties)
    q = e2m1_table(w.device)[codes] * scale.unsqueeze(-1)
    err = (q - blk.unsqueeze(2)) ** 2
    if h is not None:
        err = err * h.float().view(1, cols // SUB, 1, SUB)
    best = err.sum(-1).argmin(-1)                                        # [rows, nblk]
    sb = cand.gather(-1, best.unsqueeze(-1)).squeeze(-1).to(torch.uint8)
    codes = codes.gather(2, best.view(rows, -1, 1, 1).expand(-1, -1, 1, SUB)).squeeze(2).reshape(rows, cols)
    return codes, sb, s2


def _best_block_scale(blk: torch.Tensor, s2: torch.Tensor, h: torch.Tensor | None, targets=(6.0, 4.0),
                      neighbors: int = 2, ties: str = "even"):
    """The scale search for one 16-value block per leading index: blk [..., 16], s2 broadcastable to blk[..., 0],
    h [..., 16] or None. Returns (scale bytes [...], codes [..., 16])."""
    amax = blk.abs().amax(-1)
    cand = []
    for t in targets:
        b0 = fp8_bytes(amax / (t * s2)).to(torch.int64)
        for k in range(-neighbors, neighbors + 1):
            cand.append((b0 + k).clamp(0, 0x7E))
    cand = torch.stack(cand, dim=-1)                                     # [..., C]
    scale = fp8_value(cand.to(torch.uint8)) * s2.unsqueeze(-1)
    safe = torch.where(scale == 0, torch.ones_like(scale), scale)
    y = blk.unsqueeze(-2) / safe.unsqueeze(-1)                           # [..., C, 16]
    codes = e2m1_round(y, ties)
    q = e2m1_table(blk.device)[codes] * scale.unsqueeze(-1)
    err = (q - blk.unsqueeze(-2)) ** 2
    if h is not None:
        err = err * h.unsqueeze(-2)
    best = err.sum(-1).argmin(-1)                                        # [...]
    sb = cand.gather(-1, best.unsqueeze(-1)).squeeze(-1).to(torch.uint8)
    idx = best.unsqueeze(-1).unsqueeze(-1).expand(*best.shape, 1, SUB)
    return sb, codes.gather(-2, idx).squeeze(-2)


def nvfp4_gptq(w: torch.Tensor, H: torch.Tensor, percdamp: float = 0.01, blocksize: int = 128,
               headroom: float = 1.25, ties: str = "even"):
    """GPTQ onto the NVFP4 grid, batched over experts. w [E, rows, cols] (float32), H [E, cols, cols] the inputs'
    second moments (sum of weight^2 * x x^T over the tokens routed to each expert). Columns go left to right; each
    16-column group's scales are searched on the weights as GPTQ has updated them (weighted by diag H), and every
    column's rounding error is spread over the columns not yet quantized through the inverse Hessian.
    Returns codes [E, rows, cols], scale bytes [E, rows, cols/16], global scales [E] (amax * headroom / (6 * 448):
    room for the updates to grow past the original amax without saturating E4M3)."""
    E, rows, cols = w.shape
    W = w.float().clone()
    H = H.float().clone()
    s2 = W.abs().amax(dim=(1, 2)) * headroom / (6.0 * FP8_MAX)                    # [E]
    diag = torch.diagonal(H, dim1=1, dim2=2)
    dead = diag == 0
    diag[dead] = 1.0
    W.masked_fill_(dead.unsqueeze(1), 0.0)
    damp = percdamp * diag.mean(-1)                                              # [E]
    H += damp.view(E, 1, 1) * torch.eye(cols, device=H.device).unsqueeze(0)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    hd = torch.diagonal(H, dim1=1, dim2=2)                                       # importance for the scale search
    codes = torch.zeros((E, rows, cols), dtype=torch.int64, device=w.device)
    sbs = torch.zeros((E, rows, cols // SUB), dtype=torch.uint8, device=w.device)
    tbl = e2m1_table(w.device)
    s2v = s2.view(E, 1)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, :, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[:, i1:i2, i1:i2]
        scale = None
        for i in range(i2 - i1):
            col = i1 + i
            if col % SUB == 0:
                g = col // SUB
                hg = hd[:, col:col + SUB].unsqueeze(1).expand(E, rows, SUB)
                sb, _ = _best_block_scale(W1[:, :, i:i + SUB], s2v.expand(E, rows), hg, ties=ties)
                sbs[:, :, g] = sb
                scale = fp8_value(sb) * s2v                                       # [E, rows]
            x = W1[:, :, i]
            safe = torch.where(scale == 0, torch.ones_like(scale), scale)
            c = e2m1_round(x / safe, ties)
            codes[:, :, col] = c
            q = tbl[c] * scale
            err = (x - q) / Hinv1[:, i, i].unsqueeze(1)                          # [E, rows]
            W1[:, :, i:] -= err.unsqueeze(2) * Hinv1[:, i, i:].unsqueeze(1)
            Err1[:, :, i] = err
        W[:, :, i2:] -= torch.bmm(Err1, Hinv[:, i1:i2, i2:])
    return codes, sbs, s2


def nvfp4_dequant_batched(codes: torch.Tensor, sb: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """[E, rows, cols] codes + [E, rows, cols/16] scale bytes + [E] global scales -> float weights."""
    E, rows, cols = codes.shape
    v = e2m1_table(codes.device)[codes].view(E, rows, cols // SUB, SUB)
    return (v * (fp8_value(sb) * s2.view(E, 1, 1)).unsqueeze(-1)).view(E, rows, cols)
