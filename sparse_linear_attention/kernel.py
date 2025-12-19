"""
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License")

Citation (please cite if you use this code):

@article{zhang2025sla,
  title={SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention},
  author={Jintao Zhang and Haoxu Wang and Kai Jiang and Shuo Yang and Kaiwen Zheng and Haocheng Xi and Ziteng Wang and Hongzhou Zhu and Min Zhao and Ion Stoica and Joseph E. Gonzalez and Jun Zhu and Jianfei Chen},
  journal={arXiv preprint arXiv:2509.24006},
  year={2025}
}

Modified: Sparse Linear Attention (SLA-NEW)
- Removed Softmax branch entirely
- Mask=1: Load precomputed h_j, z_j and accumulate (Linear Attention)
- Mask=0: Skip (no memory load, no computation)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd_preprocess(
    CK, V, H, Z,
    L: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    CD: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    """
    Precompute h_j = (K^phi_j)^T V_j and z_j = rowsum((K^phi_j)^T) for each KV block.

    Args:
        CK: Feature-mapped keys K^phi, shape [B*H, L, CD]
        V: Values, shape [B*H, L, D]
        H: Output tensor for h_j = (K^phi)^T V, shape [B*H, N_BLOCKS, CD, D]
        Z: Output tensor for z_j = rowsum((K^phi)^T), shape [B*H, N_BLOCKS, CD]
    """
    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    CK += idx_bh * L * CD
    V += idx_bh * L * D
    H += (idx_bh * N_BLOCKS + idx_n) * CD * D
    Z += (idx_bh * N_BLOCKS + idx_n) * CD

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_cd = tl.arange(0, CD)

    # Load K^phi block: [BLOCK_N, CD] -> transpose to [CD, BLOCK_N]
    c_k = tl.load(CK + offs_n[None, :] * CD + offs_cd[:, None], mask=offs_n[None, :] < L, other=0)
    # Load V block: [BLOCK_N, D]
    v = tl.load(V + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < L, other=0)

    # h_j = (K^phi)^T @ V, shape [CD, D]
    h = tl.dot(c_k, v).to(H.type.element_ty)
    # z_j = rowsum((K^phi)^T) = sum over BLOCK_N, shape [CD]
    z = tl.sum(c_k, axis=1).to(Z.type.element_ty)

    tl.store(H + offs_cd[:, None] * D + offs_d[None, :], h)
    tl.store(Z + offs_cd, z)


@triton.jit
def _attn_fwd(
    CQ, H, Z, LUT, DENOM, O, H_ACC, Z_ACC,
    topk: tl.constexpr,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    CD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Sparse Linear Attention forward kernel.

    For each query block i:
      - Iterate through LUT (Mask=1 blocks only)
      - Load precomputed h_j, z_j from HBM
      - Accumulate: H_i += h_j, Z_i += z_j
      - Compute output: O_i = (Q^phi_i @ H_i) / (Q^phi_i @ Z_i)

    Args:
        CQ: Feature-mapped queries Q^phi, shape [B*H, L, CD]
        H: Precomputed h blocks, shape [B*H, N_BLOCKS, CD, D]
        Z: Precomputed z blocks, shape [B*H, N_BLOCKS, CD]
        LUT: Look-up table of selected KV block indices, shape [B*H, M_BLOCKS, topk]
        DENOM: Output denominator for backward, shape [B*H, L]
        O: Output tensor, shape [B*H, L, D]
        H_ACC: Output accumulated H per query block, shape [B*H, M_BLOCKS, CD, D]
        Z_ACC: Output accumulated Z per query block, shape [B*H, M_BLOCKS, CD]
    """
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    cq_offset = idx_bh * L * CD
    h_offset = idx_bh * N_BLOCKS * CD * D
    z_offset = idx_bh * N_BLOCKS * CD
    lut_offset = (idx_bh * M_BLOCKS + idx_m) * topk
    o_offset = idx_bh * L * D
    denom_offset = idx_bh * L
    h_acc_offset = (idx_bh * M_BLOCKS + idx_m) * CD * D
    z_acc_offset = (idx_bh * M_BLOCKS + idx_m) * CD

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_cd = tl.arange(0, CD)

    CQ_ptrs = CQ + cq_offset + offs_m[:, None] * CD + offs_cd[None, :]
    H_ptrs = H + h_offset + offs_cd[:, None] * D + offs_d[None, :]
    Z_ptrs = Z + z_offset + offs_cd
    LUT_ptr = LUT + lut_offset
    O_ptrs = O + o_offset + offs_m[:, None] * D + offs_d[None, :]
    DENOM_ptrs = DENOM + denom_offset + offs_m
    H_ACC_ptrs = H_ACC + h_acc_offset + offs_cd[:, None] * D + offs_d[None, :]
    Z_ACC_ptrs = Z_ACC + z_acc_offset + offs_cd

    # Initialize accumulators for H_i and Z_i
    h_acc = tl.zeros([CD, D], dtype=tl.float32)
    z_acc = tl.zeros([CD], dtype=tl.float32)

    # Sparse accumulation: only load and accumulate Mask=1 blocks from LUT
    for block_idx in tl.range(topk):
        idx_n = tl.load(LUT_ptr + block_idx)

        # Load precomputed h_j and z_j from HBM
        h_j = tl.load(H_ptrs + idx_n * CD * D)
        z_j = tl.load(Z_ptrs + idx_n * CD)

        # Accumulate: H_i += h_j, Z_i += z_j
        h_acc += h_j.to(tl.float32)
        z_acc += z_j.to(tl.float32)

    # Load Q^phi for this query block
    c_q = tl.load(CQ_ptrs, mask=offs_m[:, None] < L, other=0)

    # Compute denominator: denom = Q^phi @ Z, shape [BLOCK_M]
    if topk > 0:
        denom = tl.sum(c_q.to(tl.float32) * z_acc[None, :], axis=1)
    else:
        denom = tl.full([BLOCK_M], float("inf"), dtype=tl.float32)

    # Compute output: O = (Q^phi @ H) / denom
    o = tl.dot(c_q.to(tl.float32), h_acc) / tl.maximum(denom[:, None], 1e-6)

    tl.store(O_ptrs, o.to(O.type.element_ty), mask=offs_m[:, None] < L)
    tl.store(DENOM_ptrs, denom, mask=offs_m < L)
    # Store accumulated H and Z for backward pass
    tl.store(H_ACC_ptrs, h_acc.to(H_ACC.type.element_ty))
    tl.store(Z_ACC_ptrs, z_acc.to(Z_ACC.type.element_ty))


@triton.jit
def _attn_bwd_preprocess(
    CQ, O, DO, DENOM, DELTA, QO,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    CD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Backward preprocess: compute delta and QO for each query block.

    delta_i = rowsum(dO_i * O_i)
    QO_i = (Q^phi_i / denom_i)^T @ dO_i  (for dH computation)
    """
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    CQ += idx_bh * L * CD
    O += idx_bh * L * D
    DO += idx_bh * L * D
    DENOM += idx_bh * L
    DELTA += idx_bh * L
    QO += (idx_bh * M_BLOCKS + idx_m) * CD * D

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_cd = tl.arange(0, CD)

    c_q = tl.load(CQ + offs_m[None, :] * CD + offs_cd[:, None], mask=offs_m[None, :] < L, other=0)
    o = tl.load(O + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L, other=0)
    do = tl.load(DO + offs_m[:, None] * D + offs_d[None, :], mask=offs_m[:, None] < L, other=0)
    denom = tl.load(DENOM + offs_m, mask=offs_m < L, other=float('inf'))

    # delta = rowsum(dO * O)
    delta = tl.sum(o * do, axis=1).to(DELTA.type.element_ty)

    # QO = (Q^phi / denom)^T @ dO, shape [CD, D]
    c_q_normalized = c_q / tl.maximum(denom[None, :], 1e-6)
    qo = tl.dot(c_q_normalized.to(do.dtype), do).to(QO.type.element_ty)

    tl.store(DELTA + offs_m, delta, mask=offs_m < L)
    tl.store(QO + offs_cd[:, None] * D + offs_d[None, :], qo)


@triton.jit
def _attn_bwd_dcq(
    H_ACC, Z_ACC, DENOM, DO, DELTA, DCQ,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    CD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Compute gradient w.r.t. Q^phi (feature-mapped query).

    dQ^phi_i = (dO_i @ H_i^T - delta_i * Z_i^T) / denom_i
    """
    idx_m = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    h_acc_offset = (idx_bh * M_BLOCKS + idx_m) * CD * D
    z_acc_offset = (idx_bh * M_BLOCKS + idx_m) * CD
    do_offset = idx_bh * L * D
    denom_offset = idx_bh * L
    delta_offset = idx_bh * L
    dcq_offset = idx_bh * L * CD

    offs_m = idx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_cd = tl.arange(0, CD)

    H_ACC_ptrs = H_ACC + h_acc_offset + offs_cd[None, :] * D + offs_d[:, None]
    Z_ACC_ptrs = Z_ACC + z_acc_offset + offs_cd
    DO_ptrs = DO + do_offset + offs_m[:, None] * D + offs_d[None, :]
    DENOM_ptrs = DENOM + denom_offset + offs_m
    DELTA_ptrs = DELTA + delta_offset + offs_m
    DCQ_ptrs = DCQ + dcq_offset + offs_m[:, None] * CD + offs_cd[None, :]

    # Load accumulated H_i and Z_i for this query block
    h_acc = tl.load(H_ACC_ptrs)  # [D, CD]
    z_acc = tl.load(Z_ACC_ptrs)  # [CD]
    do = tl.load(DO_ptrs, mask=offs_m[:, None] < L, other=0)  # [BLOCK_M, D]
    denom = tl.load(DENOM_ptrs, mask=offs_m < L, other=float('inf'))  # [BLOCK_M]
    delta = tl.load(DELTA_ptrs, mask=offs_m < L, other=0)  # [BLOCK_M]

    # dQ^phi = (dO @ H^T - delta * Z^T) / denom
    # dO @ H^T: [BLOCK_M, D] @ [D, CD] = [BLOCK_M, CD]
    do_h = tl.dot(do, h_acc.to(do.dtype))
    # delta * Z^T: [BLOCK_M, 1] * [1, CD] = [BLOCK_M, CD]
    delta_z = delta[:, None] * z_acc[None, :]

    dc_q = (do_h - delta_z) / tl.maximum(denom[:, None], 1e-6)

    tl.store(DCQ_ptrs, dc_q.to(DCQ.type.element_ty), mask=offs_m[:, None] < L)


@triton.jit
def _attn_bwd_dhdz(
    CQ, DO, DELTA, DENOM, KBID,
    DH, DZ,
    L: tl.constexpr,
    M_BLOCKS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    CD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SLICE_FACTOR: tl.constexpr,
):
    """
    Compute gradients w.r.t. precomputed h_j and z_j.

    For each KV block j, accumulate from all query blocks i that selected it:
    dh_j += (Q^phi_i / denom_i)^T @ dO_i
    dz_j += -(Q^phi_i / denom_i)^T @ delta_i
    """
    BLOCK_M2: tl.constexpr = BLOCK_M // BLOCK_SLICE_FACTOR

    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    offs_m = tl.arange(0, BLOCK_M2)
    offs_d = tl.arange(0, D)
    offs_cd = tl.arange(0, CD)

    cq_offset = idx_bh * L * CD
    do_offset = idx_bh * L * D
    denom_offset = idx_bh * L
    delta_offset = idx_bh * L
    kbid_offset = idx_bh * M_BLOCKS * N_BLOCKS
    dh_offset = (idx_bh * N_BLOCKS + idx_n) * CD * D
    dz_offset = (idx_bh * N_BLOCKS + idx_n) * CD

    CQ_ptrs = CQ + cq_offset + offs_m[None, :] * CD + offs_cd[:, None]
    DO_ptrs = DO + do_offset + offs_m[:, None] * D + offs_d[None, :]
    DENOM_ptrs = DENOM + denom_offset + offs_m
    DELTA_ptrs = DELTA + delta_offset + offs_m
    # Use base pointer for KBID - compute offset from loop index to avoid
    # compiler optimization issues with pointer updates in loops
    KBID_base = KBID + kbid_offset + idx_n
    DH_ptrs = DH + dh_offset + offs_cd[:, None] * D + offs_d[None, :]
    DZ_ptrs = DZ + dz_offset + offs_cd

    dh = tl.zeros([CD, D], dtype=tl.float32)
    dz = tl.zeros([CD], dtype=tl.float32)

    for idx_m in tl.range(0, L, BLOCK_M2):
        # Compute query block index from loop variable
        query_block_idx = idx_m // BLOCK_M
        # Load kbid from computed offset (not from updated pointer)
        kbid = tl.load(KBID_base + query_block_idx * N_BLOCKS)
        if kbid == 1:  # This KV block was selected by this query block
            m_mask = offs_m < L - idx_m

            c_q = tl.load(CQ_ptrs, mask=m_mask[None, :], other=0)  # [CD, BLOCK_M2]
            do = tl.load(DO_ptrs, mask=m_mask[:, None], other=0)   # [BLOCK_M2, D]
            denom = tl.load(DENOM_ptrs, mask=m_mask, other=float('inf'))  # [BLOCK_M2]
            delta = tl.load(DELTA_ptrs, mask=m_mask, other=0)  # [BLOCK_M2]

            # Normalize Q^phi by denominator
            c_q_norm = c_q / tl.maximum(denom[None, :], 1e-6)  # [CD, BLOCK_M2]

            # dh += (Q^phi / denom)^T @ dO = c_q_norm @ do
            dh += tl.dot(c_q_norm.to(do.dtype), do)

            # dz += -(Q^phi / denom)^T @ delta
            dz -= tl.sum(c_q_norm * delta[None, :], axis=1)

        # Increment pointers
        CQ_ptrs += BLOCK_M2 * CD
        DO_ptrs += BLOCK_M2 * D
        DENOM_ptrs += BLOCK_M2
        DELTA_ptrs += BLOCK_M2

    tl.store(DH_ptrs, dh.to(DH.type.element_ty))
    tl.store(DZ_ptrs, dz.to(DZ.type.element_ty))


@triton.jit
def _attn_bwd_dckdv(
    CK, V, DH, DZ, DCK, DV,
    L: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    D: tl.constexpr,
    CD: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    """
    Compute gradients w.r.t. K^phi and V from gradients of h and z.

    From h_j = (K^phi_j)^T @ V_j:
      dK^phi_j = V_j @ dh_j^T
      dV_j = K^phi_j @ dh_j

    From z_j = rowsum((K^phi_j)^T):
      dK^phi_j += broadcast(dz_j)
    """
    idx_n = tl.program_id(0).to(tl.int64)
    idx_bh = tl.program_id(1).to(tl.int64)

    CK += idx_bh * L * CD
    V += idx_bh * L * D
    DH += (idx_bh * N_BLOCKS + idx_n) * CD * D
    DZ += (idx_bh * N_BLOCKS + idx_n) * CD
    DCK += idx_bh * L * CD
    DV += idx_bh * L * D

    offs_n = idx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D)
    offs_cd = tl.arange(0, CD)

    # Load K^phi, V, dH, dZ
    c_k = tl.load(CK + offs_n[:, None] * CD + offs_cd[None, :], mask=offs_n[:, None] < L, other=0)  # [BLOCK_N, CD]
    v = tl.load(V + offs_n[:, None] * D + offs_d[None, :], mask=offs_n[:, None] < L, other=0)        # [BLOCK_N, D]
    dh = tl.load(DH + offs_cd[:, None] * D + offs_d[None, :])  # [CD, D]
    dz = tl.load(DZ + offs_cd)  # [CD]

    # dV = K^phi @ dh: [BLOCK_N, CD] @ [CD, D] = [BLOCK_N, D]
    dv = tl.dot(c_k, dh.to(c_k.dtype))

    # dK^phi = V @ dh^T + dz (broadcast)
    # V @ dh^T: [BLOCK_N, D] @ [D, CD] = [BLOCK_N, CD]
    dh_T = tl.trans(dh)  # [D, CD]
    dc_k = tl.dot(v, dh_T.to(v.dtype)) + dz[None, :]  # broadcast dz to all rows

    tl.store(DCK + offs_n[:, None] * CD + offs_cd[None, :], dc_k.to(DCK.type.element_ty), mask=offs_n[:, None] < L)
    tl.store(DV + offs_n[:, None] * D + offs_d[None, :], dv.to(DV.type.element_ty), mask=offs_n[:, None] < L)


class _sparse_linear_attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c_q, c_k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N):
        """
        Forward pass of Sparse Linear Attention.

        Args:
            c_q: Feature-mapped queries Q^phi, shape [B, H, L, CD]
            c_k: Feature-mapped keys K^phi, shape [B, H, L, CD]
            v: Values, shape [B, H, L, D]
            k_block_id: Block mask indicating which KV blocks are selected, shape [B, H, M_BLOCKS, N_BLOCKS]
            lut: Look-up table of selected KV block indices, shape [B, H, M_BLOCKS, topk]
            topk: Number of KV blocks selected per query block
            BLOCK_M: Query block size
            BLOCK_N: KV block size

        Returns:
            o: Output tensor, shape [B, H, L, D]
        """
        assert c_q.is_contiguous() and c_k.is_contiguous() and v.is_contiguous()
        assert k_block_id.is_contiguous() and lut.is_contiguous()
        assert BLOCK_M == 64 or BLOCK_M == 128
        assert BLOCK_N == 64

        B, H, L, CD = c_q.shape
        D = v.shape[-1]

        M_BLOCKS = triton.cdiv(L, BLOCK_M)
        N_BLOCKS = triton.cdiv(L, BLOCK_N)

        # Output tensor
        o = torch.empty((B, H, L, D), device=v.device, dtype=v.dtype)
        denom = torch.empty((B, H, L), device=v.device, dtype=torch.float32)

        # Precomputed h and z for each KV block
        h = torch.empty((B, H, N_BLOCKS, CD, D), device=v.device, dtype=v.dtype)
        z = torch.empty((B, H, N_BLOCKS, CD), device=v.device, dtype=v.dtype)

        # Accumulated h and z per query block (computed in kernel, not Python)
        h_acc = torch.empty((B, H, M_BLOCKS, CD, D), device=v.device, dtype=v.dtype)
        z_acc = torch.empty((B, H, M_BLOCKS, CD), device=v.device, dtype=v.dtype)

        # Step 1: Precompute h_j = (K^phi_j)^T @ V_j and z_j = rowsum((K^phi_j)^T)
        grid_preprocess = (N_BLOCKS, B * H)
        _attn_fwd_preprocess[grid_preprocess](
            c_k, v, h, z,
            L, N_BLOCKS, D, CD, BLOCK_N
        )

        # Step 2: Sparse accumulation, output computation, and h_acc/z_acc computation
        # h_acc and z_acc are computed directly in the kernel (no Python gather+sum)
        grid_fwd = (M_BLOCKS, B * H)
        _attn_fwd[grid_fwd](
            c_q, h, z, lut, denom, o, h_acc, z_acc,
            topk, L, M_BLOCKS, N_BLOCKS, D, CD, BLOCK_M, BLOCK_N,
            num_warps=4 if D == 64 else 8,
            num_stages=3
        )

        ctx.save_for_backward(c_q, c_k, v, k_block_id, lut, denom, o, h, z, h_acc, z_acc)
        ctx.topk = topk
        ctx.BLOCK_M = BLOCK_M
        ctx.BLOCK_N = BLOCK_N
        return o

    @staticmethod
    def backward(ctx, do):
        c_q, c_k, v, k_block_id, lut, denom, o, h, z, h_acc, z_acc = ctx.saved_tensors
        do = do.contiguous()

        BLOCK_M, BLOCK_N = ctx.BLOCK_M, ctx.BLOCK_N
        B, H, L, CD = c_q.shape
        D = v.shape[-1]

        M_BLOCKS = triton.cdiv(L, BLOCK_M)
        N_BLOCKS = triton.cdiv(L, BLOCK_N)

        # Allocate gradient tensors
        dc_q = torch.empty_like(c_q)
        dc_k = torch.empty_like(c_k)
        dv = torch.empty_like(v)
        delta = torch.empty((B, H, L), device=c_q.device, dtype=torch.float32)
        qo = torch.empty((B, H, M_BLOCKS, CD, D), device=c_q.device, dtype=c_q.dtype)
        dh = torch.empty((B, H, N_BLOCKS, CD, D), device=c_q.device, dtype=torch.float32)
        dz = torch.empty((B, H, N_BLOCKS, CD), device=c_q.device, dtype=torch.float32)

        # Step 1: Preprocess - compute delta and QO
        grid_preprocess = (M_BLOCKS, B * H)
        _attn_bwd_preprocess[grid_preprocess](
            c_q, o, do, denom, delta, qo,
            L, M_BLOCKS, D, CD, BLOCK_M,
        )

        # Step 2: Compute dQ^phi
        grid_dcq = (M_BLOCKS, B * H)
        _attn_bwd_dcq[grid_dcq](
            h_acc, z_acc, denom, do, delta, dc_q,
            L, M_BLOCKS, D, CD, BLOCK_M,
            num_warps=4 if D == 64 else 8,
        )

        # Step 3: Compute dH and dZ for each KV block
        grid_dhdz = (N_BLOCKS, B * H)
        _attn_bwd_dhdz[grid_dhdz](
            c_q, do, delta, denom, k_block_id,
            dh, dz,
            L, M_BLOCKS, N_BLOCKS, D, CD, BLOCK_M, BLOCK_N,
            BLOCK_SLICE_FACTOR=BLOCK_M // 64,
            num_warps=4 if D == 64 else 8,
        )

        # Step 4: Compute dK^phi and dV from dH and dZ
        grid_dckdv = (N_BLOCKS, B * H)
        _attn_bwd_dckdv[grid_dckdv](
            c_k, v, dh, dz, dc_k, dv,
            L, N_BLOCKS, D, CD, BLOCK_N
        )

        return dc_q, dc_k, dv, None, None, None, None, None
