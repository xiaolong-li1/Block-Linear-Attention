"""
Pure PyTorch Reference Implementation for Sparse Linear Attention.

This module provides a reference implementation to verify the correctness
of the Triton kernel in kernel.py.

The algorithm:
1. Preprocess: For each KV block j, compute:
   - h_j = (K^phi_j)^T @ V_j  [CD, D]
   - z_j = sum(K^phi_j, dim=0)  [CD]

2. Forward: For each query block i, using LUT to select KV blocks:
   - Accumulate: H_i = sum(h_j for j in LUT[i]), Z_i = sum(z_j for j in LUT[i])
   - Compute: O_i = (Q^phi_i @ H_i) / max(Q^phi_i @ Z_i, eps)
"""

import torch
import torch.nn.functional as F


def sparse_linear_attention_reference(c_q, c_k, v, k_block_id, lut, topk, BLOCK_M, BLOCK_N):
    """
    Pure PyTorch reference implementation of sparse linear attention.
    Uses autograd for automatic gradient computation.

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
    B, H, L, CD = c_q.shape
    D = v.shape[-1]

    M_BLOCKS = (L + BLOCK_M - 1) // BLOCK_M
    N_BLOCKS = (L + BLOCK_N - 1) // BLOCK_N

    # Step 1: Precompute h_j and z_j for each KV block
    # h_j = (K^phi_j)^T @ V_j, shape [B, H, N_BLOCKS, CD, D]
    # z_j = sum(K^phi_j, dim=0), shape [B, H, N_BLOCKS, CD]
    h_list = []
    z_list = []

    for n in range(N_BLOCKS):
        start_n = n * BLOCK_N
        end_n = min(start_n + BLOCK_N, L)

        # Get KV block: [B, H, block_len, CD] and [B, H, block_len, D]
        c_k_block = c_k[:, :, start_n:end_n, :]  # [B, H, block_len, CD]
        v_block = v[:, :, start_n:end_n, :]      # [B, H, block_len, D]

        # h_j = (K^phi)^T @ V = [B, H, CD, block_len] @ [B, H, block_len, D] = [B, H, CD, D]
        h_j = torch.einsum('bhsc,bhsd->bhcd', c_k_block, v_block)
        h_list.append(h_j)

        # z_j = sum(K^phi, dim=seq) = [B, H, CD]
        z_j = c_k_block.sum(dim=2)
        z_list.append(z_j)

    # Stack to get [B, H, N_BLOCKS, CD, D] and [B, H, N_BLOCKS, CD]
    h = torch.stack(h_list, dim=2)  # [B, H, N_BLOCKS, CD, D]
    z = torch.stack(z_list, dim=2)  # [B, H, N_BLOCKS, CD]

    # Step 2: Accumulate H_i and Z_i for each query block using LUT
    # Use advanced indexing to gather selected blocks
    o_blocks = []

    for m in range(M_BLOCKS):
        start_m = m * BLOCK_M
        end_m = min(start_m + BLOCK_M, L)

        # Get Q^phi block: [B, H, block_len, CD]
        c_q_block = c_q[:, :, start_m:end_m, :]

        if topk > 0:
            # Get indices for this query block: [B, H, topk]
            indices = lut[:, :, m, :topk]  # [B, H, topk]

            # Gather h and z for selected KV blocks
            # h: [B, H, N_BLOCKS, CD, D] -> gather on dim 2
            # Need to expand indices for gathering
            # indices shape: [B, H, topk] -> [B, H, topk, CD, D]
            indices_h = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, CD, D)
            h_selected = torch.gather(h, dim=2, index=indices_h)  # [B, H, topk, CD, D]
            h_acc = h_selected.sum(dim=2)  # [B, H, CD, D]

            # indices shape: [B, H, topk] -> [B, H, topk, CD]
            indices_z = indices.unsqueeze(-1).expand(-1, -1, -1, CD)
            z_selected = torch.gather(z, dim=2, index=indices_z)  # [B, H, topk, CD]
            z_acc = z_selected.sum(dim=2)  # [B, H, CD]

            # Compute denominator: denom = Q^phi @ Z, shape [B, H, block_len]
            denom = torch.einsum('bhsc,bhc->bhs', c_q_block, z_acc)  # [B, H, block_len]

            # Compute output: O = (Q^phi @ H) / denom
            numerator = torch.einsum('bhsc,bhcd->bhsd', c_q_block, h_acc)  # [B, H, block_len, D]

            # Normalize
            o_block = numerator / denom.unsqueeze(-1).clamp(min=1e-6)
        else:
            # No blocks selected - return zeros
            o_block = torch.zeros(B, H, end_m - start_m, D, device=v.device, dtype=v.dtype)

        o_blocks.append(o_block)

    # Concatenate along sequence dimension
    o = torch.cat(o_blocks, dim=2)  # [B, H, L, D]

    return o
