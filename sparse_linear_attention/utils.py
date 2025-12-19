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

Utility functions for Sparse Linear Attention.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def compress_kernel(
    X, XM,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """
    Mean pooling kernel for compressing sequence blocks.

    Computes the mean of each block along the sequence dimension.
    """
    idx_l = tl.program_id(0)
    idx_bh = tl.program_id(1)

    offs_l = idx_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_d = tl.arange(0, D)

    x_offset = idx_bh * L * D
    xm_offset = idx_bh * ((L + BLOCK_L - 1) // BLOCK_L) * D
    x = tl.load(X + x_offset + offs_l[:, None] * D + offs_d[None, :], mask=offs_l[:, None] < L, other=0)

    nx = min(BLOCK_L, L - idx_l * BLOCK_L)
    x_mean = tl.sum(x, axis=0, dtype=tl.float32) / nx
    tl.store(XM + xm_offset + idx_l * D + offs_d, x_mean.to(XM.dtype.element_ty))


def mean_pool(x, BLK):
    """
    Mean pool a tensor along the sequence dimension.

    Args:
        x: Input tensor of shape [B, H, L, D]
        BLK: Block size for pooling

    Returns:
        Pooled tensor of shape [B, H, L_BLOCKS, D]
    """
    assert x.is_contiguous()

    B, H, L, D = x.shape
    L_BLOCKS = (L + BLK - 1) // BLK
    x_mean = torch.empty((B, H, L_BLOCKS, D), device=x.device, dtype=x.dtype)

    grid = (L_BLOCKS, B * H)
    compress_kernel[grid](x, x_mean, L, D, BLK)
    return x_mean


def get_block_map(q, k, topk_ratio, BLKQ=64, BLKK=64):
    """
    Compute the sparse block map using mean-pooled attention scores.

    This function determines which KV blocks are most relevant to each Q block
    by computing attention scores at the block level using mean pooling.

    Args:
        q: Query tensor of shape [B, H, L, D]
        k: Key tensor of shape [B, H, L, D]
        topk_ratio: Ratio of top KV blocks to select (0.0 to 1.0)
        BLKQ: Block size for queries
        BLKK: Block size for keys

    Returns:
        sparse_map: Binary mask of shape [B, H, Q_BLOCKS, K_BLOCKS] where 1 indicates selected
        lut: Look-up table of selected block indices, shape [B, H, Q_BLOCKS, topk]
        topk: Actual number of blocks selected
    """
    # Apply smooth-k technique from SageAttention for better block importance estimation
    arg_k = k - torch.mean(k, dim=-2, keepdim=True)

    # Mean pool Q and K to block level
    pooled_qblocks = mean_pool(q, BLKQ)
    pooled_kblocks = mean_pool(arg_k, BLKK)

    # Compute block-level attention scores
    pooled_score = pooled_qblocks @ pooled_kblocks.transpose(-1, -2)

    # Select top-k blocks
    K = pooled_score.shape[-1]
    topk = min(K, int(topk_ratio * K))
    lut = torch.topk(pooled_score, topk, dim=-1, sorted=False).indices

    # Create sparse map
    sparse_map = torch.zeros_like(pooled_score, dtype=torch.int8)
    sparse_map.scatter_(-1, lut, 1)

    return sparse_map, lut, topk
