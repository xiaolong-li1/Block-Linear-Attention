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

Sparse Linear Attention (SLA-NEW)
================================

This is a modified version of SLA that removes the Softmax attention branch entirely.
Only sparse Linear Attention is performed on selected KV blocks.

Key differences from original SLA:
- Mask=1 blocks: Perform Linear Attention (accumulate precomputed h_j, z_j)
- Mask=0 blocks: Skip entirely (no memory load, no computation)
- Single output (no O_s + O_l combination)

Usage:
    from sparse_linear_attention import SparseLinearAttention

    attn = SparseLinearAttention(
        head_dim=64,
        topk=0.5,  # Select 50% of KV blocks
        feature_map='softmax'
    )

    output = attn(q, k, v)  # q, k, v: [B, H, L, D]
"""

from .core import (
    SparseLinearAttention,
    SparseLinearAttentionWithProjection,
)

__all__ = [
    "SparseLinearAttention",
    "SparseLinearAttentionWithProjection",
]
