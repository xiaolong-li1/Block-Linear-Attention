"""
Copyright (c) 2025 by SLA team.

Licensed under the Apache License, Version 2.0 (the "License")

Sparse Linear Attention (Block Linear Attention)
================================================

Usage:
    from sparse_linear_attention import SparseLinearAttention

    attn = SparseLinearAttention(
        topk=0.5,  # Select 50% of KV blocks
        feature_map='softmax'
    )

    output = attn(q, k, v)  # q, k, v: [B, H, L, D]
"""

from .core import SparseLinearAttention

__all__ = ["SparseLinearAttention"]
