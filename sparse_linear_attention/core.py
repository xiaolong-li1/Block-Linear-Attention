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
- Mask=1: Sparse Linear Attention (load precomputed h_j, z_j and accumulate)
- Mask=0: Skip (no memory load, no computation)
- Single output (no O_s + O_l combination)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import _sparse_linear_attention
from .utils import get_block_map


class SparseLinearAttention(nn.Module):
    def __init__(self, topk, feature_map='softmax', BLKQ=64, BLKK=64, use_bf16=True):
        """
        Sparse Linear Attention module.

        Args:
            topk: Ratio of KV blocks selected for sparse attention (0.0 to 1.0).
            feature_map: Feature map for linear attention, one of ['elu', 'relu', 'softmax'].
            BLKQ: Block size for queries.
            BLKK: Block size for keys/values.
            use_bf16: Whether to use bfloat16 (default) or float16 for computation.
        """
        super().__init__()
        self.dtype = torch.bfloat16 if use_bf16 else torch.float16
        self.topk = topk
        self.BLKQ = BLKQ
        self.BLKK = BLKK

        # Feature map selection
        if feature_map == 'elu':
            def elu_feature_map(x):
                return F.elu(x) + 1
            self.feature_map_q = elu_feature_map
            self.feature_map_k = elu_feature_map
        elif feature_map == 'relu':
            self.feature_map_q = nn.ReLU()
            self.feature_map_k = nn.ReLU()
        elif feature_map == 'softmax':
            def softmax_feature_map(x):
                return F.softmax(x, dim=-1)
            self.feature_map_q = softmax_feature_map
            self.feature_map_k = softmax_feature_map
        else:
            raise NotImplementedError(f'Not supported feature map {feature_map}.')

    def forward(self, q, k, v, return_sparsity=False):
        """
        Forward pass of Sparse Linear Attention.

        Args:
            q: Queries of shape (B, H, L, D).
            k: Keys of shape (B, H, L, D).
            v: Values of shape (B, H, L, D).
            return_sparsity: Whether to return the actual sparsity ratio.

        Returns:
            o: Output tensor of shape (B, H, L, D).
            sparsity (optional): Actual sparsity ratio if return_sparsity=True.
        """
        dtype = q.dtype

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        current_topk = float(self.topk)
        B, H, L, _ = q.shape
        num_q_blocks = (L + self.BLKQ - 1) // self.BLKQ
        num_k_blocks = (L + self.BLKK - 1) // self.BLKK
        est_topk = min(num_k_blocks, int(max(current_topk, 0.0) * num_k_blocks))

        if est_topk <= 0:
            # No blocks selected - return zeros
            device = q.device
            sparse_map = torch.zeros((B, H, num_q_blocks, num_k_blocks), dtype=torch.int8, device=device)
            lut = torch.empty((B, H, num_q_blocks, 0), dtype=torch.int64, device=device)
            real_topk = 0
        else:
            sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=current_topk, BLKQ=self.BLKQ, BLKK=self.BLKK)

        sparse_map = sparse_map.contiguous()
        lut = lut.contiguous()

        # Convert to computation dtype
        q = q.to(self.dtype)
        k = k.to(self.dtype)
        v = v.to(self.dtype)

        # Apply feature maps for linear attention
        c_q = self.feature_map_q(q).contiguous().to(self.dtype)
        c_k = self.feature_map_k(k).contiguous().to(self.dtype)

        # Call the sparse linear attention kernel
        o = _sparse_linear_attention.apply(c_q, c_k, v, sparse_map, lut, real_topk, self.BLKQ, self.BLKK)

        # Convert back to original dtype
        o = o.to(dtype)

        if return_sparsity:
            return o, real_topk / sparse_map.shape[-1]
        else:
            return o
