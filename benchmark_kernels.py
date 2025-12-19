"""Benchmark individual kernels."""

import torch
import torch.nn.functional as F
import time
from sparse_linear_attention.utils import get_block_map
from sparse_linear_attention.kernel import (
    _attn_fwd_preprocess,
    _attn_fwd,
    _attn_bwd_preprocess,
    _attn_bwd_dcq,
    _attn_bwd_dhdz,
    _attn_bwd_dckdv,
)


def benchmark_fn(fn, warmup=10, repeat=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / repeat * 1000


def main():
    print("=" * 70)
    print("Benchmark Individual Kernels")
    print("=" * 70)

    B, H, L, D = 1, 8, 4096, 64
    CD = D
    topk_ratio = 0.5
    BLOCK_M, BLOCK_N = 64, 64
    dtype = torch.bfloat16

    M_BLOCKS = L // BLOCK_M
    N_BLOCKS = L // BLOCK_N

    print(f"\nConfig: B={B}, H={H}, L={L}, D={D}")
    print(f"M_BLOCKS={M_BLOCKS}, N_BLOCKS={N_BLOCKS}, topk_ratio={topk_ratio}")
    print("-" * 70)

    # Create inputs
    q = torch.randn(B, H, L, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, L, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, L, D, device='cuda', dtype=dtype)

    c_q = F.softmax(q, dim=-1).to(dtype).contiguous()
    c_k = F.softmax(k, dim=-1).to(dtype).contiguous()

    # Reshape for kernel
    c_q_flat = c_q.view(B * H, L, CD).contiguous()
    c_k_flat = c_k.view(B * H, L, CD).contiguous()
    v_flat = v.view(B * H, L, D).contiguous()

    sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio, BLOCK_M, BLOCK_N)
    lut_flat = lut.view(B * H, M_BLOCKS, real_topk).contiguous()
    sparse_map_flat = sparse_map.view(B * H, M_BLOCKS, N_BLOCKS).contiguous()

    # Precompute buffers
    h = torch.zeros(B * H, N_BLOCKS, CD, D, device='cuda', dtype=dtype)
    z = torch.zeros(B * H, N_BLOCKS, CD, device='cuda', dtype=dtype)
    denom = torch.zeros(B * H, L, device='cuda', dtype=dtype)
    o = torch.zeros(B * H, L, D, device='cuda', dtype=dtype)

    # 1. _attn_fwd_preprocess
    def run_preprocess():
        _attn_fwd_preprocess[(N_BLOCKS, B * H)](
            c_k_flat, v_flat, h, z,
            L, N_BLOCKS, D, CD, BLOCK_N,
            num_warps=4,
        )

    time_preprocess = benchmark_fn(run_preprocess)
    print(f"_attn_fwd_preprocess:     {time_preprocess:.3f} ms")

    # Run once to fill h, z
    run_preprocess()

    # 2. _attn_fwd
    def run_fwd():
        _attn_fwd[(M_BLOCKS, B * H)](
            c_q_flat, h, z, lut_flat, denom, o,
            real_topk, L, M_BLOCKS, N_BLOCKS, D, CD, BLOCK_M, BLOCK_N,
            num_warps=4,
        )

    time_fwd = benchmark_fn(run_fwd)
    print(f"_attn_fwd:                {time_fwd:.3f} ms")

    # Total forward
    def run_total_fwd():
        run_preprocess()
        run_fwd()

    time_total_fwd = benchmark_fn(run_total_fwd)
    print(f"Total forward:            {time_total_fwd:.3f} ms")

    # Backward kernels
    do = torch.randn_like(o)
    delta = torch.zeros(B * H, L, device='cuda', dtype=dtype)
    h_acc = torch.zeros(B * H, M_BLOCKS, CD, D, device='cuda', dtype=dtype)
    z_acc = torch.zeros(B * H, M_BLOCKS, CD, device='cuda', dtype=dtype)

    # 3. _attn_bwd_preprocess
    def run_bwd_preprocess():
        _attn_bwd_preprocess[(M_BLOCKS, B * H)](
            c_q_flat, h, z, lut_flat, do, denom,
            delta, h_acc, z_acc,
            real_topk, L, M_BLOCKS, N_BLOCKS, D, CD, BLOCK_M, BLOCK_N,
            num_warps=4,
        )

    time_bwd_preprocess = benchmark_fn(run_bwd_preprocess)
    print(f"_attn_bwd_preprocess:     {time_bwd_preprocess:.3f} ms")

    run_bwd_preprocess()

    # 4. _attn_bwd_dcq
    dc_q = torch.zeros_like(c_q_flat)
    def run_bwd_dcq():
        _attn_bwd_dcq[(M_BLOCKS, B * H)](
            h_acc, z_acc, do, denom, delta, dc_q,
            L, M_BLOCKS, D, CD, BLOCK_M,
            num_warps=4,
        )

    time_bwd_dcq = benchmark_fn(run_bwd_dcq)
    print(f"_attn_bwd_dcq:            {time_bwd_dcq:.3f} ms")

    # 5. _attn_bwd_dhdz
    dh = torch.zeros(B * H, N_BLOCKS, CD, D, device='cuda', dtype=dtype)
    dz = torch.zeros(B * H, N_BLOCKS, CD, device='cuda', dtype=dtype)
    def run_bwd_dhdz():
        _attn_bwd_dhdz[(N_BLOCKS, B * H)](
            c_q_flat, do, delta, denom, sparse_map_flat,
            dh, dz,
            L, M_BLOCKS, N_BLOCKS, D, CD, BLOCK_M, BLOCK_N,
            BLOCK_SLICE_FACTOR=BLOCK_M // 64,
            num_warps=4,
        )

    time_bwd_dhdz = benchmark_fn(run_bwd_dhdz)
    print(f"_attn_bwd_dhdz:           {time_bwd_dhdz:.3f} ms")

    run_bwd_dhdz()

    # 6. _attn_bwd_dckdv
    dc_k = torch.zeros_like(c_k_flat)
    dv = torch.zeros_like(v_flat)
    def run_bwd_dckdv():
        _attn_bwd_dckdv[(N_BLOCKS, B * H)](
            c_k_flat, v_flat, dh, dz, dc_k, dv,
            L, N_BLOCKS, D, CD, BLOCK_N,
            num_warps=4,
        )

    time_bwd_dckdv = benchmark_fn(run_bwd_dckdv)
    print(f"_attn_bwd_dckdv:          {time_bwd_dckdv:.3f} ms")

    # Total backward
    def run_total_bwd():
        run_bwd_preprocess()
        run_bwd_dcq()
        run_bwd_dhdz()
        run_bwd_dckdv()

    time_total_bwd = benchmark_fn(run_total_bwd)
    print(f"Total backward:           {time_total_bwd:.3f} ms")

    print("-" * 70)
    print(f"\nSummary:")
    print(f"  Forward:  {time_total_fwd:.3f} ms")
    print(f"  Backward: {time_total_bwd:.3f} ms")
    print(f"  Total:    {time_total_fwd + time_total_bwd:.3f} ms")

    # Compare with SDPA
    def run_sdpa():
        with torch.no_grad():
            return F.scaled_dot_product_attention(q, k, v)

    time_sdpa = benchmark_fn(run_sdpa)
    print(f"\n  SDPA:     {time_sdpa:.3f} ms")
    print(f"  SLA/SDPA: {(time_total_fwd + time_total_bwd) / time_sdpa:.1f}x")


if __name__ == '__main__':
    main()
