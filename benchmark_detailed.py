"""Detailed benchmark to find bottleneck."""

import torch
import torch.nn.functional as F
import time
from sparse_linear_attention import SparseLinearAttention
from sparse_linear_attention.utils import get_block_map
from sparse_linear_attention.kernel import _sparse_linear_attention


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
    print("Detailed Benchmark: Breaking down SLA overhead")
    print("=" * 70)

    B, H, L, D = 1, 8, 4096, 64
    topk = 0.5
    dtype = torch.bfloat16
    BLKQ, BLKK = 64, 64

    print(f"\nConfig: B={B}, H={H}, L={L}, D={D}, topk={topk}")
    print("-" * 70)

    q = torch.randn(B, H, L, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, L, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, L, D, device='cuda', dtype=dtype)

    # 1. SDPA baseline
    def run_sdpa():
        with torch.no_grad():
            return F.scaled_dot_product_attention(q, k, v)

    time_sdpa = benchmark_fn(run_sdpa)
    print(f"SDPA:                     {time_sdpa:.3f} ms")

    # 2. get_block_map only
    def run_block_map():
        with torch.no_grad():
            return get_block_map(q, k, topk, BLKQ, BLKK)

    time_block_map = benchmark_fn(run_block_map)
    print(f"get_block_map:            {time_block_map:.3f} ms")

    # 3. Feature map only
    def run_feature_map():
        with torch.no_grad():
            c_q = F.softmax(q, dim=-1)
            c_k = F.softmax(k, dim=-1)
            return c_q, c_k

    time_feature_map = benchmark_fn(run_feature_map)
    print(f"Feature map (softmax):    {time_feature_map:.3f} ms")

    # 4. Kernel only (pre-computed block map)
    sparse_map, lut, real_topk = get_block_map(q, k, topk, BLKQ, BLKK)
    c_q = F.softmax(q, dim=-1).to(dtype).contiguous()
    c_k = F.softmax(k, dim=-1).to(dtype).contiguous()
    v_cont = v.contiguous()
    sparse_map = sparse_map.contiguous()
    lut = lut.contiguous()

    def run_kernel_only():
        with torch.no_grad():
            return _sparse_linear_attention.apply(c_q, c_k, v_cont, sparse_map, lut, real_topk, BLKQ, BLKK)

    time_kernel = benchmark_fn(run_kernel_only)
    print(f"Kernel only:              {time_kernel:.3f} ms")

    # 5. Full SLA
    sla = SparseLinearAttention(head_dim=D, topk=topk, feature_map='softmax', BLKQ=BLKQ, BLKK=BLKK)
    def run_sla():
        with torch.no_grad():
            return sla(q, k, v)

    time_sla = benchmark_fn(run_sla)
    print(f"Full SLA:                 {time_sla:.3f} ms")

    print("-" * 70)
    print(f"\nBreakdown:")
    print(f"  get_block_map:   {time_block_map:.3f} ms ({time_block_map/time_sla*100:.1f}%)")
    print(f"  feature_map:     {time_feature_map:.3f} ms ({time_feature_map/time_sla*100:.1f}%)")
    print(f"  kernel:          {time_kernel:.3f} ms ({time_kernel/time_sla*100:.1f}%)")
    print(f"  overhead:        {time_sla - time_block_map - time_feature_map - time_kernel:.3f} ms")

    print(f"\nSLA / SDPA = {time_sla / time_sdpa:.1f}x slower")
    print(f"Kernel only / SDPA = {time_kernel / time_sdpa:.1f}x slower")


if __name__ == '__main__':
    main()
