"""Benchmark: Sparse Linear Attention vs SDPA"""

import torch
import torch.nn.functional as F
import time
from sparse_linear_attention import SparseLinearAttention


def benchmark_fn(fn, warmup=10, repeat=100):
    """Benchmark a function with warmup and timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timing
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / repeat * 1000  # ms


def main():
    print("=" * 70)
    print("Benchmark: Sparse Linear Attention vs SDPA")
    print("=" * 70)

    configs = [
        # (B, H, L, D)
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (1, 8, 4096, 64),
        (2, 8, 1024, 64),
        (2, 8, 2048, 64),
        (1, 16, 2048, 64),
        (1, 8, 4096, 128),
    ]

    topk = 0.5
    dtype = torch.bfloat16

    print(f"\nSettings: topk={topk}, dtype={dtype}")
    print("-" * 70)
    print(f"{'Config':<25} {'SDPA (ms)':<15} {'SLA (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    for B, H, L, D in configs:
        # Create inputs
        q = torch.randn(B, H, L, D, device='cuda', dtype=dtype)
        k = torch.randn(B, H, L, D, device='cuda', dtype=dtype)
        v = torch.randn(B, H, L, D, device='cuda', dtype=dtype)

        # SDPA (need to handle shape)
        def run_sdpa():
            # SDPA expects (B, H, L, D) and does softmax attention
            with torch.no_grad():
                out = F.scaled_dot_product_attention(q, k, v)
            return out

        # Sparse Linear Attention
        sla = SparseLinearAttention(
            head_dim=D,
            topk=topk,
            feature_map='softmax',
            BLKQ=64,
            BLKK=64,
        )

        def run_sla():
            with torch.no_grad():
                out = sla(q, k, v)
            return out

        # Warmup and check outputs
        try:
            out_sdpa = run_sdpa()
            out_sla = run_sla()
        except Exception as e:
            print(f"B={B}, H={H}, L={L}, D={D}: Error - {e}")
            continue

        # Benchmark
        time_sdpa = benchmark_fn(run_sdpa)
        time_sla = benchmark_fn(run_sla)
        speedup = time_sdpa / time_sla

        config_str = f"B={B}, H={H}, L={L}, D={D}"
        print(f"{config_str:<25} {time_sdpa:<15.3f} {time_sla:<15.3f} {speedup:<10.2f}x")

    print("-" * 70)

    # Memory benchmark
    print("\n" + "=" * 70)
    print("Memory Usage Comparison (L=4096)")
    print("=" * 70)

    B, H, L, D = 1, 8, 4096, 64
    q = torch.randn(B, H, L, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, L, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, L, D, device='cuda', dtype=dtype)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # SDPA memory
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out_sdpa = F.scaled_dot_product_attention(q, k, v)
    torch.cuda.synchronize()
    mem_sdpa = torch.cuda.max_memory_allocated() / 1024**2

    del out_sdpa
    torch.cuda.empty_cache()

    # SLA memory
    sla = SparseLinearAttention(head_dim=D, topk=topk, feature_map='softmax')
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        out_sla = sla(q, k, v)
    torch.cuda.synchronize()
    mem_sla = torch.cuda.max_memory_allocated() / 1024**2

    print(f"SDPA peak memory: {mem_sdpa:.2f} MB")
    print(f"SLA peak memory:  {mem_sla:.2f} MB")
    print(f"Memory ratio:     {mem_sla/mem_sdpa:.2f}x")


if __name__ == '__main__':
    main()
