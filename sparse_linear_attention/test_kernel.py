"""
Test script to verify the correctness of the Triton kernel against
the pure PyTorch reference implementation.

Usage:
    python -m sparse_linear_attention.test_kernel
"""

import torch
import torch.nn.functional as F

from .kernel import _sparse_linear_attention
from .reference_pytorch import sparse_linear_attention_reference
from .utils import get_block_map


def test_forward_correctness(
    B=2, H=4, L=256, D=64, CD=64, topk_ratio=0.5, BLOCK_M=64, BLOCK_N=64,
    dtype=torch.bfloat16, device='cuda', rtol=1e-2, atol=1e-2
):
    """
    Test forward pass correctness by comparing Triton kernel with PyTorch reference.
    """
    print(f"\n{'='*60}")
    print(f"Testing Forward Pass Correctness")
    print(f"B={B}, H={H}, L={L}, D={D}, CD={CD}")
    print(f"topk_ratio={topk_ratio}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print(f"dtype={dtype}")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Generate random inputs
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)

    # Apply feature map (softmax)
    c_q = F.softmax(q, dim=-1).contiguous()
    c_k = F.softmax(k, dim=-1).contiguous()
    v = v.contiguous()

    # Get block map and LUT
    sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=BLOCK_M, BLKK=BLOCK_N)
    sparse_map = sparse_map.contiguous()
    lut = lut.contiguous()

    print(f"real_topk: {real_topk}")
    print(f"sparse_map shape: {sparse_map.shape}")
    print(f"lut shape: {lut.shape}")

    # Run Triton kernel
    o_triton = _sparse_linear_attention.apply(c_q, c_k, v, sparse_map, lut, real_topk, BLOCK_M, BLOCK_N)

    # Run PyTorch reference (use float32 for reference)
    o_ref = sparse_linear_attention_reference(
        c_q.float(), c_k.float(), v.float(), sparse_map, lut, real_topk, BLOCK_M, BLOCK_N
    )

    # Compare results
    diff = (o_triton.float() - o_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\nForward Pass Results:")
    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")
    print(f"  Output shape: {o_triton.shape}")
    print(f"  Output dtype: {o_triton.dtype}")

    # Check if within tolerance
    is_close = torch.allclose(o_triton.float(), o_ref.float(), rtol=rtol, atol=atol)
    if is_close:
        print(f"  ✓ PASSED (within rtol={rtol}, atol={atol})")
    else:
        print(f"  ✗ FAILED (exceeds rtol={rtol}, atol={atol})")

        # Print some debug info
        print(f"\nDebug info:")
        print(f"  o_triton stats: min={o_triton.min().item():.4f}, max={o_triton.max().item():.4f}, mean={o_triton.mean().item():.4f}")
        print(f"  o_ref stats: min={o_ref.min().item():.4f}, max={o_ref.max().item():.4f}, mean={o_ref.mean().item():.4f}")

    return is_close


def test_backward_correctness(
    B=2, H=4, L=256, D=64, CD=64, topk_ratio=0.5, BLOCK_M=64, BLOCK_N=64,
    dtype=torch.bfloat16, device='cuda', rtol=1e-1, atol=1e-1
):
    """
    Test backward pass correctness by comparing gradients from Triton kernel with PyTorch reference.
    """
    print(f"\n{'='*60}")
    print(f"Testing Backward Pass Correctness")
    print(f"B={B}, H={H}, L={L}, D={D}, CD={CD}")
    print(f"topk_ratio={topk_ratio}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print(f"dtype={dtype}")
    print(f"{'='*60}")

    torch.manual_seed(42)

    # Generate random inputs
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)

    # Apply feature map (softmax)
    c_q_base = F.softmax(q, dim=-1)
    c_k_base = F.softmax(k, dim=-1)

    # Get block map and LUT (use same for both)
    sparse_map, lut, real_topk = get_block_map(q, k, topk_ratio=topk_ratio, BLKQ=BLOCK_M, BLKK=BLOCK_N)
    sparse_map = sparse_map.contiguous()
    lut = lut.contiguous()

    print(f"real_topk: {real_topk}")

    # Create inputs with gradients for Triton
    c_q_triton = c_q_base.clone().contiguous().requires_grad_(True)
    c_k_triton = c_k_base.clone().contiguous().requires_grad_(True)
    v_triton = v.clone().contiguous().requires_grad_(True)

    # Create inputs with gradients for reference (use float32 for numerical stability)
    c_q_ref = c_q_base.float().clone().contiguous().requires_grad_(True)
    c_k_ref = c_k_base.float().clone().contiguous().requires_grad_(True)
    v_ref = v.float().clone().contiguous().requires_grad_(True)

    # Forward + backward with Triton
    o_triton = _sparse_linear_attention.apply(c_q_triton, c_k_triton, v_triton, sparse_map, lut, real_topk, BLOCK_M, BLOCK_N)
    grad_output = torch.randn_like(o_triton)
    o_triton.backward(grad_output)

    # Forward + backward with reference
    o_ref = sparse_linear_attention_reference(c_q_ref, c_k_ref, v_ref, sparse_map, lut, real_topk, BLOCK_M, BLOCK_N)
    o_ref.backward(grad_output.float())

    # Compare gradients
    results = []

    for name, grad_triton, grad_ref in [
        ('dc_q', c_q_triton.grad, c_q_ref.grad),
        ('dc_k', c_k_triton.grad, c_k_ref.grad),
        ('dv', v_triton.grad, v_ref.grad),
    ]:
        if grad_triton is None or grad_ref is None:
            print(f"\n{name}: Missing gradient (triton={grad_triton is not None}, ref={grad_ref is not None})")
            results.append(False)
            continue

        diff = (grad_triton.float() - grad_ref.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        is_close = torch.allclose(grad_triton.float(), grad_ref.float(), rtol=rtol, atol=atol)
        results.append(is_close)

        print(f"\n{name}:")
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        print(f"  Shape: {grad_triton.shape}")

        if is_close:
            print(f"  ✓ PASSED (within rtol={rtol}, atol={atol})")
        else:
            print(f"  ✗ FAILED (exceeds rtol={rtol}, atol={atol})")
            print(f"  triton grad stats: min={grad_triton.min().item():.4f}, max={grad_triton.max().item():.4f}, mean={grad_triton.mean().item():.4f}")
            print(f"  ref grad stats: min={grad_ref.min().item():.4f}, max={grad_ref.max().item():.4f}, mean={grad_ref.mean().item():.4f}")

    return all(results)


def test_different_configs():
    """
    Test with various configurations.
    """
    print(f"\n{'='*60}")
    print(f"Testing Different Configurations")
    print(f"{'='*60}")

    configs = [
        # (B, H, L, D, CD, topk_ratio, BLOCK_M, BLOCK_N)
        (1, 1, 128, 64, 64, 0.5, 64, 64),
        (2, 4, 256, 64, 64, 0.25, 64, 64),
        (2, 4, 256, 64, 64, 0.75, 64, 64),
        (1, 2, 512, 64, 64, 0.5, 64, 64),
        (2, 4, 256, 128, 64, 0.5, 64, 64),  # Different D
        (1, 1, 256, 64, 64, 1.0, 64, 64),   # Full attention (topk=1.0)
        (2, 4, 256, 64, 64, 0.5, 128, 64),  # BLOCK_M=128
    ]

    all_passed = True
    for config in configs:
        B, H, L, D, CD, topk_ratio, BLOCK_M, BLOCK_N = config
        print(f"\nConfig: B={B}, H={H}, L={L}, D={D}, topk={topk_ratio}, BLOCK_M={BLOCK_M}")

        try:
            fwd_passed = test_forward_correctness(
                B=B, H=H, L=L, D=D, CD=CD,
                topk_ratio=topk_ratio, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                rtol=5e-2, atol=5e-2
            )
            all_passed = all_passed and fwd_passed
        except Exception as e:
            print(f"  ✗ FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_gradient_check():
    """
    Numerical gradient check using torch.autograd.gradcheck.
    """
    print(f"\n{'='*60}")
    print(f"Numerical Gradient Check (Reference Implementation)")
    print(f"{'='*60}")

    torch.manual_seed(42)

    B, H, L, D = 1, 1, 64, 32
    BLOCK_M, BLOCK_N = 64, 64
    topk_ratio = 0.5

    device = 'cuda'
    dtype = torch.float64  # Need float64 for gradcheck

    # Generate random inputs
    q = torch.randn(B, H, L, D, device=device, dtype=dtype)
    k = torch.randn(B, H, L, D, device=device, dtype=dtype)
    v = torch.randn(B, H, L, D, device=device, dtype=dtype)

    # Apply feature map (softmax)
    c_q = F.softmax(q, dim=-1).contiguous().requires_grad_(True)
    c_k = F.softmax(k, dim=-1).contiguous().requires_grad_(True)
    v = v.contiguous().requires_grad_(True)

    # Get block map and LUT
    sparse_map, lut, real_topk = get_block_map(
        q.float(), k.float(), topk_ratio=topk_ratio, BLKQ=BLOCK_M, BLKK=BLOCK_N
    )
    sparse_map = sparse_map.contiguous()
    lut = lut.contiguous()

    print(f"Testing numerical gradients for reference implementation...")
    print(f"This may take a while...")

    try:
        passed = torch.autograd.gradcheck(
            lambda cq, ck, vv: sparse_linear_attention_reference(
                cq, ck, vv, sparse_map, lut, real_topk, BLOCK_M, BLOCK_N
            ),
            (c_q, c_k, v),
            eps=1e-6,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=False
        )
        if passed:
            print("  ✓ Numerical gradient check PASSED")
        else:
            print("  ✗ Numerical gradient check FAILED")
        return passed
    except Exception as e:
        print(f"  ✗ Gradient check failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("SPARSE LINEAR ATTENTION KERNEL CORRECTNESS TESTS")
    print("="*60)

    results = {}

    # Basic forward test
    print("\n[1/4] Basic Forward Test")
    results['forward'] = test_forward_correctness()

    # Basic backward test
    print("\n[2/4] Basic Backward Test")
    results['backward'] = test_backward_correctness()

    # Different configurations
    print("\n[3/4] Different Configurations Test")
    results['configs'] = test_different_configs()

    # Gradient check (optional, slow)
    print("\n[4/4] Numerical Gradient Check")
    results['gradcheck'] = test_gradient_check()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED!")
    print("="*60)

    return all_passed


if __name__ == '__main__':
    run_all_tests()
