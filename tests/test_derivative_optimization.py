"""Test and compare old vs new derivative computation methods.

This script validates that the vectorized derivative computation produces
the same results as the current implementation while being faster.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np

from src.model.schrodinger_model import SchrodingerNet


def compute_derivatives_old(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple:
    """Current implementation: 6 separate autograd.grad calls."""
    # Split into real and imaginary parts
    u = h.real
    v = h.imag

    # Create grad_outputs once
    ones = torch.ones_like(u)

    # First derivatives w.r.t time (2 calls)
    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )[0]

    v_t = torch.autograd.grad(
        v, t,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )[0]

    # First derivatives w.r.t space (2 calls)
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )[0]

    v_x = torch.autograd.grad(
        v, x,
        grad_outputs=ones,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Second derivatives w.r.t space (2 calls)
    ones_x = torch.ones_like(u_x)
    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=ones_x,
        create_graph=True,
        retain_graph=True,
    )[0]

    v_xx = torch.autograd.grad(
        v_x, x,
        grad_outputs=ones_x,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Pack as complex
    h_t = torch.complex(u_t, v_t).squeeze()
    h_x = torch.complex(u_x, v_x).squeeze()
    h_xx = torch.complex(u_xx, v_xx).squeeze()

    return h_t, h_x, h_xx


def compute_derivatives_new(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple:
    """Optimized implementation: vectorized autograd with fewer calls.
    
    Key optimization: compute u and v derivatives w.r.t. both x and t together,
    reducing from 6 separate autograd calls to 4 calls total (33% reduction).
    
    Call 1: u w.r.t. [x, t] -> (u_x, u_t)
    Call 2: v w.r.t. [x, t] -> (v_x, v_t)
    Call 3: u_x w.r.t. x -> u_xx
    Call 4: v_x w.r.t. x -> v_xx
    """
    # Split into real and imaginary parts
    u = h.real
    v = h.imag

    # Create grad_outputs once
    ones_u = torch.ones_like(u)
    ones_v = torch.ones_like(v)
    
    # Call 1: Compute u derivatives w.r.t. BOTH x and t in one call
    u_grads = torch.autograd.grad(
        outputs=u,
        inputs=[x, t],
        grad_outputs=ones_u,
        create_graph=True,
        retain_graph=True,
    )
    u_x = u_grads[0]
    u_t = u_grads[1]
    
    # Call 2: Compute v derivatives w.r.t. BOTH x and t in one call
    v_grads = torch.autograd.grad(
        outputs=v,
        inputs=[x, t],
        grad_outputs=ones_v,
        create_graph=True,
        retain_graph=True,
    )
    v_x = v_grads[0]
    v_t = v_grads[1]

    # Second derivatives w.r.t space
    ones_ux = torch.ones_like(u_x)
    ones_vx = torch.ones_like(v_x)
    
    # Call 3: u_xx
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=ones_ux,
        create_graph=True,
        retain_graph=True,
    )[0]
    
    # Call 4: v_xx  (final call, no need to retain)
    v_xx = torch.autograd.grad(
        outputs=v_x,
        inputs=x,
        grad_outputs=ones_vx,
        create_graph=True,
        retain_graph=False,  # Final use, free the graph
    )[0]

    # Pack as complex
    h_t = torch.complex(u_t, v_t).squeeze()
    h_x = torch.complex(u_x, v_x).squeeze()
    h_xx = torch.complex(u_xx, v_xx).squeeze()

    return h_t, h_x, h_xx


def test_correctness(batch_size=512, device="cuda", tolerance=1e-6):
    """Test that all three methods produce identical results."""
    print(f"\n{'='*70}")
    print(f"Correctness Test (batch_size={batch_size}, device={device})")
    print(f"{'='*70}")
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Create model
    model = SchrodingerNet(hidden_layers=5, hidden_neurons=50, activation="tanh").to(device)
    model.eval()
    
    # Create sample input
    x = torch.randn(batch_size, 1, device=device, requires_grad=True)
    t = torch.randn(batch_size, 1, device=device, requires_grad=True)
    
    # Compute with OLD method (ground truth)
    print("\nComputing with OLD method (6 calls - baseline)...")
    x_old = x.detach().clone().requires_grad_(True)
    t_old = t.detach().clone().requires_grad_(True)
    h_old = model.predict_h(x_old, t_old)
    h_t_old, h_x_old, h_xx_old = compute_derivatives_old(h_old, x_old, t_old)
    
    # Compute with NEW method
    print("Computing with NEW method (4 calls)...")
    x_new = x.detach().clone().requires_grad_(True)
    t_new = t.detach().clone().requires_grad_(True)
    h_new = model.predict_h(x_new, t_new)
    h_t_new, h_x_new, h_xx_new = compute_derivatives_new(h_new, x_new, t_new)
    
    # Compare results
    print("\nComparing NEW vs OLD:")
    diff_t_new = torch.abs(h_t_old - h_t_new).max().item()
    diff_x_new = torch.abs(h_x_old - h_x_new).max().item()
    diff_xx_new = torch.abs(h_xx_old - h_xx_new).max().item()
    
    print(f"  h_t  max diff: {diff_t_new:.2e}")
    print(f"  h_x  max diff: {diff_x_new:.2e}")
    print(f"  h_xx max diff: {diff_xx_new:.2e}")
    
    # Check tolerance
    passed_new = all([diff_t_new < tolerance, diff_x_new < tolerance, diff_xx_new < tolerance])
    
    if passed_new:
        print(f"\n✓ PASSED: NEW method matches OLD within tolerance ({tolerance:.2e})")
        return True
    else:
        print(f"\n✗ FAILED: NEW method exceeds tolerance ({tolerance:.2e})")
        return False


def benchmark_performance(batch_size=512, device="cuda", n_iterations=100):
    """Benchmark all three methods with ISOLATED derivative computation.
    
    Key improvement: Precompute h once and benchmark ONLY the derivative
    computation to isolate the true speedup from model overhead.
    """
    print(f"\n{'='*70}")
    print(f"Performance Benchmark (batch_size={batch_size}, n_iterations={n_iterations})")
    print(f"  ISOLATED derivatives only (no model forward pass)")
    print(f"{'='*70}")
    
    # Set seed
    torch.manual_seed(42)
    
    # Create model
    model = SchrodingerNet(hidden_layers=5, hidden_neurons=50, activation="tanh").to(device)
    model.eval()
    
    # Precompute h ONCE (this is the key optimization)
    x_base = torch.randn(batch_size, 1, device=device)
    t_base = torch.randn(batch_size, 1, device=device)
    with torch.no_grad():
        h_base = model.predict_h(x_base, t_base)
    
    # Warmup
    print("\nWarming up GPU...")
    for _ in range(10):
        x_w = x_base.detach().clone().requires_grad_(True)
        t_w = t_base.detach().clone().requires_grad_(True)
        h_w = h_base.detach().clone()
        h_w = torch.complex(h_w.real.requires_grad_(True), h_w.imag.requires_grad_(True))
        # Actually, for complex autograd, we need to work with x,t gradients
        # Let me fix this
        with torch.enable_grad():
            x_w = x_base.detach().clone().requires_grad_(True)
            t_w = t_base.detach().clone().requires_grad_(True)
            h_w = model.predict_h(x_w, t_w)
            _, _, _ = compute_derivatives_old(h_w, x_w, t_w)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark OLD method (6 calls)
    print(f"\nBenchmarking OLD method (6 calls)...")
    times_old = []
    for i in range(n_iterations):
        x_test = x_base.detach().clone().requires_grad_(True)
        t_test = t_base.detach().clone().requires_grad_(True)
        h_test = model.predict_h(x_test, t_test)  # Need to recompute for gradient graph
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        _, _, _ = compute_derivatives_old(h_test, x_test, t_test)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        times_old.append(time.time() - start)
    
    time_old = np.mean(times_old)
    std_old = np.std(times_old)
    
    # Benchmark NEW method (4 calls)
    print(f"Benchmarking NEW method (4 calls)...")
    times_new = []
    for i in range(n_iterations):
        x_test = x_base.detach().clone().requires_grad_(True)
        t_test = t_base.detach().clone().requires_grad_(True)
        h_test = model.predict_h(x_test, t_test)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        _, _, _ = compute_derivatives_new(h_test, x_test, t_test)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        times_new.append(time.time() - start)
    
    time_new = np.mean(times_new)
    std_new = np.std(times_new)
    
    # Calculate speedups
    speedup_new = time_old / time_new
    
    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"  OLD (6 calls):     {time_old*1000:.3f} ± {std_old*1000:.3f} ms  [baseline]")
    print(f"  NEW (4 calls):     {time_new*1000:.3f} ± {std_new*1000:.3f} ms  [{speedup_new:.2f}x]")
    print(f"{'='*70}")
    
    # Determine winner
    if speedup_new >= 1.5:
        print(f"\n🏆 WINNER: NEW method")
        print(f"  Speedup: {speedup_new:.2f}x")
        winner = "NEW"
        best_speedup = speedup_new
    else:
        print(f"\n⚠ No clear winner. Best speedup: {speedup_new:.2f}x")
        winner = "NONE"
        best_speedup = speedup_new
    
    return winner, best_speedup, time_old, time_new


def main():
    """Run comprehensive derivative optimization tests."""
    print("\n" + "="*70)
    print("DERIVATIVE OPTIMIZATION: Complete Test Suite")
    print("Testing OLD (6 calls) vs NEW (4 calls)")
    print("="*70)
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Test 1: Correctness
    passed_correctness = test_correctness(batch_size=512, device=device, tolerance=1e-6)
    
    if not passed_correctness:
        print("\n✗ Correctness test FAILED. Cannot proceed.")
        return False
    
    # Test 2: Performance at different batch sizes
    print(f"\n{'='*70}")
    print("MULTI-BATCH PERFORMANCE TESTING")
    print("Testing larger batches to see scaling behavior...")
    print(f"{'='*70}")
    
    results = []
    batch_sizes = [512, 1024, 2048, 4096, 8192, 16384]
    
    for bs in batch_sizes:
        print(f"\n{'─'*70}")
        print(f"Batch size: {bs}")
        print(f"{'─'*70}")
        winner, best_speedup, time_old, time_new = benchmark_performance(
            batch_size=bs, device=device, n_iterations=50
        )
        results.append({
            'batch_size': bs,
            'winner': winner,
            'speedup': best_speedup,
            'time_old': time_old,
            'time_new': time_new
        })
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print(f"\n✓ Correctness: PASSED")
    print(f"\nPerformance across batch sizes:")
    print(f"{'─'*70}")
    print(f"{'Batch':>8} | {'OLD (ms)':>10} | {'NEW (ms)':>10} | {'Winner':>8} | {'Speedup':>8}")
    print(f"{'─'*70}")
    for r in results:
        print(f"{r['batch_size']:>8} | {r['time_old']*1000:>10.2f} | {r['time_new']*1000:>10.2f} | "
              f"{r['winner']:>8} | {r['speedup']:>7.2f}x")
    print(f"{'─'*70}")
    
    # Determine overall recommendation with detailed analysis
    new_wins = sum(1 for r in results if r['winner'] == 'NEW')
    
    # Calculate average speedups
    avg_speedup_new = np.mean([r['time_old'] / r['time_new'] for r in results])
    
    # Calculate speedups for typical training batch sizes (512-2048)
    training_range = [r for r in results if 512 <= r['batch_size'] <= 2048]
    if training_range:
        train_speedup_new = np.mean([r['time_old'] / r['time_new'] for r in training_range])
    else:
        train_speedup_new = avg_speedup_new
    
    print(f"\nOverall Statistics:")
    print(f"  NEW method:     avg {avg_speedup_new:.2f}x speedup (all batches), {train_speedup_new:.2f}x (training range)")
    print(f"  Wins: NEW={new_wins}")
    
    # Decision logic prioritizing training batch sizes
    print(f"\n{'='*70}")
    print("DECISION CRITERIA")
    print(f"{'='*70}")
    print(f"1. Training batch range (512-2048): NEW {train_speedup_new:.2f}x")
    print(f"2. Overall average: NEW {avg_speedup_new:.2f}x")
    print(f"3. Minimum threshold for integration: 1.3x speedup")
    
    # Determine winner based on training batch performance
    if train_speedup_new >= 1.3:
        recommendation = "NEW"
        speedup = train_speedup_new
        print(f"\n🎯 RECOMMENDATION: Integrate NEW method (4 autograd calls)")
        print(f"   Reason: Better performance on typical training batches")
        print(f"   Training batch speedup: {train_speedup_new:.2f}x")
        print(f"   Overall average: {avg_speedup_new:.2f}x")
    else:
        recommendation = "OLD"
        speedup = train_speedup_new
        print(f"\n⚠ RECOMMENDATION: Keep OLD method")
        print(f"   Reason: NEW speedup ({train_speedup_new:.2f}x) below threshold (1.3x)")
    
    print(f"\n{'='*70}")
    if recommendation != "OLD":
        print(f"✓ PROCEED: Integrate {recommendation} method into src/loss/physics_loss.py")
        print(f"  Expected training speedup: ~{speedup:.1f}x on derivatives")
        print(f"  Combined with other optimizations → 2-3x total epoch speedup")
    else:
        print(f"✗ SKIP Phase 1: Move to Phase 2 (BC precomputation) instead")
        print(f"  Derivative optimization insufficient on this GPU")
    print(f"{'='*70}")
    
    return passed_correctness and (avg_speedup_new >= 1.3)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

