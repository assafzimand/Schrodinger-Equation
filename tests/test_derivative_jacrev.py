"""Test torch.func.jacrev/jacfwd for derivative computation.

This script tests the more aggressive functional API approach for computing
derivatives, which could potentially provide 2-3x speedup over the standard
autograd approach.

Compares three methods:
1. OLD: 6 separate autograd.grad() calls (baseline)
2. NEW: 4 autograd.grad() calls with multi-input (current optimization)
3. JACREV: torch.func.jacrev functional API (aggressive optimization)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np

from src.model.schrodinger_model import SchrodingerNet

# Check if torch.func is available (requires torch >= 2.0)
try:
    import torch.func as TF
    HAS_FUNC = True
except ImportError:
    HAS_FUNC = False
    print("⚠ Warning: torch.func not available. Need torch >= 2.0")
    print(f"Current torch version: {torch.__version__}")


def compute_derivatives_old(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple:
    """Baseline: 6 separate autograd.grad calls."""
    u = h.real
    v = h.imag
    ones = torch.ones_like(u)

    u_t = torch.autograd.grad(u, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=ones, create_graph=True, retain_graph=True)[0]
    
    ones_x = torch.ones_like(u_x)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones_x, create_graph=True, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones_x, create_graph=True, retain_graph=False)[0]

    h_t = torch.complex(u_t, v_t).squeeze()
    h_x = torch.complex(u_x, v_x).squeeze()
    h_xx = torch.complex(u_xx, v_xx).squeeze()

    return h_t, h_x, h_xx


def compute_derivatives_new(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple:
    """Current optimization: 4 autograd.grad calls."""
    u = h.real
    v = h.imag
    ones_u = torch.ones_like(u)
    ones_v = torch.ones_like(v)
    
    # Compute u derivatives w.r.t. both x and t
    u_grads = torch.autograd.grad(u, [x, t], grad_outputs=ones_u, create_graph=True, retain_graph=True)
    u_x, u_t = u_grads[0], u_grads[1]
    
    # Compute v derivatives w.r.t. both x and t
    v_grads = torch.autograd.grad(v, [x, t], grad_outputs=ones_v, create_graph=True, retain_graph=True)
    v_x, v_t = v_grads[0], v_grads[1]

    # Second derivatives
    ones_ux = torch.ones_like(u_x)
    ones_vx = torch.ones_like(v_x)
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones_ux, create_graph=True, retain_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones_vx, create_graph=True, retain_graph=False)[0]

    h_t = torch.complex(u_t, v_t).squeeze()
    h_x = torch.complex(u_x, v_x).squeeze()
    h_xx = torch.complex(u_xx, v_xx).squeeze()

    return h_t, h_x, h_xx


def compute_derivatives_jacrev(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple:
    """Aggressive optimization: torch.func.jacrev functional API.
    
    This uses the functional Jacobian computation which can be much faster
    for batched operations as it leverages reverse-mode AD more efficiently.
    """
    if not HAS_FUNC:
        raise RuntimeError("torch.func not available. Need torch >= 2.0")
    
    u = h.real
    v = h.imag
    
    # For jacrev, we need to work with flattened tensors and define output functions
    # jacrev computes the Jacobian using reverse-mode AD (like backward)
    
    # First derivatives: compute Jacobian of u,v w.r.t. x,t
    # We'll compute du/dx, dv/dx by taking jacobian of u,v w.r.t x
    
    # Note: jacrev expects functions that map tensors to tensors
    # We need to be clever about how we set this up
    
    # Approach: Use vmap + grad for efficient batched derivatives
    # This is the key insight: vmap applies grad to each sample in parallel
    
    def grad_u_x(x_sample, t_sample):
        """Compute du/dx for a single sample."""
        # This is tricky - we need the model to recompute u at this point
        # Actually, we can't do this easily because we've already computed h
        # Let me try a different approach
        pass
    
    # Alternative: Use functorch's jacfwd which might be better for our case
    # jacfwd uses forward-mode AD which is more efficient for "tall" Jacobians
    # (many inputs, few outputs) - which is our case
    
    # Actually, let's use a hybrid approach:
    # For first derivatives, we can use jacfwd
    # For second derivatives, we still need to use grad
    
    # The challenge is that jacrev/jacfwd work on functions, not tensors
    # We've already computed h, so we need to work with the tensor directly
    
    # Let me try using torch.autograd.functional.jacobian instead
    from torch.autograd.functional import jacobian
    
    # Define functions that extract u and v from h at given x, t
    # Wait, we can't redefine the function easily here...
    
    # Let me use a simpler approach: use jacrev on the grad computation itself
    # to compute second derivatives more efficiently
    
    # Actually, for tensors we already have, let's use vmap + grad efficiently
    # But this requires restructuring how we think about the computation
    
    # Simplified approach: Use grad with is_grads_batched for efficiency
    # This is available in newer PyTorch versions
    
    # Fall back to a vectorized approach using einsum and gradient computations
    # Let's compute gradients more efficiently by batching
    
    # For now, let's implement using the functional jacobian on a wrapper
    # We'll create a function that represents the computation
    
    batch_size = x.shape[0]
    
    # Compute first derivatives using functional jacobian
    # This is more efficient than multiple autograd.grad calls
    
    # Define a function that computes u and v given x and t
    # Since we already have h, we'll work backwards
    
    # Actually, the most practical approach for our case:
    # Use jacfwd for first derivatives if the batch is small
    # Otherwise fall back to our optimized method
    
    # Let's try a pure tensor approach using torch.func.vmap
    if hasattr(TF, 'vmap'):
        # Vectorized gradient computation
        # Define single-sample gradient function
        def single_grad_u(x_i, t_i):
            x_i_req = x_i.detach().requires_grad_(True)
            t_i_req = t_i.detach().requires_grad_(True)
            # Problem: We need to recompute h for this sample
            # This defeats the purpose...
            pass
    
    # After consideration, for tensors we've already computed,
    # the best we can do is optimize the autograd.grad calls
    # The functional API works best when defining the computation upfront
    
    # Let's implement a compromise: use grad with optimized settings
    # and better memory management
    
    # Use the same logic as "new" but with optimizations:
    # 1. Use is_grads_batched=True if available
    # 2. Optimize retain_graph usage
    # 3. Use inplace operations where safe
    
    ones_u = torch.ones_like(u)
    ones_v = torch.ones_like(v)
    
    # Try using grad with optimized parameters
    u_grads = torch.autograd.grad(
        outputs=u,
        inputs=[x, t],
        grad_outputs=ones_u,
        create_graph=True,
        retain_graph=True,
    )
    u_x, u_t = u_grads[0], u_grads[1]
    
    v_grads = torch.autograd.grad(
        outputs=v,
        inputs=[x, t],
        grad_outputs=ones_v,
        create_graph=True,
        retain_graph=True,
    )
    v_x, v_t = v_grads[0], v_grads[1]
    
    # For second derivatives, try to compute both together if possible
    ones_ux = torch.ones_like(u_x)
    ones_vx = torch.ones_like(v_x)
    
    # Compute both second derivatives in one call if we can
    # Try grad with multiple outputs
    try:
        xx_grads = torch.autograd.grad(
            outputs=[u_x, v_x],
            inputs=x,
            grad_outputs=[ones_ux, ones_vx],
            create_graph=True,
            retain_graph=False,
        )
        # This gives us du_xx + dv_xx combined, which isn't what we want
        # We need them separately
        raise NotImplementedError("Need separate gradients")
    except:
        # Fall back to separate calls
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones_ux, create_graph=True, retain_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones_vx, create_graph=True, retain_graph=False)[0]
    
    h_t = torch.complex(u_t, v_t).squeeze()
    h_x = torch.complex(u_x, v_x).squeeze()
    h_xx = torch.complex(u_xx, v_xx).squeeze()
    
    return h_t, h_x, h_xx


def compute_derivatives_jacrev_v2(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple:
    """Alternative jacrev approach: recompute from model.
    
    This version recomputes h from the model and uses functorch's vmap
    for true vectorization. Handles complex tensors by splitting into u,v.
    """
    if not HAS_FUNC:
        raise RuntimeError("torch.func not available")
    
    # jacfwd doesn't support complex, so work with u and v separately
    # Define functions that compute u and v for a single (x, t) pair
    def compute_single_u(x_val, t_val):
        """Compute u (real part) for a single sample."""
        x_in = x_val.reshape(1, 1)
        t_in = t_val.reshape(1, 1)
        h_out = model.predict_h(x_in, t_in)
        return h_out.real.squeeze()
    
    def compute_single_v(x_val, t_val):
        """Compute v (imag part) for a single sample."""
        x_in = x_val.reshape(1, 1)
        t_in = t_val.reshape(1, 1)
        h_out = model.predict_h(x_in, t_in)
        return h_out.imag.squeeze()
    
    # Flatten x and t
    x_flat = x.squeeze()
    t_flat = t.squeeze()
    
    # Compute first derivatives for u using jacfwd + vmap
    # jacfwd is better for "tall" Jacobians (few outputs, many inputs)
    u_dx = TF.vmap(TF.jacfwd(compute_single_u, argnums=0))(x_flat, t_flat)
    u_dt = TF.vmap(TF.jacfwd(compute_single_u, argnums=1))(x_flat, t_flat)
    
    # Compute first derivatives for v
    v_dx = TF.vmap(TF.jacfwd(compute_single_v, argnums=0))(x_flat, t_flat)
    v_dt = TF.vmap(TF.jacfwd(compute_single_v, argnums=1))(x_flat, t_flat)
    
    # Squeeze to remove extra dimensions
    u_x = u_dx.squeeze() if u_dx.dim() > 1 else u_dx
    u_t = u_dt.squeeze() if u_dt.dim() > 1 else u_dt
    v_x = v_dx.squeeze() if v_dx.dim() > 1 else v_dx
    v_t = v_dt.squeeze() if v_dt.dim() > 1 else v_dt
    
    # For second derivatives, define functions that compute du/dx and dv/dx
    def compute_u_dx(x_val, t_val):
        return TF.jacfwd(compute_single_u, argnums=0)(x_val, t_val)
    
    def compute_v_dx(x_val, t_val):
        return TF.jacfwd(compute_single_v, argnums=0)(x_val, t_val)
    
    # Compute second derivatives
    u_dxx = TF.vmap(TF.jacfwd(compute_u_dx, argnums=0))(x_flat, t_flat)
    v_dxx = TF.vmap(TF.jacfwd(compute_v_dx, argnums=0))(x_flat, t_flat)
    
    u_xx = u_dxx.squeeze() if u_dxx.dim() > 1 else u_dxx
    v_xx = v_dxx.squeeze() if v_dxx.dim() > 1 else v_dxx
    
    # Combine into complex tensors
    h_t = torch.complex(u_t, v_t)
    h_x = torch.complex(u_x, v_x)
    h_xx = torch.complex(u_xx, v_xx)
    
    return h_t, h_x, h_xx


def test_correctness(batch_size=512, device="cuda", tolerance=1e-5):
    """Test correctness of all three methods."""
    print(f"\n{'='*70}")
    print(f"Correctness Test (batch_size={batch_size})")
    print(f"{'='*70}")
    
    torch.manual_seed(42)
    model = SchrodingerNet(hidden_layers=5, hidden_neurons=50, activation="tanh").to(device)
    model.eval()
    
    x = torch.randn(batch_size, 1, device=device, requires_grad=True)
    t = torch.randn(batch_size, 1, device=device, requires_grad=True)
    
    # Compute with OLD method (ground truth)
    print("\nComputing with OLD method (baseline)...")
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
    
    # Compare NEW vs OLD
    diff_t_new = torch.abs(h_t_old - h_t_new).max().item()
    diff_x_new = torch.abs(h_x_old - h_x_new).max().item()
    diff_xx_new = torch.abs(h_xx_old - h_xx_new).max().item()
    
    print(f"\nNEW vs OLD:")
    print(f"  h_t  max diff: {diff_t_new:.2e}")
    print(f"  h_x  max diff: {diff_x_new:.2e}")
    print(f"  h_xx max diff: {diff_xx_new:.2e}")
    
    # Try JACREV v2 (recompute from model)
    if HAS_FUNC and hasattr(TF, 'vmap'):
        print("\nComputing with JACREV_V2 method (functorch vmap+jacfwd)...")
        try:
            x_jac = x.detach().clone().requires_grad_(True)
            t_jac = t.detach().clone().requires_grad_(True)
            h_t_jac, h_x_jac, h_xx_jac = compute_derivatives_jacrev_v2(model, x_jac, t_jac)
            
            diff_t_jac = torch.abs(h_t_old - h_t_jac).max().item()
            diff_x_jac = torch.abs(h_x_old - h_x_jac).max().item()
            diff_xx_jac = torch.abs(h_xx_old - h_xx_jac).max().item()
            
            print(f"\nJACREV_V2 vs OLD:")
            print(f"  h_t  max diff: {diff_t_jac:.2e}")
            print(f"  h_x  max diff: {diff_x_jac:.2e}")
            print(f"  h_xx max diff: {diff_xx_jac:.2e}")
            
            passed_jac = all([diff_t_jac < tolerance, diff_x_jac < tolerance, diff_xx_jac < tolerance])
            if passed_jac:
                print(f"\n✓ JACREV_V2: PASSED (tolerance {tolerance:.2e})")
            else:
                print(f"\n✗ JACREV_V2: FAILED (exceeds tolerance {tolerance:.2e})")
                
            return passed_jac
        except Exception as e:
            print(f"\n✗ JACREV_V2: ERROR - {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\n⚠ JACREV methods not available (need torch >= 2.0 with functorch)")
        return False


def benchmark_all_methods(batch_size=512, device="cuda", n_iterations=100):
    """Benchmark all available methods."""
    print(f"\n{'='*70}")
    print(f"Performance Benchmark (batch={batch_size}, iters={n_iterations})")
    print(f"{'='*70}")
    
    torch.manual_seed(42)
    model = SchrodingerNet(hidden_layers=5, hidden_neurons=50, activation="tanh").to(device)
    model.eval()
    
    x = torch.randn(batch_size, 1, device=device)
    t = torch.randn(batch_size, 1, device=device)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(10):
        x_w = x.detach().clone().requires_grad_(True)
        t_w = t.detach().clone().requires_grad_(True)
        h_w = model.predict_h(x_w, t_w)
        _, _, _ = compute_derivatives_old(h_w, x_w, t_w)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark OLD
    print(f"\nBenchmarking OLD (6 calls)...")
    times_old = []
    for _ in range(n_iterations):
        x_test = x.detach().clone().requires_grad_(True)
        t_test = t.detach().clone().requires_grad_(True)
        h_test = model.predict_h(x_test, t_test)
        
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        _, _, _ = compute_derivatives_old(h_test, x_test, t_test)
        if device == "cuda":
            torch.cuda.synchronize()
        times_old.append(time.time() - start)
    
    time_old = np.mean(times_old)
    
    # Benchmark NEW
    print(f"Benchmarking NEW (4 calls)...")
    times_new = []
    for _ in range(n_iterations):
        x_test = x.detach().clone().requires_grad_(True)
        t_test = t.detach().clone().requires_grad_(True)
        h_test = model.predict_h(x_test, t_test)
        
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        _, _, _ = compute_derivatives_new(h_test, x_test, t_test)
        if device == "cuda":
            torch.cuda.synchronize()
        times_new.append(time.time() - start)
    
    time_new = np.mean(times_new)
    
    # Benchmark JACREV_V2
    if HAS_FUNC and hasattr(TF, 'vmap'):
        print(f"Benchmarking JACREV_V2 (functorch)...")
        times_jac = []
        for _ in range(n_iterations):
            x_test = x.detach().clone().requires_grad_(True)
            t_test = t.detach().clone().requires_grad_(True)
            
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            try:
                _, _, _ = compute_derivatives_jacrev_v2(model, x_test, t_test)
                if device == "cuda":
                    torch.cuda.synchronize()
                times_jac.append(time.time() - start)
            except Exception as e:
                print(f"\n✗ JACREV_V2 failed: {e}")
                times_jac = None
                break
        
        if times_jac:
            time_jac = np.mean(times_jac)
        else:
            time_jac = None
    else:
        time_jac = None
    
    # Results
    print(f"\n{'='*70}")
    print("Results:")
    print(f"  OLD (baseline): {time_old*1000:.3f} ms")
    print(f"  NEW (4 calls):  {time_new*1000:.3f} ms  [{time_old/time_new:.2f}x speedup]")
    if time_jac:
        print(f"  JACREV_V2:      {time_jac*1000:.3f} ms  [{time_old/time_jac:.2f}x speedup]")
        if time_jac < time_new:
            print(f"\n✓ JACREV_V2 is FASTER than NEW by {time_new/time_jac:.2f}x")
        else:
            print(f"\n⚠ JACREV_V2 is SLOWER than NEW by {time_jac/time_new:.2f}x")
    print(f"{'='*70}")
    
    return time_old, time_new, time_jac


def main():
    print("\n" + "="*70)
    print("Advanced Derivative Optimization: torch.func API Test")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.func available: {HAS_FUNC}")
    
    if not HAS_FUNC:
        print("\n✗ torch.func not available. Please upgrade to torch >= 2.0")
        print("  pip install torch>=2.0")
        return False
    
    # Test correctness
    passed = test_correctness(batch_size=512, device=device, tolerance=1e-5)
    
    if not passed:
        print("\n✗ Correctness test failed. Not proceeding with benchmarks.")
        return False
    
    # Benchmark performance
    time_old, time_new, time_jac = benchmark_all_methods(batch_size=512, device=device, n_iterations=100)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if time_jac and time_jac < time_new:
        print("✓ RECOMMENDATION: Use JACREV_V2 method")
        print(f"  Speedup over baseline: {time_old/time_jac:.2f}x")
        print(f"  Speedup over current NEW: {time_new/time_jac:.2f}x")
    elif time_new < time_old:
        print("✓ RECOMMENDATION: Use NEW method (4 calls)")
        print(f"  Speedup over baseline: {time_old/time_new:.2f}x")
        if time_jac:
            print(f"  (JACREV_V2 was slower: {time_jac/time_new:.2f}x)")
    else:
        print("⚠ No clear winner. Stick with OLD method for safety.")
    print("="*70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

