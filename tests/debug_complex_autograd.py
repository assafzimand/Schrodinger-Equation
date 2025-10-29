"""Debug why complex autograd gives different results.

Test different approaches to computing derivatives of complex tensors.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.model.schrodinger_model import SchrodingerNet

def test_complex_autograd():
    """Compare different ways of computing complex derivatives."""
    
    print("="*70)
    print("DEBUGGING COMPLEX AUTOGRAD")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    
    # Create simple test case
    model = SchrodingerNet(hidden_layers=5, hidden_neurons=50, activation="tanh").to(device)
    model.eval()
    
    batch_size = 4  # Small for debugging
    x = torch.randn(batch_size, 1, device=device, requires_grad=True)
    t = torch.randn(batch_size, 1, device=device, requires_grad=True)
    
    # Get h (complex)
    h = model.predict_h(x, t)
    u = h.real
    v = h.imag
    
    print(f"\nInput shapes: x={x.shape}, t={t.shape}")
    print(f"Output: h={h.shape} (complex)")
    print(f"  u (real) shape: {u.shape}")
    print(f"  v (imag) shape: {v.shape}")
    
    # Method 1: Split u, v and compute separately (GROUND TRUTH)
    print("\n" + "="*70)
    print("METHOD 1: Split u/v, compute separately (BASELINE)")
    print("="*70)
    
    ones_u = torch.ones_like(u)
    ones_v = torch.ones_like(v)
    
    u_x_split = torch.autograd.grad(u, x, grad_outputs=ones_u, create_graph=True, retain_graph=True)[0]
    v_x_split = torch.autograd.grad(v, x, grad_outputs=ones_v, create_graph=True, retain_graph=True)[0]
    
    h_x_split = torch.complex(u_x_split, v_x_split)
    
    print(f"u_x: {u_x_split.flatten()[:3]}")
    print(f"v_x: {v_x_split.flatten()[:3]}")
    print(f"h_x: {h_x_split.flatten()[:3]}")
    
    # Method 2: Direct complex autograd with ones_like
    print("\n" + "="*70)
    print("METHOD 2: Direct complex autograd (grad_outputs=ones_like)")
    print("="*70)
    
    x2 = x.detach().clone().requires_grad_(True)
    t2 = t.detach().clone().requires_grad_(True)
    h2 = model.predict_h(x2, t2)
    
    ones_h = torch.ones_like(h2)
    print(f"grad_outputs (ones_like): {ones_h.flatten()[:3]}")
    print(f"  dtype: {ones_h.dtype}, is_complex: {ones_h.is_complex()}")
    
    h_x_direct = torch.autograd.grad(h2, x2, grad_outputs=ones_h, create_graph=True, retain_graph=True)[0]
    
    print(f"h_x from direct: {h_x_direct.flatten()[:3]}")
    
    diff = torch.abs(h_x_split - h_x_direct).max().item()
    print(f"\nDifference from Method 1: {diff:.6e}")
    
    # Method 3: Try with complex grad_outputs = 1+0j
    print("\n" + "="*70)
    print("METHOD 3: Complex autograd with grad_outputs=complex(1, 0)")
    print("="*70)
    
    x3 = x.detach().clone().requires_grad_(True)
    t3 = t.detach().clone().requires_grad_(True)
    h3 = model.predict_h(x3, t3)
    
    # Create complex grad_outputs with real=1, imag=0
    grad_out_real = torch.ones_like(h3.real)
    grad_out_imag = torch.zeros_like(h3.imag)
    grad_out_complex = torch.complex(grad_out_real, grad_out_imag)
    
    print(f"grad_outputs (1+0j): {grad_out_complex.flatten()[:3]}")
    
    h_x_complex_grad = torch.autograd.grad(h3, x3, grad_outputs=grad_out_complex, create_graph=True, retain_graph=True)[0]
    
    print(f"h_x from complex grad: {h_x_complex_grad.flatten()[:3]}")
    
    diff = torch.abs(h_x_split - h_x_complex_grad).max().item()
    print(f"\nDifference from Method 1: {diff:.6e}")
    
    # Method 4: Check what PyTorch doc says about complex autograd
    print("\n" + "="*70)
    print("METHOD 4: Understanding PyTorch complex autograd")
    print("="*70)
    
    # According to PyTorch docs, for complex functions:
    # PyTorch computes the Wirtinger derivative by default
    # But we can get the real derivative by working with .real and .imag separately
    
    # Let's check if the issue is the grad_outputs or the computation itself
    x4 = x.detach().clone().requires_grad_(True)
    t4 = t.detach().clone().requires_grad_(True)
    h4 = model.predict_h(x4, t4)
    
    # Split and compute real/imag separately using same graph
    u4 = h4.real
    v4 = h4.imag
    
    # Now try computing grad of complex h by computing real/imag grads
    u_x_4 = torch.autograd.grad(u4, x4, grad_outputs=torch.ones_like(u4), create_graph=True, retain_graph=True)[0]
    v_x_4 = torch.autograd.grad(v4, x4, grad_outputs=torch.ones_like(v4), create_graph=True, retain_graph=True)[0]
    h_x_4 = torch.complex(u_x_4, v_x_4)
    
    print(f"h_x from split on same graph: {h_x_4.flatten()[:3]}")
    diff = torch.abs(h_x_split - h_x_4).max().item()
    print(f"Difference from Method 1: {diff:.6e}")
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\nMethod 1 (baseline - split u/v):")
    print(f"  h_x[0:3] = {h_x_split.flatten()[:3]}")
    
    print(f"\nMethod 2 (direct complex, ones_like):")
    print(f"  h_x[0:3] = {h_x_direct.flatten()[:3]}")
    print(f"  Error: {torch.abs(h_x_split - h_x_direct).max().item():.6e}")
    
    print(f"\nMethod 3 (direct complex, 1+0j grad):")
    print(f"  h_x[0:3] = {h_x_complex_grad.flatten()[:3]}")
    print(f"  Error: {torch.abs(h_x_split - h_x_complex_grad).max().item():.6e}")
    
    print(f"\nMethod 4 (split on same graph):")
    print(f"  h_x[0:3] = {h_x_4.flatten()[:3]}")
    print(f"  Error: {torch.abs(h_x_split - h_x_4).max().item():.6e}")
    
    # Determine which method works
    if torch.abs(h_x_split - h_x_direct).max().item() < 1e-6:
        print("\n✓ Method 2 (direct ones_like) WORKS!")
        return "METHOD2"
    elif torch.abs(h_x_split - h_x_complex_grad).max().item() < 1e-6:
        print("\n✓ Method 3 (complex 1+0j) WORKS!")
        return "METHOD3"
    else:
        print("\n✗ Neither direct complex method works correctly")
        print("  Must use split u/v approach (4 calls)")
        return "SPLIT"


if __name__ == "__main__":
    result = test_complex_autograd()
    print(f"\nResult: {result}")

