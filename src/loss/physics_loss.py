"""Physics-informed loss function for the Schrödinger equation.

Implements the three-component loss from PRD §4:
    L = MSE₀ + MSE_b + MSE_f

where:
- MSE₀: Initial condition loss
- MSE_b: Boundary condition loss (periodic BC)
- MSE_f: PDE residual loss

Equation: i*h_t + 0.5*h_xx + |h|²*h = 0
"""

from typing import Tuple
import time

import torch
import torch.nn as nn


def compute_derivatives(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute derivatives using vectorized autograd (GPU-safe).

    Computes ∂h/∂t, ∂h/∂x, and ∂²h/∂x² using PyTorch autograd.
    All operations stay on the same device (GPU if available).

    Args:
        h: Complex field values, shape (batch_size,)
        x: Spatial coordinates, shape (batch_size, 1), requires_grad=True
        t: Temporal coordinates, shape (batch_size, 1), requires_grad=True

    Returns:
        Tuple of (h_t, h_x, h_xx):
        - h_t: ∂h/∂t, complex tensor
        - h_x: ∂h/∂x, complex tensor
        - h_xx: ∂²h/∂x², complex tensor
    """
    # Split into real and imaginary parts (no device movement)
    u = h.real
    v = h.imag

    # Create grad_outputs once (same device/dtype as u)
    ones = torch.ones_like(u)

    # First derivatives w.r.t time: compute both u_t and v_t
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

    # First derivatives w.r.t space: compute both u_x and v_x
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

    # Second derivatives w.r.t space: differentiate u_x and v_x
    # Need grad_outputs with same shape as u_x and v_x
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

    # Pack as complex (stays on device)
    h_t = torch.complex(u_t, v_t).squeeze()
    h_x = torch.complex(u_x, v_x).squeeze()
    h_xx = torch.complex(u_xx, v_xx).squeeze()

    return h_t, h_x, h_xx


def pde_residual(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_xx: torch.Tensor,
) -> torch.Tensor:
    """Compute the PDE residual: i*h_t + 0.5*h_xx + |h|²*h (GPU-safe).

    Args:
        h: Complex field
        h_t: Time derivative ∂h/∂t
        h_xx: Second spatial derivative ∂²h/∂x²

    Returns:
        Complex residual tensor
    """
    # r = i*h_t + 0.5*h_xx + |h|²*h
    # All operations stay on device
    residual = 1j * h_t + 0.5 * h_xx + (h.abs() ** 2) * h
    return residual


class SchrodingerLoss(nn.Module):
    """Physics-informed loss for the Schrödinger equation.

    Implements the three-term loss from PRD §4:
        L = MSE₀ + MSE_b + MSE_f

    Attributes:
        weights: Optional weights for each loss component (w0, wb, wf)
    """

    def __init__(
        self,
        weight_initial: float = 1.0,
        weight_boundary: float = 1.0,
        weight_residual: float = 1.0,
    ):
        """Initialize the physics-informed loss.

        Args:
            weight_initial: Weight for initial condition loss (default: 1.0)
            weight_boundary: Weight for boundary condition loss (default: 1.0)
            weight_residual: Weight for PDE residual loss (default: 1.0)
        """
        super(SchrodingerLoss, self).__init__()
        self.weight_initial = weight_initial
        self.weight_boundary = weight_boundary
        self.weight_residual = weight_residual
        # Stores timing information from the most recent forward call
        self.last_timings = {}

    def initial_condition_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t_0: torch.Tensor,
        u_0_true: torch.Tensor,
        v_0_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE₀: Initial condition loss.

        From PRD §4.1:
        MSE₀ = (1/N₀) Σ |h_θ(x_i, 0) - 2*sech(x_i)|²

        Args:
            model: Neural network model
            x_0: Spatial points at t=0, shape (N_0, 1)
            t_0: Time points (all zeros), shape (N_0, 1)
            u_0_true: True real part at initial condition
            v_0_true: True imaginary part at initial condition

        Returns:
            Scalar MSE loss
        """
        # Verify IC times are constant (typically t=0)
        with torch.no_grad():
            t0_flat = t_0.view(-1)
            dt0 = (torch.max(t0_flat) - torch.min(t0_flat)).abs().item()
            if dt0 > 1e-8:
                print("[IC ERROR] Initial-condition samples contain multiple time values.")
                raise ValueError(
                    "Initial-condition samples must all share the same time (expected t=0)."
                )

        # Predict at initial condition
        uv_pred = model(x_0, t_0)
        u_pred = uv_pred[:, 0]
        v_pred = uv_pred[:, 1]

        # MSE between prediction and true initial condition
        mse_u = torch.mean((u_pred - u_0_true) ** 2)
        mse_v = torch.mean((v_pred - v_0_true) ** 2)

        return mse_u + mse_v

    def boundary_condition_loss(
        self,
        model: nn.Module,
        x_b: torch.Tensor,
        t_b: torch.Tensor,
        x_min: float = -5.0,
        x_max: float = 5.0,
    ) -> torch.Tensor:
        """Compute MSE_b: Boundary condition loss for periodic BC (GPU-safe).

        From PRD §4.2:
        MSE_b = (1/N_b) Σ (|h(-5,t_i) - h(5,t_i)|² +
                          |h_x(-5,t_i) - h_x(5,t_i)|²)

        Args:
            model: Neural network model
            x_b: Boundary x-coordinates (mix of x_min and x_max)
            t_b: Time coordinates at boundaries
            x_min: Left boundary (default: -5.0)
            x_max: Right boundary (default: 5.0)

        Returns:
            Scalar MSE loss
        """
        # Separate left and right boundary points using isclose to x_min/x_max
        x_vals = x_b.view(-1)
        left_mask = torch.isclose(
            x_vals,
            torch.as_tensor(x_min, device=x_vals.device, dtype=x_vals.dtype),
            atol=1e-6,
        )
        right_mask = torch.isclose(
            x_vals,
            torch.as_tensor(x_max, device=x_vals.device, dtype=x_vals.dtype),
            atol=1e-6,
        )

        n_left = int(left_mask.sum().item())
        n_right = int(right_mask.sum().item())
        if n_left == 0 or n_right == 0:
            print("[BC ERROR] Missing boundary samples on one side.")
            print(f"           n_left={n_left}, n_right={n_right}")
            raise ValueError("Boundary-condition samples must include both x=x_min and x=x_max.")

        # Extract and verify times on both sides
        t_left_vec = t_b[left_mask].view(-1)
        t_right_vec = t_b[right_mask].view(-1)
        if n_left != n_right:
            print("[BC ERROR] Boundary time counts differ between left and right.")
            print(f"           n_left={n_left}, n_right={n_right}")
            raise ValueError(
                "Boundary-condition times must be paired (same count on both sides)."
            )

        # Sort and compare times (pair by sorted order)
        t_left_sorted, _ = torch.sort(t_left_vec)
        t_right_sorted, _ = torch.sort(t_right_vec)
        max_dt = torch.max(torch.abs(t_left_sorted - t_right_sorted)).item()
        if max_dt > 1e-8:
            print("[BC ERROR] Boundary times are not matched between x_min and x_max.")
            print(f"           max |Δt| between sorted times = {max_dt:.3e}")
            raise ValueError(
                "Boundary-condition times must match across x=±L for periodic BC."
            )

        # After verification, use sorted, paired times
        t_left = t_left_sorted.unsqueeze(1)
        t_right = t_right_sorted.unsqueeze(1)
        
        # Enable gradients (stays on device)
        t_left = t_left.requires_grad_(True) if not t_left.requires_grad else t_left
        t_right = t_right.requires_grad_(True) if not t_right.requires_grad else t_right

        # Create boundary coordinates (same device/dtype as t_left/t_right)
        x_left = torch.full_like(t_left, x_min, requires_grad=True)
        x_right = torch.full_like(t_right, x_max, requires_grad=True)

        # Predict at boundaries
        h_left = model.predict_h(x_left, t_left)
        h_right = model.predict_h(x_right, t_right)

        # Compute spatial derivatives at boundaries
        _, h_x_left, _ = compute_derivatives(h_left, x_left, t_left)
        _, h_x_right, _ = compute_derivatives(h_right, x_right, t_right)

        # Periodic BC: h(-5,t) = h(5,t) and h_x(-5,t) = h_x(5,t)
        # Use |a-b|² = (a-b).real² + (a-b).imag² for complex differences
        diff_value = h_left - h_right
        mse_value = torch.mean(diff_value.real ** 2 + diff_value.imag ** 2)
        
        diff_derivative = h_x_left - h_x_right
        mse_derivative = torch.mean(
            diff_derivative.real ** 2 + diff_derivative.imag ** 2
        )

        return mse_value + mse_derivative

    def pde_residual_loss(
        self,
        model: nn.Module,
        x_f: torch.Tensor,
        t_f: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE_f: PDE residual loss (vectorized, GPU-safe).

        From PRD §4.3:
        r_θ = i*∂_t h_θ + 0.5*∂_xx h_θ + |h_θ|²*h_θ
        MSE_f = (1/N_f) Σ |r_θ(x_i, t_i)|²

        Args:
            model: Neural network model
            x_f: Collocation spatial points, shape (N_f, 1)
            t_f: Collocation temporal points, shape (N_f, 1)

        Returns:
            Scalar MSE loss
        """
        # Enable gradient computation (no device movement)
        x_f = x_f.requires_grad_(True) if not x_f.requires_grad else x_f
        t_f = t_f.requires_grad_(True) if not t_f.requires_grad else t_f

        # Predict complex field
        t0 = time.time()
        h = model.predict_h(x_f, t_f)
        t1 = time.time()

        # Compute derivatives (vectorized, stays on device)
        h_t, h_x, h_xx = compute_derivatives(h, x_f, t_f)
        t2 = time.time()

        # Compute PDE residual
        residual = pde_residual(h, h_t, h_xx)
        t3 = time.time()

        # MSE of residual magnitude: |r|² = r.real² + r.imag²
        mse_residual = torch.mean(residual.real ** 2 + residual.imag ** 2)

        # Record granular timings for PDE term
        # Note: These are per-batch timings; the training loop aggregates averages
        if isinstance(getattr(self, "last_timings", None), dict):
            self.last_timings.update({
                "time/pde/predict": t1 - t0,
                "time/pde/derivatives": t2 - t1,
                "time/pde/residual": t3 - t2,
            })

        return mse_residual

    def forward(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t_0: torch.Tensor,
        u_0: torch.Tensor,
        v_0: torch.Tensor,
        x_b: torch.Tensor,
        t_b: torch.Tensor,
        x_f: torch.Tensor,
        t_f: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """Compute total physics-informed loss.

        Args:
            model: Neural network model
            x_0, t_0: Initial condition points
            u_0, v_0: True values at initial condition
            x_b, t_b: Boundary condition points
            x_f, t_f: Collocation points for PDE residual
            return_components: If True, return (total, mse0, mseb, msef)

        Returns:
            Total loss, or tuple of (total, mse0, mseb, msef) if
            return_components=True
        """
        # Reset timings container for this batch
        self.last_timings = {}

        # Compute individual loss components with timings
        t_ic0 = time.time()
        mse_0 = self.initial_condition_loss(model, x_0, t_0, u_0, v_0)
        t_ic1 = time.time()

        t_bc0 = time.time()
        mse_b = self.boundary_condition_loss(model, x_b, t_b)
        t_bc1 = time.time()

        t_pde0 = time.time()
        mse_f = self.pde_residual_loss(model, x_f, t_f)
        t_pde1 = time.time()

        # Store coarse timings
        self.last_timings.update(
            {
                "time/ic": t_ic1 - t_ic0,
                "time/bc": t_bc1 - t_bc0,
                "time/pde": t_pde1 - t_pde0,
            }
        )

        # Weighted total loss
        total_loss = (
            self.weight_initial * mse_0
            + self.weight_boundary * mse_b
            + self.weight_residual * mse_f
        )

        if return_components:
            return total_loss, mse_0, mse_b, mse_f
        else:
            return total_loss


if __name__ == "__main__":
    """Module can be imported."""
    print("Physics loss module loaded successfully.")
    print("Use tests/test_loss.py to test the loss functions.")


