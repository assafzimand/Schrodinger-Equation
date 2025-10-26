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

import torch
import torch.nn as nn


def compute_derivatives(
    h: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute spatial and temporal derivatives of h using autograd.

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
    # Split into real and imaginary parts for autograd
    u = h.real
    v = h.imag

    # First derivatives
    # ∂u/∂t and ∂v/∂t
    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    v_t = torch.autograd.grad(
        v,
        t,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True,
    )[0]

    # ∂u/∂x and ∂v/∂x
    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]

    v_x = torch.autograd.grad(
        v,
        x,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Second derivatives
    # ∂²u/∂x² and ∂²v/∂x²
    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    v_xx = torch.autograd.grad(
        v_x,
        x,
        grad_outputs=torch.ones_like(v_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Combine into complex derivatives
    h_t = torch.complex(u_t, v_t).squeeze()
    h_x = torch.complex(u_x, v_x).squeeze()
    h_xx = torch.complex(u_xx, v_xx).squeeze()

    return h_t, h_x, h_xx


def pde_residual(
    h: torch.Tensor,
    h_t: torch.Tensor,
    h_xx: torch.Tensor,
) -> torch.Tensor:
    """Compute the PDE residual: i*h_t + 0.5*h_xx + |h|²*h.

    Args:
        h: Complex field
        h_t: Time derivative ∂h/∂t
        h_xx: Second spatial derivative ∂²h/∂x²

    Returns:
        Complex residual tensor
    """
    # r = i*h_t + 0.5*h_xx + |h|²*h
    residual = 1j * h_t + 0.5 * h_xx + (torch.abs(h) ** 2) * h
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
        """Compute MSE_b: Boundary condition loss for periodic BC.

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
        # Separate left and right boundary points
        left_mask = x_b[:, 0] < 0
        right_mask = ~left_mask

        n_left = left_mask.sum()
        n_right = right_mask.sum()
        n_pairs = min(n_left, n_right)

        if n_pairs == 0:
            return torch.tensor(0.0, device=x_b.device)

        # Get matching time points for both boundaries
        t_left = t_b[left_mask][:n_pairs].clone().detach().requires_grad_(True)
        t_right = t_b[right_mask][:n_pairs].clone().detach().requires_grad_(True)

        # Create boundary coordinates
        x_left = torch.full_like(t_left, x_min, requires_grad=True)
        x_right = torch.full_like(t_right, x_max, requires_grad=True)

        # Predict at boundaries
        h_left = model.predict_h(x_left, t_left)
        h_right = model.predict_h(x_right, t_right)

        # Compute spatial derivatives at boundaries
        _, h_x_left, _ = compute_derivatives(h_left, x_left, t_left)
        _, h_x_right, _ = compute_derivatives(h_right, x_right, t_right)

        # Periodic BC: h(-5,t) = h(5,t) and h_x(-5,t) = h_x(5,t)
        mse_value = torch.mean(torch.abs(h_left - h_right) ** 2)
        mse_derivative = torch.mean(torch.abs(h_x_left - h_x_right) ** 2)

        return mse_value + mse_derivative

    def pde_residual_loss(
        self,
        model: nn.Module,
        x_f: torch.Tensor,
        t_f: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE_f: PDE residual loss.

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
        # Enable gradient computation
        x_f = x_f.clone().detach().requires_grad_(True)
        t_f = t_f.clone().detach().requires_grad_(True)

        # Predict complex field
        h = model.predict_h(x_f, t_f)

        # Compute derivatives
        h_t, h_x, h_xx = compute_derivatives(h, x_f, t_f)

        # Compute PDE residual
        residual = pde_residual(h, h_t, h_xx)

        # MSE of residual magnitude
        mse_residual = torch.mean(torch.abs(residual) ** 2)

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
        # Compute individual loss components
        mse_0 = self.initial_condition_loss(model, x_0, t_0, u_0, v_0)
        mse_b = self.boundary_condition_loss(model, x_b, t_b)
        mse_f = self.pde_residual_loss(model, x_f, t_f)

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


