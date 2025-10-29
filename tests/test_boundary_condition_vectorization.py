"""Regression test for vectorized boundary-condition derivatives.

Ensures that the optimized implementation in
`SchrodingerLoss.boundary_condition_loss` matches the previous
per-boundary computation that performed two separate derivative calls.
"""

import torch

from src.loss.physics_loss import SchrodingerLoss, compute_derivatives


class AnalyticModel(torch.nn.Module):
    """Simple analytic model with closed-form derivatives.

    The model output is chosen so that gradients can be computed easily
    and deterministically without relying on a trained network.
    """

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        u = torch.sin(x) * torch.cos(t)
        v = torch.cos(x) * torch.sin(t)
        return torch.cat((u, v), dim=1)

    def predict_h(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        uv = self.forward(x, t)
        return torch.complex(uv[:, 0], uv[:, 1])


def boundary_loss_reference(
    model: AnalyticModel,
    x_left: torch.Tensor,
    t_left: torch.Tensor,
    x_right: torch.Tensor,
    t_right: torch.Tensor,
) -> torch.Tensor:
    """Baseline boundary-condition loss using two autograd calls.

    Mirrors the previous implementation where left/right boundaries were
    differentiated independently.
    """

    x_left_ref = x_left.clone().detach().requires_grad_(True)
    t_left_ref = t_left.clone().detach().requires_grad_(True)
    x_right_ref = x_right.clone().detach().requires_grad_(True)
    t_right_ref = t_right.clone().detach().requires_grad_(True)

    h_left = model.predict_h(x_left_ref, t_left_ref)
    h_right = model.predict_h(x_right_ref, t_right_ref)

    _, h_x_left, _ = compute_derivatives(h_left, x_left_ref, t_left_ref)
    _, h_x_right, _ = compute_derivatives(h_right, x_right_ref, t_right_ref)

    diff_value = h_left - h_right
    diff_derivative = h_x_left - h_x_right

    mse_value = torch.mean(diff_value.real ** 2 + diff_value.imag ** 2)
    mse_derivative = torch.mean(
        diff_derivative.real ** 2 + diff_derivative.imag ** 2
    )

    return mse_value + mse_derivative


def test_boundary_condition_vectorization_matches_reference(device: str) -> None:
    torch.manual_seed(0)

    model = AnalyticModel().to(device)
    loss_module = SchrodingerLoss().to(device)

    batch_size = 16
    t_vals = torch.linspace(0.0, 1.0, batch_size, device=device).unsqueeze(1)
    x_left = torch.full((batch_size, 1), -5.0, device=device)
    x_right = torch.full((batch_size, 1), 5.0, device=device)

    loss_module.eval()
    model.eval()

    optimized_loss = loss_module.boundary_condition_loss(
        model,
        x_left.clone(),
        t_vals.clone(),
        x_right.clone(),
        t_vals.clone(),
    )

    reference_loss = boundary_loss_reference(model, x_left, t_vals, x_right, t_vals)

    torch.testing.assert_close(
        optimized_loss,
        reference_loss,
        rtol=1e-6,
        atol=1e-8,
    )
    
    print(f"✓ Test passed on {device}!")
    print(f"  Optimized loss: {optimized_loss.item():.10f}")
    print(f"  Reference loss: {reference_loss.item():.10f}")
    print(f"  Abs difference: {abs(optimized_loss.item() - reference_loss.item()):.2e}")


if __name__ == "__main__":
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    
    print("Testing boundary condition vectorization...\n")
    for device in devices:
        print(f"Testing on {device}...")
        test_boundary_condition_vectorization_matches_reference(device)
        print()
    
    print("✓ All tests passed!")

