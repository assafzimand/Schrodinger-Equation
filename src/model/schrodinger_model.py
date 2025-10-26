"""Multi-layer perceptron for Schrödinger equation approximation.

This module implements a feedforward neural network that approximates the
complex-valued solution h(x,t) = u(x,t) + i*v(x,t) of the NLSE.

Architecture (as per PRD):
- Input: (x, t) ∈ R²
- 5 hidden layers × 100 neurons each
- Activation: tanh
- Output: (u, v) ∈ R² (real and imaginary parts)
- Layer names: layer_1, layer_2, layer_3, layer_4, layer_5, output
"""

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn


class SchrodingerNet(nn.Module):
    """Physics-informed neural network for the Schrödinger equation.

    This network approximates h(x,t) = u(x,t) + i*v(x,t), where h satisfies:
        i*h_t + 0.5*h_xx + |h|²*h = 0

    Architecture:
        - Input layer: 2 neurons (x, t)
        - Hidden layers: 5 × 100 neurons with tanh activation
        - Output layer: 2 neurons (u, v)

    Attributes:
        layer_1, layer_2, layer_3, layer_4, layer_5: Hidden layers
        output: Output layer
        activation: Activation function (tanh)
    """

    def __init__(
        self,
        hidden_layers: int = 5,
        hidden_neurons: int = 100,
        activation: str = "tanh",
    ):
        """Initialize the Schrödinger network.

        Args:
            hidden_layers: Number of hidden layers (default: 5)
            hidden_neurons: Number of neurons per hidden layer (default: 100)
            activation: Activation function name (default: "tanh")

        Raises:
            ValueError: If activation function is not supported
        """
        super(SchrodingerNet, self).__init__()

        if hidden_layers != 5:
            raise ValueError(
                f"Model must have exactly 5 hidden layers per PRD, "
                f"got {hidden_layers}"
            )

        # Set activation function
        if activation.lower() == "tanh":
            self.activation = nn.Tanh()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                f"Unsupported activation: {activation}. "
                f"Use 'tanh', 'relu', or 'sigmoid'."
            )

        # Input dimension: (x, t)
        input_dim = 2
        # Output dimension: (u, v)
        output_dim = 2

        # Define layers with explicit names as per PRD
        self.layer_1 = nn.Linear(input_dim, hidden_neurons)
        self.layer_2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.layer_3 = nn.Linear(hidden_neurons, hidden_neurons)
        self.layer_4 = nn.Linear(hidden_neurons, hidden_neurons)
        self.layer_5 = nn.Linear(hidden_neurons, hidden_neurons)
        self.output = nn.Linear(hidden_neurons, output_dim)

        # Initialize weights using Xavier/Glorot initialization
        self._initialize_weights()

        # Storage for forward hooks (for Step 2 introspection)
        # Use custom name to avoid conflict with PyTorch's internal _forward_hooks
        self._custom_hook_handles: Dict[
            str, List[torch.utils.hooks.RemovableHandle]
        ] = {}

    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization.

        This is recommended for tanh activation to prevent gradient issues.
        """
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Spatial coordinates, shape (batch_size, 1)
            t: Temporal coordinates, shape (batch_size, 1)

        Returns:
            Tensor of shape (batch_size, 2) where [:, 0] is u and [:, 1] is v

        Raises:
            ValueError: If inputs are not 2D tensors

        Example:
            >>> model = SchrodingerNet()
            >>> x = torch.randn(100, 1)
            >>> t = torch.randn(100, 1)
            >>> uv = model(x, t)
            >>> u, v = uv[:, 0], uv[:, 1]
        """
        # Ensure inputs are 2D (strict requirement)
        if x.dim() != 2:
            raise ValueError(
                f"Input x must be 2D with shape (batch_size, 1), "
                f"got {x.dim()}D with shape {x.shape}. "
                f"Use x.unsqueeze(1) to convert 1D to 2D."
            )
        if t.dim() != 2:
            raise ValueError(
                f"Input t must be 2D with shape (batch_size, 1), "
                f"got {t.dim()}D with shape {t.shape}. "
                f"Use t.unsqueeze(1) to convert 1D to 2D."
            )

        # Concatenate inputs: (x, t)
        inputs = torch.cat([x, t], dim=1)

        # Forward through hidden layers
        h = self.activation(self.layer_1(inputs))
        h = self.activation(self.layer_2(h))
        h = self.activation(self.layer_3(h))
        h = self.activation(self.layer_4(h))
        h = self.activation(self.layer_5(h))

        # Output layer (no activation)
        uv = self.output(h)

        return uv

    def predict_h(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predict complex field h = u + i*v.

        Args:
            x: Spatial coordinates
            t: Temporal coordinates

        Returns:
            Complex tensor h(x,t) = u(x,t) + i*v(x,t)
        """
        uv = self.forward(x, t)
        u = uv[:, 0]
        v = uv[:, 1]
        h = torch.complex(u, v)
        return h

    def register_forward_hook_by_name(
        self, layer_name: str, hook: Callable
    ) -> torch.utils.hooks.RemovableHandle:
        """Register a forward hook on a specific layer by name.

        This enables introspection of intermediate activations for Step 2.

        Args:
            layer_name: Name of the layer ('layer_1', ..., 'layer_5', 'output')
            hook: Hook function with signature:
                  hook(module, input, output) -> None or modified output

        Returns:
            Handle that can be used to remove the hook

        Raises:
            ValueError: If layer_name is not valid

        Example:
            >>> model = SchrodingerNet()
            >>> activations = {}
            >>> def capture_activation(name):
            ...     def hook(module, input, output):
            ...         activations[name] = output.detach()
            ...     return hook
            >>> handle = model.register_forward_hook_by_name(
            ...     'layer_3', capture_activation('layer_3')
            ... )
            >>> # Now activations['layer_3'] will be populated during forward pass
            >>> handle.remove()  # Remove hook when done
        """
        valid_layers = [
            "layer_1",
            "layer_2",
            "layer_3",
            "layer_4",
            "layer_5",
            "output",
        ]

        if layer_name not in valid_layers:
            raise ValueError(
                f"Invalid layer name: {layer_name}. "
                f"Valid names: {valid_layers}"
            )

        layer = getattr(self, layer_name)
        handle = layer.register_forward_hook(hook)

        # Store handle for later removal
        if layer_name not in self._custom_hook_handles:
            self._custom_hook_handles[layer_name] = []
        self._custom_hook_handles[layer_name].append(handle)

        return handle

    def remove_all_hooks(self):
        """Remove all registered forward hooks."""
        for layer_name, handles in self._custom_hook_handles.items():
            for handle in handles:
                handle.remove()
        self._custom_hook_handles.clear()

    def count_parameters(self) -> int:
        """Count total number of trainable parameters.

        Returns:
            Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_names(self) -> List[str]:
        """Get list of layer names in order.

        Returns:
            List of layer names
        """
        return ["layer_1", "layer_2", "layer_3", "layer_4", "layer_5", "output"]

    def summary(self) -> str:
        """Generate a summary string of the network architecture.

        Returns:
            Multi-line string describing the network
        """
        lines = []
        lines.append("=" * 70)
        lines.append("SchrodingerNet Architecture")
        lines.append("=" * 70)
        lines.append(f"Input dimension:  2 (x, t)")
        lines.append(f"Output dimension: 2 (u, v)")
        lines.append(f"Activation:       {self.activation.__class__.__name__}")
        lines.append("")
        lines.append("Layers:")

        for name, module in self.named_children():
            if isinstance(module, nn.Linear):
                lines.append(
                    f"  {name:12s}: Linear({module.in_features:4d} → "
                    f"{module.out_features:4d})"
                )

        lines.append("")
        lines.append(f"Total parameters: {self.count_parameters():,}")
        lines.append("=" * 70)

        return "\n".join(lines)


def create_model(
    hidden_layers: int = 5,
    hidden_neurons: int = 100,
    activation: str = "tanh",
    device: Optional[str] = None,
) -> SchrodingerNet:
    """Factory function to create and initialize a SchrodingerNet.

    Args:
        hidden_layers: Number of hidden layers (must be 5)
        hidden_neurons: Neurons per hidden layer
        activation: Activation function name
        device: Device to place model on ('cuda', 'cpu', or None for auto)

    Returns:
        Initialized SchrodingerNet on specified device
    """
    model = SchrodingerNet(
        hidden_layers=hidden_layers,
        hidden_neurons=hidden_neurons,
        activation=activation,
    )

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    return model


if __name__ == "__main__":
    """Test the model with random inputs."""
    print("Testing SchrodingerNet...")

    # Create model
    model = create_model()
    print("\n" + model.summary())

    # Test with random batch
    batch_size = 100
    x = torch.randn(batch_size, 1)
    t = torch.randn(batch_size, 1)

    print(f"\nTest forward pass:")
    print(f"  Input shapes: x={x.shape}, t={t.shape}")

    # Forward pass
    uv = model(x, t)
    print(f"  Output shape: {uv.shape}")
    print(f"  Output range: u ∈ [{uv[:, 0].min():.3f}, {uv[:, 0].max():.3f}], "
          f"v ∈ [{uv[:, 1].min():.3f}, {uv[:, 1].max():.3f}]")

    # Test complex prediction
    h = model.predict_h(x, t)
    print(f"  Complex h shape: {h.shape}")
    print(f"  |h| range: [{h.abs().min():.3f}, {h.abs().max():.3f}]")

    # Test hook registration
    print(f"\nTest forward hook registration:")
    activations = {}

    def capture_activation(name):
        def hook(module, input, output):
            activations[name] = output.detach().clone()
        return hook

    # Register hooks on all layers
    handles = []
    for layer_name in model.get_layer_names():
        handle = model.register_forward_hook_by_name(
            layer_name, capture_activation(layer_name)
        )
        handles.append(handle)

    # Run forward pass
    _ = model(x, t)

    # Check captured activations
    print(f"  Captured activations from {len(activations)} layers:")
    for layer_name, activation in activations.items():
        print(f"    {layer_name:12s}: shape={activation.shape}")

    # Clean up hooks
    model.remove_all_hooks()
    print(f"  ✓ All hooks removed")

    print("\n" + "=" * 70)
    print("✓ Model test completed successfully!")
    print("=" * 70)

