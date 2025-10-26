"""Test script for SchrodingerNet model."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.schrodinger_model import SchrodingerNet, create_model


def test_model_architecture():
    """Test model architecture and parameter count."""
    print("=" * 70)
    print("Testing SchrodingerNet Model")
    print("=" * 70)

    # Create model
    model = create_model()

    print("\n1. Architecture:")
    print(model.summary())

    # Verify parameter count
    expected_params = (
        2 * 100  # layer_1: 2 -> 100
        + 100  # layer_1 bias
        + 100 * 100  # layer_2: 100 -> 100
        + 100  # layer_2 bias
        + 100 * 100  # layer_3
        + 100  # layer_3 bias
        + 100 * 100  # layer_4
        + 100  # layer_4 bias
        + 100 * 100  # layer_5
        + 100  # layer_5 bias
        + 100 * 2  # output: 100 -> 2
        + 2  # output bias
    )

    actual_params = model.count_parameters()
    print(f"\n   Expected parameters: {expected_params:,}")
    print(f"   Actual parameters:   {actual_params:,}")
    assert actual_params == expected_params, "Parameter count mismatch!"
    print("   ✓ Parameter count correct")


def test_forward_pass():
    """Test forward pass with random inputs."""
    print("\n2. Forward Pass:")

    model = SchrodingerNet()

    # Test with various batch sizes
    batch_sizes = [1, 10, 100, 1000]

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 1)
        t = torch.randn(batch_size, 1)

        uv = model(x, t)

        assert uv.shape == (
            batch_size,
            2,
        ), f"Output shape mismatch for batch size {batch_size}"

    print(f"   ✓ Tested batch sizes: {batch_sizes}")

    # Test that 1D inputs are REJECTED (strict mode)
    x_1d = torch.randn(50)
    t_1d = torch.randn(50)
    
    try:
        uv = model(x_1d, t_1d)
        assert False, "Should have raised ValueError for 1D input!"
    except ValueError as e:
        assert "must be 2D" in str(e), f"Wrong error message: {e}"
        print("   ✓ 1D inputs correctly rejected with ValueError")
    
    # Verify the error message is helpful
    try:
        model(x_1d, t_1d)
    except ValueError as e:
        assert "unsqueeze" in str(e), "Error should suggest using unsqueeze"
        print(f"   ✓ Error message is helpful: {str(e)[:80]}...")


def test_complex_prediction():
    """Test complex-valued prediction."""
    print("\n3. Complex Prediction:")

    model = SchrodingerNet()

    x = torch.tensor([[0.0], [1.0], [-1.0]])
    t = torch.tensor([[0.0], [0.5], [1.0]])

    h = model.predict_h(x, t)

    assert h.dtype == torch.complex64, "Output should be complex"
    assert h.shape == (3,), f"Complex output shape mismatch: {h.shape}"

    print(f"   ✓ Complex output shape: {h.shape}")
    print(f"   ✓ Complex dtype: {h.dtype}")
    print(f"   Sample values:")
    for i, (xi, ti, hi) in enumerate(zip(x, t, h)):
        print(f"     h({xi.item():.1f}, {ti.item():.1f}) = {hi.item()}")


def test_hook_registration():
    """Test forward hook registration and removal."""
    print("\n4. Forward Hook Registration:")

    model = SchrodingerNet()

    # Dictionary to store activations
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach().clone()

        return hook

    # Register hooks
    handles = {}
    for layer_name in model.get_layer_names():
        handle = model.register_forward_hook_by_name(
            layer_name, make_hook(layer_name)
        )
        handles[layer_name] = handle

    print(f"   Registered hooks on {len(handles)} layers")

    # Run forward pass
    x = torch.randn(20, 1)
    t = torch.randn(20, 1)
    _ = model(x, t)

    # Verify activations were captured
    assert len(activations) == 6, "Not all activations captured"

    for layer_name, activation in activations.items():
        expected_shape = (20, 100) if layer_name != "output" else (20, 2)
        assert (
            activation.shape == expected_shape
        ), f"Wrong shape for {layer_name}"

    print(f"   ✓ Captured activations:")
    for layer_name in model.get_layer_names():
        print(f"     {layer_name:12s}: {activations[layer_name].shape}")

    # Remove hooks
    model.remove_all_hooks()
    print("   ✓ Hooks removed successfully")

    # Verify hooks are removed (activations should not update)
    activations.clear()
    _ = model(x, t)
    assert len(activations) == 0, "Hooks were not properly removed"
    print("   ✓ Verified hooks removal")


def test_layer_names():
    """Test layer naming convention."""
    print("\n5. Layer Naming:")

    model = SchrodingerNet()
    layer_names = model.get_layer_names()

    expected_names = [
        "layer_1",
        "layer_2",
        "layer_3",
        "layer_4",
        "layer_5",
        "output",
    ]

    assert (
        layer_names == expected_names
    ), f"Layer names don't match PRD spec: {layer_names}"

    print(f"   ✓ Layer names: {', '.join(layer_names)}")

    # Verify all layers exist
    for name in expected_names:
        assert hasattr(model, name), f"Layer {name} not found"

    print("   ✓ All required layers exist")


def test_device_handling():
    """Test device handling (CPU/CUDA)."""
    print("\n6. Device Handling:")

    # Test CPU
    model_cpu = create_model(device="cpu")
    print(f"   ✓ Model created on CPU")

    x = torch.randn(10, 1)
    t = torch.randn(10, 1)
    uv = model_cpu(x, t)
    assert uv.device.type == "cpu", "Output not on CPU"
    print(f"   ✓ Forward pass works on CPU")

    # Test CUDA if available
    if torch.cuda.is_available():
        model_cuda = create_model(device="cuda")
        print(f"   ✓ Model created on CUDA")

        x_cuda = x.cuda()
        t_cuda = t.cuda()
        uv_cuda = model_cuda(x_cuda, t_cuda)
        assert uv_cuda.device.type == "cuda", "Output not on CUDA"
        print(f"   ✓ Forward pass works on CUDA")
    else:
        print("   ⚠ CUDA not available, skipping CUDA tests")


if __name__ == "__main__":
    test_model_architecture()
    test_forward_pass()
    test_complex_prediction()
    test_hook_registration()
    test_layer_names()
    test_device_handling()

    print("\n" + "=" * 70)
    print("✓ All model tests passed successfully!")
    print("=" * 70)

