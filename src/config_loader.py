"""Configuration loader with argparse + YAML + dataclass validation.

This module provides utilities to load configuration from YAML files,
validate them using dataclasses, and support environment variable overrides.
"""

import argparse
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml


T = TypeVar("T")


@dataclass
class SolverConfig:
    """Configuration for the NLSE solver.
    
    Attributes:
        x_min: Minimum spatial coordinate
        x_max: Maximum spatial coordinate
        t_min: Minimum time coordinate
        t_max: Maximum time coordinate
        nx: Number of spatial grid points
        nt: Number of temporal grid points
        alpha: Nonlinearity coefficient (default 1.0 for standard NLSE)
    """
    x_min: float = -5.0
    x_max: float = 5.0
    t_min: float = 0.0
    t_max: float = 1.5708  # π/2
    nx: int = 256
    nt: int = 100
    alpha: float = 1.0


@dataclass
class DatasetConfig:
    """Configuration for dataset generation.
    
    Attributes:
        n_collocation: Number of collocation points (N_f)
        n_initial: Number of initial condition points (N_0)
        n_boundary: Number of boundary condition points (N_b)
        sampling_method: Sampling method ('lhs' for Latin Hypercube)
        seed: Random seed for reproducibility
        save_path: Path to save generated dataset
    """
    n_collocation: int = 20000
    n_initial: int = 50
    n_boundary: int = 50
    sampling_method: str = "lhs"
    seed: int = 42
    save_path: str = "data/processed/dataset.npz"


@dataclass
class TrainConfig:
    """Configuration for model training.
    
    Attributes:
        epochs: Number of training epochs
        learning_rate: Initial learning rate
        batch_size: Batch size for training
        device: Device to use ('cuda', 'cpu', or 'auto')
        hidden_layers: Number of hidden layers
        hidden_neurons: Number of neurons per hidden layer
        activation: Activation function name
        optimizer: Optimizer name ('adam', 'lbfgs', etc.)
        scheduler: Learning rate scheduler type (optional)
        seed: Random seed for reproducibility
        dtype: Data type precision ('float32' or 'float64')
        deterministic: Whether to use deterministic CuDNN
        mlflow_experiment: Name for MLflow experiment
        checkpoint_dir: Directory to save model checkpoints
    """
    epochs: int = 10000
    learning_rate: float = 1e-3
    batch_size: int = 1024
    device: str = "auto"
    hidden_layers: int = 5
    hidden_neurons: int = 100
    activation: str = "tanh"
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    seed: int = 42
    dtype: str = "float32"
    deterministic: bool = False
    mlflow_experiment: str = "schrodinger_step1"
    checkpoint_dir: str = "outputs/checkpoints"


@dataclass
class Config:
    """Root configuration containing all sub-configs.
    
    Attributes:
        solver: Solver configuration
        dataset: Dataset configuration
        train: Training configuration
    """
    solver: SolverConfig = field(default_factory=SolverConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file.
    
    Args:
        path: Path to the YAML file
        
    Returns:
        Dictionary containing the YAML contents
        
    Raises:
        FileNotFoundError: If the file does not exist
        yaml.YAMLError: If the file is not valid YAML
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def apply_env_overrides(config_dict: Dict[str, Any], prefix: str = "SCHRODINGER_") -> Dict[str, Any]:
    """Apply environment variable overrides to configuration.
    
    Environment variables should be named: PREFIX_SECTION_KEY (uppercase)
    For example: SCHRODINGER_TRAIN_LEARNING_RATE
    
    Args:
        config_dict: Configuration dictionary
        prefix: Prefix for environment variables
        
    Returns:
        Updated configuration dictionary
    """
    for section, values in config_dict.items():
        if isinstance(values, dict):
            for key in values:
                env_var = f"{prefix}{section.upper()}_{key.upper()}"
                if env_var in os.environ:
                    # Try to infer type from existing value
                    old_value = values[key]
                    new_value = os.environ[env_var]
                    
                    if isinstance(old_value, bool):
                        values[key] = new_value.lower() in ("true", "1", "yes")
                    elif isinstance(old_value, int):
                        values[key] = int(new_value)
                    elif isinstance(old_value, float):
                        values[key] = float(new_value)
                    else:
                        values[key] = new_value
    
    return config_dict


def dict_to_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Convert a dictionary to a dataclass instance.
    
    Args:
        cls: Dataclass type
        data: Dictionary with configuration values
        
    Returns:
        Instance of the dataclass
    """
    field_types = {f.name: f.type for f in fields(cls)}
    kwargs = {}
    
    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            
            # Handle nested dataclasses
            if hasattr(field_type, "__dataclass_fields__"):
                kwargs[key] = dict_to_dataclass(field_type, value)
            else:
                kwargs[key] = value
    
    return cls(**kwargs)


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file with validation.
    
    Args:
        config_path: Path to configuration file. If None, uses defaults.
        
    Returns:
        Validated Config instance
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If configuration is invalid
    """
    if config_path is None:
        # Return default configuration
        return Config()
    
    config_file = Path(config_path)
    
    # Load YAML
    config_dict = load_yaml(config_file)
    
    # Apply environment variable overrides
    config_dict = apply_env_overrides(config_dict)
    
    # Convert to dataclass with validation
    config = dict_to_dataclass(Config, config_dict)
    
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Deep Learning Approximation of the Schrödinger Equation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cuda/cpu)",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed override",
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs override",
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate override",
    )
    
    return parser.parse_args()


def load_config_from_args(args: Optional[argparse.Namespace] = None) -> Config:
    """Load configuration from command-line arguments.
    
    Args:
        args: Parsed arguments. If None, parses from sys.argv
        
    Returns:
        Validated Config instance with CLI overrides applied
    """
    if args is None:
        args = parse_args()
    
    # Load base config from file
    config = load_config(args.config)
    
    # Apply CLI overrides
    if args.device is not None:
        config.train.device = args.device
    if args.seed is not None:
        config.train.seed = args.seed
        config.dataset.seed = args.seed
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.learning_rate is not None:
        config.train.learning_rate = args.learning_rate
    
    return config


def save_config(config: Config, output_path: Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration instance
        output_path: Path to save YAML file
    """
    def dataclass_to_dict(obj: Any) -> Any:
        """Recursively convert dataclass to dictionary."""
        if hasattr(obj, "__dataclass_fields__"):
            return {
                key: dataclass_to_dict(getattr(obj, key))
                for key in obj.__dataclass_fields__
            }
        return obj
    
    config_dict = dataclass_to_dict(config)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # Test config loading
    config = load_config_from_args()
    print("Configuration loaded successfully!")
    print(f"\nSolver Config:\n{config.solver}")
    print(f"\nDataset Config:\n{config.dataset}")
    print(f"\nTrain Config:\n{config.train}")

