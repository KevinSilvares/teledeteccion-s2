@dataclass
class TrainConfig:
    """
    Centralized configuration for hyperparameters and paths for the Sen2SR training phase.
    """
    # Hyperparameters
    batch_size: int = 4
    epochs: int = 50
    learning_rate: float = 1e-6

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    lr_train_path: Path = base_dir / "data" / "lr"
    lr_val_path: Path = base_dir / "data" / "lr_val"
    hr_train_path: Path = base_dir / "data" / "hr"
    hr_val_path: Path = base_dir / "data" / "hr_val"

    model_path: Path = base_dir / "model" / "sen2sr-lite"
    weights_path: Path = base_dir / "model" / "custom_weights"
