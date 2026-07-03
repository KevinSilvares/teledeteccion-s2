from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainConfig:
    """
    Centralized configuration for hyperparameters and paths for the SegFormer training phase.
    """
    # Hyperparameters
    batch_size: int = 2
    epochs: int = 50
    learning_rate: float = 1e-6

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    tif_train_path: Path = base_dir / "data" / "train" / "imgs_tif"
    tif_val_path: Path = base_dir / "data" / "val" / "imgs_tif"
    mask_train_path: Path = base_dir / "data" / "train" / "mask"
    mask_val_path: Path = base_dir / "data" / "val" / "mask"

    output_path: Path = base_dir / "data" / "output"

    best_model: Path = base_dir / "model" / "best_model"
    weights_path: Path = base_dir / "model" / "best_weights"
