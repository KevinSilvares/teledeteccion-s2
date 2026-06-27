import mlstac
import torch
# import time
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast


if __name__ == "__main__":
    # Settings
    # Hyperparams
    BATCH_SIZE = 4
    EPOCHS = 50
    LEARNING_RATE = 1e-6
    
    # Data Paths
    LR_PATH = Path(r"data/lr")
    HR_PATH = Path(r"data/hr")

    # Resolve paths
    script_dir = Path(__file__).resolve()
    lr_path = script_dir.parent.parent / LR_PATH
    lr_val_path = script_dir.parent.parent / LR_PATH / "val"
    hr_path = script_dir.parent.parent / HR_PATH
    hr_val_path = script_dir.parent.parent / HR_PATH / "val"

    model_path = script_dir.parent.parent / "model" / "sen2sr-lite"
    weights_path = script_dir.parent.parent / "model" / "custom_weights"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "Sen2SR only runs in GPU."

