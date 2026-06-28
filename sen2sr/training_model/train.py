import mlstac
import torch
import uuid
import logging
from pathlib import Path
from torch.utils.data import DataLoader

from train_config import TrainConfig
from sr_dataset import SRDataset
from loss import MaskedL1Loss


logging.basicConfig(level = logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_model(config: TrainConfig, device: torch.device) -> torch.nn.Module:
    """
    Handles the download and initialization of the model.
    """
    config.model_path.mkdir(parents = True, exist_ok = True)

    if not any(config.model_path.iterdir()):
        logger.info("Model not found. Downloading from Hugging Face...")
        mlstac.download(
            file = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json",
            output_dir = config.model_path
        )

    logger.info("Loading model to GPU")
    model = mlstac.load(str(config.model_path)).compiled_model(device = device)
    model = model.to(device)

    # Unfreezes model weights
    for param in model.parameters():
        param.requires_grad = True

    return model


def train_epoch(
    model: torch.nn.Module,
    dataloder: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
    ) -> float:
    """
    Executes one full training epoch and returns the average loss.
    """
    model.train()
    total_loss = 0.0

    for batch_lr, batch_hr in dataloader:
        batch_lr, batch_hr = batch_lr.to(device), batch_hr.to(device)

        # Resets the optimizers grads to avoid destroying the original model weights
        optimizer.zero_grad(set_to_none = True)

        # Forward pass
        pred = model(batch_lr)
        # Loss calculation
        loss = criterion(pred, batch_hr)
        # Backward pass
        loss.backward()
        # Weights update
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(datalaoder)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloder: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device
    ) -> float:
    """
    Executes validation and returns the average loss.
    """
    model.eval()
    total_loss = 0.0

    for batch_lr_val, batch_hr_val in dataloader:
        batch_lr_val, batch_hr_val = batch_lr_val.to(device), batch_hr_val.to(device)

        pred_val = model(batch_lr_val)
        loss_val = criterion(pred_val, batch_hr_val)

        total_loss += loss_val.item()
    return total_loss / len(datalaoder)


def main() -> None:
    # Unique id to avoid overwriting existing weights
    unique_id = uuid.uuid4().hex[:8]
    config = TrainConfig()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Sen2SR requires a GPU.")
    
    try:
        # Datasets and Dataloaders init
        dataset_train = SRDataset(config.lr_train_path, config.hr_train_path)
        dataset_val = SRDataset(config.lr_val_path, config.hr_val_path)

        dataloader_train = DataLoader(dataset = dataset_train, batch_size = config.batch_size, shuffle = True)
        dataloader_val = DataLoader(dataset = dataset_val, batch_size = config.batch_size, shuffle = False)

        logger.info(f"Found {len(dataset_train)} pairs for training.")
        logger.info(f"Found {len(dataset_val)} pairs for validation.")

        # Model, Loss and Optimizer init
        model = get_model(config, device)
        criterion = MaskedL1Loss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate)

        # Training Loop
        best_val_loss = float("inf")
        logger.info("=== Starting Training Sen2SR ===")
        for epoch in range(config.epochs):
            train_loss = train_epoch(model, dataloader_train, criterion, optimizer, device)
            val_loss = validate(model, dataloader_val, criterion, device)

            logger.info(f"Epoch {epoch + 1:02d}/{config.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            # Save best weights if the new val loss is better than the current val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                config.weights_path.mkdir(parents = True, exist_ok = True)
                full_path = config.weights_path / f"custom_weights_best_{unique_id}.pth"

                torch.save(model.state_dict(), full_path)
                logger.info(f"New best model saved.")
    except Exception as e:
        logger.error(f"Training crashed:\n {e}")
        raise


if __name__ == "__main__":
    main()