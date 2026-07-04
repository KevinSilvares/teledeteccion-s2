import logging
import torch

from transformers import TrainingArguments, Trainer
from pathlib import Path
from dataset import SegFormerDataset

from training_config import TrainConfig
from model import Model
from loss import IOU


logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def get_model() -> torch.nn.Module:
    """
    Initializes a custom model wrapper and returns and adapted 4-channel SegFormer to process R, G, B, NIR Tif files.
    """
    logging.info("Initializing Model wrapper.")
    return Model().load_base_model()


def main() -> None:
    logger.info("=== Starting SegFormer Fine-Tuning Pipeline ===")
    config = TrainConfig()

    logger.info("Loading Training and Validation Datasets.")
    logger.info(str(config.tif_train_path) + str(config.mask_train_path))
    dataset_train = SegFormerDataset(config.tif_train_path, config.mask_train_path, True)
    dataset_val = SegFormerDataset(config.tif_val_path, config.mask_val_path, False)
    logger.debug(f"Loaded {len(dataset_train)} train patches and {len(dataset_val)} validation patches.")

    logger.info("Loading evaluation metric (mIoU).")
    metric = IOU(num_labels = 2)

    model = get_model()

    logger.info("Configuring TrainingArguments.")
    args_train = TrainingArguments(
        learning_rate = config.learning_rate,
        num_train_epochs = config.epochs,
        per_device_train_batch_size = config.batch_size,
        per_device_eval_batch_size = config.batch_size,
        eval_strategy = "epoch",
        save_strategy = "epoch",
        save_total_limit = 3,
        output_dir = str(config.weights_path / "checkpoints"),
        load_best_model_at_end = True,
        remove_unused_columns = False,
        greater_is_better = True,
        dataloader_num_workers = 4,
        logging_steps = 10,
        fp16 = torch.cuda.is_available()
    )

    logger.info("Initializing Hugging-Face Trainer.")
    train = Trainer(
        model = model,
        args = args_train,
        train_dataset = dataset_train,
        eval_dataset = dataset_val,
        compute_metrics = metric
    )

    logger.info("=== Starting Training ===")
    train.train()

    logger.info(f"Saving best model in: {config.best_model}")
    train.save_model(str(config.best_model))


if __name__ == "__main__":
    main()