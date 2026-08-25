import mlstac
import torch
import logging
import tqdm
import rasterio
import sen2sr
import numpy as np
from tqdm import tqdm
from pathlib import Path
from affine import Affine

from .resolver_config import ResolverConfig

logging.basicConfig(level = logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _get_latest_weights(weights_list: list[Path]) -> Path | None:
    """
    Gets the latest weights saved based on OS modification time.
    """
    if weights_list:
        latest_weights_path = max(weights_list, key = lambda p: p.stat().st_mtime)
        logger.info(f"Custom weights found. Loading most recent checkpoint: {latest_weights_path}")
        return latest_weights_path
    return None


def get_model(config: ResolverConfig, device: torch.device) -> torch.nn.Module:
    """
    Handles the download and initialization of the model.
    """
    config.model_path.mkdir(parents = True, exist_ok = True)

    logger.debug(f"Checking if model is already downloaded in {config.model_path}")
    if not any(config.model_path.iterdir()):
        logger.info("Model not found. Downloading from Hugging Face...")
        mlstac.download(
            file = "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json",
            output_dir = config.model_path
        )

    logger.info("Loading model to GPU")
    model = mlstac.load(str(config.model_path)).compiled_model(device = device)
    
    logger.debug(f"Checking if weights already exist in {config.weights_path}.")
    weights = _get_latest_weights(list(config.weights_path.glob("*.pth")))

    if weights:
        state_dict = torch.load(weights, map_location = device, weights_only = True)
        model.load_state_dict(state_dict)
    else:
        logger.info(f"Custom weights not found. Proceeding with base pre-trained weights.")

    model = model.to(device)
    # Sets the model to the eval mode
    model.eval()

    return model


def _calc_transform(sr_tensor: torch.Tensor, base_transform, base_scale: float):
    scale = sr_tensor.shape[1] / base_scale
    output_transform = base_transform * Affine.scale(1 / scale, 1 / scale)
    return output_transform


@torch.no_grad()
def _process_image(file_path: Path, output_path: Path, model: torch.nn.Module, device = torch.device) -> None:
    """
    Reads a 10-band Low-Res Tif, Super-Resolves it and saves the new Super-Resolved image.
    """
    with rasterio.open(file_path) as src:
        img = src.read()
        metadata = src.profile.copy()
        base_transform = src.transform
        base_crs = src.crs

    if img.shape[1] % 128 != 0 or img.shape[2] % 128 != 0:
        logger.error(f"Image dimensions are not divisible by 128. It's impossible to resolve this image.")
        return

    img = np.clip(img, 0.0, 1.0)
    img = torch.from_numpy(img).float().to(device)

    sr_tensor = sen2sr.predict_large(img, model, overlap = 16)
    output_transform = _calc_transform(sr_tensor, base_transform, img.shape[1])

    sr_img = sr_tensor.cpu().numpy()

    sr_channels, sr_height, sr_width = sr_img.shape

    metadata.update({
        "height": sr_height,
        "width": sr_width,
        "crs": base_crs,
        "transform": output_transform,
        "dtype": "float32",
        "compress": "lzw",
        "driver": "GTiff"
    })

    with rasterio.open(output_path, "w", **metadata) as dst:
        dst.write(sr_img)


def main() -> None:
    config = ResolverConfig()

    config.resolved_files_path.mkdir(parents = True, exist_ok = True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Sen2SR requires a GPU.")

    try:
        tif_files = sorted(config.ingested_tif_path.glob("*.tif"))
        if not tif_files:
            logger.warning(f"No Tif files found in {config.ingested_tif_path}.")
        logger.info(f"Found {len(tif_files)} Tif files to super-resolve.")

        model = get_model(config, device)
        
        for file in tqdm(tif_files, desc = "Super-Resolving files"):
            output_file_path = config.resolved_files_path / file.name
            _process_image(file, output_file_path, model, device)
            
        logger.info("=== Pipeline for Super-Resolving Finished ===")

    except Exception as e:
        print(e)
        # logger.error(f"Resolving crashed: \n {e}")


if __name__ == "__main__":
    main()