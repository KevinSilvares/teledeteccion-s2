import logging
import rasterio
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path


logger = logging.getLogger(__name__)


def adapt_masks(tif_masks_dir: Path | str, output_dir: Path | str) -> None:
    """
    Converts .tif files created by QGIS into PNG for the SegFormer training loop.
    """
    tif_masks_dir = Path(tif_masks_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents = True, exist_ok = True)

    tif_masks = sorted(tif_masks_dir.glob("*.tif"))

    if not tif_masks:
        logger.warning(f"No .tif masks found in {tif_masks_dir}.")
        return

    logger.info(f"Found {len(tif_masks)} elements. Starting conversion.")

    for tif in tqdm(tif_masks, desc = "Converting Tif Masks to PNG"):
        file_name = f"{tif.stem}.png"
        output_path = output_dir / file_name

        with rasterio.open(tif) as src:
            # Data is only in the first band
            mask = src.read(1)
            width = src.width
            height = src.height

            # It should always be a byte (uint8), so this is a double check
            mask = mask.astype(np.uint8)

            # Mode = L because I need a black and white iamge (1 channel)
            img = Image.fromarray(mask, mode = "L")
            img.save(output_path)

            logger.info(f"Converted {file_name} to PNG and saved in {output_path}.")
    
    logger.info("=== Pipeline for Converting Tif Mask to PNG Completed ===")


if __name__ == "__main__":
    # NOTE: I'll keep this simple for now. Take a look at cleaner coding later.
    base_dir = Path(__file__).resolve().parent.parent
    tif_masks_dir = base_dir / "data" / "mask_adaptation"
    output_dir = base_dir / "data" / "output_masks_png"
    
    adapt_masks(tif_masks_dir, output_dir)
