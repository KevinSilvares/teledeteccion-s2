import logging
import random
import rioxarray as riox
import xarray as xr
from pathlib import Path


logger = logging.getLogger(__name__)


def _create_output_paths(output_tif_path: str | Path, output_mask_path: str | Path) -> tuple[Path, Path]:
    """
    Creates the output directories for both .tif files and mask files. Return the Path object of the provided path.
    """
    logger.debug("Creating output paths.")
    output_tif_path = Path(output_tif_path)
    output_mask_path = Path(output_mask_path)

    output_tif_path.mkdir(parents = True, exist_ok = True)
    output_mask_path.mkdir(parents = True, exist_ok = True)

    return output_tif_path, output_mask_path


def _load_rasters(tif_path: str | Path, mask_path: str | Path) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Loads both .tif and mask rasters in memory and ensures spatial dimensions match.
    """
    logger.debug("Loading rasters in memory.")
    tif_raster = riox.open_rasterio(Path(tif_path))
    mask_raster = riox.open_rasterio(Path(mask_path))

    if tif_raster.rio.height != mask_raster.rio.height or tif_raster.rio.width != mask_raster.rio.width:
        raise RuntimeError(f"Shape mismatch between Tif raster and mask raster: {tif_path} | {mask_path}.")
    
    return tif_raster, mask_raster


def adapt_rasters(
    tif_path: str | Path,
    mask_path: str | Path,
    output_tif_path: str | Path,
    output_mask_path: str | Path,
    patch_size: int = 512,
    overlap: float = 0.0,
    keep_emtpy_prob: float = 0.1
    ) -> None:
    """
    Tiles a large Sentinel-2 image and its corresponding mask into fixed size patches.
    Retains a percentage of empty mask to prevent model False Positives.
    """
    logger.info(f"Adapting {tif_path} | {mask_path}.")

    output_tif_path, output_mask_path = _create_output_paths(output_tif_path, output_mask_path)

    tif_raster, mask_raster = _load_rasters(tif_path, mask_path)

    # Everything should be alright at this point since it passed the shape mismatch check before
    height, width = tif_rater.rio.height, tif_rater.rio.width
    # Calculates the step for the traversing on the rasters
    step = int(patch_size * (1 - overlap))

    i = 0
    # NOTE: Might want to take a look at this if I ever change the patch_size
    for y in range(0, height - patch_size + 1, step):
        for x in range(0, width - patch_size + 1, step):
            tif_patch = tif_raster.isel(
                x = slice(x, x + patch_size),
                y = slice(y, y + patch_size)
            )

            mask_patch = mask_raster.isel(
                x = slice(x, x + patch_size),
                y = slice(y, y + patch_size)
            )

            # Checks if the mask patch has any data and trashes the no data ones
            if (mask_patch > 0).any() and random.random() < keep_emtpy_prob:
                i += 1

                tif_name = f"patch_{i:04d}.tif"
                mask_name = f"patch_{i:04d}_mask.png"

                output_tif = output_tif_path / tif_name
                output_mask = output_mask_path / mask_name

                tif_patch.rio.to_raster(output_tif, compress = "lzw")

                # Saves the mask as uint8 to save memory and computation power
                mask_patch = mask_patch.astype("uint8")
                mask_patch.rio.to_raster(output_mask, driver = "PNG")

                logger.info(f"{i:04d} Pair saved. {output_tif} | {output_mask}")
    
    logger.info(f"=== Raster and Mask adaptation finished. Total: {i} files processed and saved. ===")


