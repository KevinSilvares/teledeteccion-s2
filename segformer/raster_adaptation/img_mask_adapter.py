import logging
import random
import rioxarray as riox
import xarray as xr
from tqdm import tqdm
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

    tif_h, tif_w = tif_raster.rio.height, tif_raster.rio.width
    mask_h, mask_w = mask_raster.rio.height, mask_raster.rio.width

    # There is a weird bug with gdal where it adds one more pixel than it should. I'm getting a error for shape mismatch for just one pixel.
    # Ex: Tif = 2048x2048 | Mask: 2049x2049. Since the starting coord is the same, I can just crop this extra pixel without losing anything.
    if tif_h != mask_h or tif_w != mask_w:
        logger.warning(f"Shape mismatch: Tif: {tif_w}x{tif_h} | Mask: {mask_w}x{mask_h}. Cropping mask to fit.")

        mask_raster = mask_raster.isel(
            x = slice(0, tif_w),
            y = slice(0, tif_h)
        )

    # If the last cropping didn't work it means something went horribly wrong.
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
    Tiles a large Super-Resolved image and its corresponding mask into fixed size patches.
    Retains a percentage of empty mask to prevent model False Positives.
    """
    logger.info(f"Adapting {tif_path} | {mask_path}.")

    output_tif_path, output_mask_path = _create_output_paths(output_tif_path, output_mask_path)

    tif_raster, mask_raster = _load_rasters(tif_path, mask_path)

    # Everything should be alright at this point since it passed the shape mismatch check before
    height, width = tif_raster.rio.height, tif_raster.rio.width
    # Calculates the step for the traversing on the rasters
    step = int(patch_size * (1 - overlap))

    unique_id = tif_path.stem.replace("S2_patch_", "")
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
            if ((mask_patch > 0).any()) or (random.random() < keep_emtpy_prob):
                i += 1

                tif_name = f"patch_{unique_id}_{i:04d}.tif"
                mask_name = f"patch_{unique_id}_{i:04d}_mask.png"

                output_tif = output_tif_path / tif_name
                output_mask = output_mask_path / mask_name

                tif_patch.rio.to_raster(output_tif, compress = "lzw")

                # Saves the mask as uint8 to save memory and computation power
                mask_patch = mask_patch.astype("uint8")
                mask_patch.rio.to_raster(output_mask, driver = "PNG")

                logger.info(f"{i:04d} Pair saved. {output_tif} | {output_mask}")
    
    logger.info(f"=== Raster and Mask adaptation finished. Total: {i} files processed and saved. ===")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    tif_dir = base_dir / "data" / "tif_files"
    mask_dir = base_dir / "data" / "output_masks_png"

    # NOTE: I'll select manually where each goes for the first iterations. CHANGE LATER.
    output_tif_path = base_dir / "data" / "adapted_tif_files"
    output_mask_path = base_dir / "data" / "adapted_mask_files"

    tif_files = sorted(tif_dir.glob("*.tif"))
    if not tif_files:
        logger.warning(f"No Tif files found in {tif_dir}.")

    for tif in tqdm(tif_files, desc = "Adapting Tif and masks"):
        file_name = tif.stem.replace("S2_patch_", "mask_")
        mask_path = mask_dir / f"{file_name}.png"

        # This should not happen, but I don't want the whole loop to break because of 1 mismatch.
        if not mask_path.exists():
            logger.error(f"Missing corresponding mask: {mask_path}. Skipping.")
            continue

        adapt_rasters(tif, mask_path, output_tif_path, output_mask_path)

    