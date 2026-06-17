import logging
from tqdm import tqdm
from pathlib import Path

import tiler
import raster_resampler as resampler
import rgbn_joiner as rgbn_j 


# Settings
# Basic logging
logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")

# Patch settings
PATCH_SIZE: int = 128
OVERLAP: float = 0.5


def _resolve_paths(base_dir: Path) -> tuple[Path, Path, Path, Path]:
    """
    Resolves data input and output directories dynamically based on script location.
    """
    # Not a big fan of magic strings. Might change. ARGS_PARSE TO BE ADDED
    rgb_path = base_dir / "data" / "rgb_files"
    nir_path = base_dir / "data" / "nir_files"
    output_lr = base_dir / "data" / "lr"
    output_hr = base_dir / "data" / "hr"
    return rgb_path, nir_path, output_lr, output_hr


def main():
    script_dir = Path(__file__).resolve().parent.parent
    rgb_path, nir_path, output_lr, output_hr = _resolve_paths(script_dir)

    # Searches for the file name pattern. File names are identical until the _IRG_ part, which identifies false-color
    nir_files = sorted(list(nir_path.glob("*_IRG_*.tif")))
    logging.info(f"Found {len(nir_files)} elements to process.")

    if not nir_files:
        logging.info(f"No NIR files found in {nir_path}. Exiting")
        return

    progress_bar = tqdm(nir_files, unit = "pair", dynamic_ncols = True)

    for nir_file_path in progress_bar:
        nir_file_name = nir_file_path.name
        rgb_file_name = nir_file_name.replace("IRG_", "")
        rgb_file_path = rgb_path / rgb_file_name

        if not rgb_file_path.exists():
            tqdm.write(f"No RGB pair founded for {nir_file_name}. Skipping.")
            continue

        progress_bar.set_description(f"Processing {rgb_file_name}")

        try:
            # 4-band RGBNir raster
            raster_rgbn = rgbn_j.join_rgb_nir(rgb_file_path, nir_file_path)

            # Resample 2.5 m/px HR and 10 m/px LR
            raster_hr = resampler.resample(raster = raster_rgbn, res = 2.5)
            raster_lr = resampler.resample(raster = raster_rgbn, res = 10)

            # Path tiling
            tiler.tile(
                raster_lr = raster_lr,
                raster_hr = raster_hr,
                output_path_lr = output_lr,
                output_path_hr = output_hr,
                tile_size = PATCH_SIZE,
                overlap = OVERLAP
            )
        except Exception as e:
            tqdm.write(f"Failed to process {rgb_file_name}: {e}")

    logging.info("=== Pipeline Processing Finished ===")


if __name__ == "__main__":
    main() 