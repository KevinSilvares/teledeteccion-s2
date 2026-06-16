import logging
from pathlib import Path
import xarray as xr
import rioxarray as riox
import os


class RasterTiler():
    """
    Handles the generation of matched low-res and high-res raster patches.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def tile(
        self, 
        raster_lr: xr.DataArray,
        raster_hr: xr.DataArray,
        output_path_lr: Path | str,
        output_path_hr: Path | str,
        tile_size: int = 128,
        overlap: float = 0.5,
        scale_factor: int = 4
        ) -> None:
        """
        Tiles a pair of low-resolution and high-resolution rasters and saves the patches to the output directories.
        """
        # Ensure outputs are Path objects
        out_lr = Path(output_path_lr)
        out_hr = Path(output_path_hr)
        # Create directories
        output_path_lr.mkdir(parents = True, exist_ok = True)
        output_path_hr.mkdir(parents = True, exist_ok = True)

        patch_size_lr = tile_size
        patch_size_hr = patch_size_lr * scale_factor

        # Calculate the step with overlap
        step_lr = int(patch_size_lr * (1 - overlap))
        width_lr, height_lr = raster_lr.rio.width, raster_lr.rio.height

        # scandir() yields best performance comapred to listdir()
        num = sum(1 for _ in os.scandir(output_path_lr)) 
        
        for y in range(0, height_lr - patch_size_lr + 1, step_lr):
            for x in range(0, width_lr - patch_size_lr + 1, step_lr):
                # Crop LR
                patch_lr = raster_lr.isel(
                    x = slice(x, x + patch_size_lr),
                    y = slice(y, y + patch_size_lr)
                )

                # Crop HR
                x_hr = x * scale_factor
                y_hr = y * scale_factor
                patch_hr = raster_hr.isel(
                    x = slice(x_hr, x_hr + patch_size_hr),
                    y = slice(y_hr, y_hr + patch_size_hr)
                )

                # Check if patches have data
                if patch_lr.any() and patch_hr.any():
                    num += 1
                    file_name = f"patch_{num:04d}.tif"

                    self._save_files(file_name, patch_lr, patch_hr, output_path_lr, output_path_hr)

        self.logger.info(f"Saving pair {num:04d} LR ({tile_size}) | HR ({tile_size * scale_factor})")

    
    def _save_files(
        self,
        file_name: str,
        patch_lr: xr.DataArray,
        patch_hr: xr.DataArray,
        output_path_lr: Path | str,
        output_path_hr: Path | str,
        ) -> None:
        """
        Save both patches to their respective output directories.
        """
        output_lr = output_path_lr / file_name
        output_hr = output_path_hr / file_name

        patch_lr.rio.to_raster(output_lr, compress = "lzw")
        patch_hr.rio.to_raster(output_hr, compress = "lzw")