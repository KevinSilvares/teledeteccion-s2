import logging
from pathlib import Path
import xarray as xr
import rioxarray
from rasterio.enums import Resampling


logger = logging.getLogger(__name__)


def resample(self, raster: xr.DataArray, resample_type: Resampling = Resampling.average, res: float = 2.5) -> xr.DataArray:
    """Reprojects and resamples a raster to a new resolution."""
    logger.info(f"Resampling raster with type {resample_type.name}")
    return raster.rio.reproject(raster.rio.crs, resolution = res, resampling = resample_type, num_threads = True)


def resample_tif(self, tif_path: Path | str, res: float):
    """
    Opens a TIF file and applies average resampling.
    Average resampling is used by default as it yields the best results for this dataset.
    """
    # Assuming resampling type should always be "Average" because it's given better results
    raster = rioxarray.open_rasterio(tif_path)
    return self.resample(raster, tif_path, Resampling.average, res)


def save_to_disk(self, resampled_raster: xr.DataArray, tif_path: Path | str, res: float):
    """
    Saves the resampled raster to the output directory.
    """
    file_name = f"{tif_path.stem}_{res}.tif"
    output_path = self.OUTPUT_PATH_BASE / file_name

    resampled_raster.rio.to_raster(output_path)
    logger.info(f"Resampled saved in {output_path}") 