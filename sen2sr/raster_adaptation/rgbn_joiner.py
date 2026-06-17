import logging
from pathlib import Path
import xarray as xr
import rioxarray as riox


logger = logging.getLogger(__name__)


def join_rgb_nir(rgb_path: Path | str, nir_path: Path | str) -> xr.DataArray:    
    """
    Joins an RGB raster with the NIR band of a False-Color raster to create a 4-band RGBNir DataArray.
    """    
    logger.info(f"Reading files: {Path(rgb_path).name} | {Path(nir_path).name}")
    raster_rgb = riox.open_rasterio(Path(rgb_path))
    raster_nir = riox.open_rasterio(Path(nir_path))

    # Size and CRS check
    if (raster_rgb.rio.width != raster_nir.rio.width) or (raster_rgb.rio.height != raster_nir.rio.height):
        error_message = f"Size mismatch: RGB {raster_rgb.rio.shape}, NIR {raster_nir.rio.shape}"
        logger.error(error_message)
        raise ValueError(error_message)
    if raster_rgb.rio.crs != raster_nir.rio.crs:
        error_message = f"CRS mismatch between RGB and NIR rasters. RGB: {raster_rgb.crs}, NIR: {raster_nir.crs}"
        logger.error(error_message)
        raise ValueError(error_message)

    logger.debug("Joining rasters")
    # Assuming NIR is the first band in the false-color image as it is in PNOA imagery
    nir = raster_nir.sel(band = 1)
    nir = nir.expand_dims(dim = "band")

    raster_rgbn = xr.concat([raster_rgb, nir], dim = "band")
    raster_rgbn = raster_rgbn.assign_coords(band = ["Red", "Green", "Blue", "NIR"])

    return raster_rgbn
   

def save_raster_to_disk(raster_rgbn: xr.DataArray, output_path: Path | str) -> None:
    """
    Saves the 4-band RGBNir raster to the output path.
    """
    output_path = Path(output_path)

    # Ensure outputh path ends with .tif
    if output_path.suffix not in [".tif", ".tiff"]:
        output_path = output_path.with_suffix(".tif")

    out_path.parent.mkdir(parents = True, exist_ok = True)

    raster_rgbn.rio.to_raster(
        output_path,
        driver = "GTIFF",
        compress = "lzw",
        tiled = True,
        interleave = "pixel",
        BIGTIFF = "YES"
    )       

    logger.info(f"Raster saved to {output_path}")