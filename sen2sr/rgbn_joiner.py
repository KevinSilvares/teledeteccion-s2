from pathlib import Path
import xarray as xr
import rioxarray as riox


def join_rasters(self, rgb_path: Path | str, nir_path: Path | str) -> xr.DataArray:    
    """
    Joins a RGB raster with the NIR band of a False-Color raster to create a 4-band RGBNir DataArray.
    """    
    print(f"Reading files: {Path(rgb_path).name} | {Path(nir_path).name}")
    raster_rgb = riox.open_rasterio(Path(rgb_path))
    raster_nir = riox.open_rasterio(Path(nir_path))

    if (raster_rgb.rio.width != raster_nir.rio.width) or (raster_rgb.rio.height != raster_nir.rio.height):
        raise ValueError(f"Size mismatch: RGB is {raster_rgb.rio.shape}, NIR is {raster_nir.shape}")
    if raster_rgb.crs != raster_nir.crs:
        raise ValueError(f"CRS mismatch between RGB and NIR rasters. RGB: {raster_rgb.crs}, NIR: {raster_nir.crs}")

    print("Joining rasters")
    # Assuming NIR is the first band in the false-color image as it is in PNOA imagery
    nir = raster_nir.sel(band = 1)
    nir = nir.expand_dims(dim = "band")

    raster_rgbn = xr.concat([raster_rgb, nir], dim = "band")
    raster_rgbn = raster_rgbn.assign_coords(band = ["Red", "Green", "Blue", "NIR"])

    return raster_rgbn
   

def save_raster_to_disk(self, raster_rgbn: xr.DataArray, output_path: Path | str) -> None:
    """
    Saves the 4-band RGBNir raster to the output path.
    """
    output_path = Path(output_path)
    output_path.mkdir(parents = True, exist_ok = True)

    # Ensure outputh path ends with .tif
    if output_path.suffix != ".tif":
        output_path = output_path.with_suffix(".tif")

    raster_rgbn.rio.to_raster(
        output_path,
        driver = "GTIFF",
        compress = "lzw",
        tiled = True,
        interleave = "pixel",
        BIGTIFF = "YES"
    )       

    print(f"Raster saved to {output_path}")