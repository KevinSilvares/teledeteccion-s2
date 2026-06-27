import logging
import math
import uuid
from pyproj import Transformer


logger = logging.getLogger(__name__)


def calc_epsg(lon: float, lat: float) -> int:
    """
    Calculates dynamically the UTM EPSG for a given WGS84 coordinates.
    """
    epsg_base = 32600 if lat >= 0 else 32700
    epsg_calc = int((lon + 180) / 6) + 1
    epsg_utm = epsg_base + epsg_calc
    return epsg_utm


def _generate_bboxes(columns: int,
    rows: int,
    epsg_utm: int,
    min_x_utm: float,
    min_y_utm: float,
    patch_size: int = 5120
    ) -> list[dict[str, any]]:
    """
    Generates a list of bounding boxes and calculates their center POI in WGS84.
    """
    logger.info("Generating bounding boxes")
    
    bboxes = []

    # Adapts the EPSG to the Copernicus API (EPSG:4326 = WGS84)
    transformer_degrees = Transformer.from_crs(f"EPSG:{epsg_utm}", "EPSG:4326", always_xy = True)

    for row in range(rows):
        for col in range(columns):
            # Calculates the minimum lon and lat for the patch
            box_min_x_utm = min_x_utm + (col * patch_size)
            box_min_y_utm = min_y_utm + (row * patch_size)

            # Calculates the lon and lat center
            x_utm_center = box_min_x_utm + (patch_size / 2)
            y_utm_center = box_min_y_utm + (patch_size / 2)

            lon_center, lat_center = transformer_degrees.transform(x_utm_center, y_utm_center)

            bboxes.append({
                "id": uuid.uuid4().hex[:8],
                "poi": {"lat": lat_center, "lon": lon_center}
            })

    return bboxes


def generate_grid_from_wgs84(min_lon: float, min_lat: float, max_lon: float, max_lat: float, patch_size: int = 5120) -> list[dict[str, any]]:
    """
    Generates a grid of patches transforming coordinates from WGS84 (EPSG:4326) to a dynamically calculated EPSG.
    """
    logger.info("Calculating grid for the given area")

    # Ensures the given minimum and maximums are correct
    min_lon, max_lon = min(min_lon, max_lon), max(min_lon, max_lon)
    min_lat, max_lat = min(min_lat, max_lat), max(min_lat, max_lat)

    epsg_utm = calc_epsg(min_lon, min_lat)

    transformer_metres = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_utm}", always_xy = True)

    # Bounds in metres
    min_x_utm, min_y_utm = transformer_metres.transform(min_lon, min_lat)
    max_x_utm, max_y_utm = transformer_metres.transform(max_lon, max_lat)
    height_total_metres = max_y_utm - min_y_utm
    width_total_metres = max_x_utm - min_x_utm

    columns = math.ceil(width_total_metres / patch_size)
    rows = math.ceil(height_total_metres / patch_size)
    logger.info(f"Total area: {width_total_metres / 1000:.1f} km x {height_total_metres / 1000:.1f} km")
    logger.info(f"Grid: {columns} columns x {rows} rows = {columns * rows} patches.")

    return _generate_bboxes(columns, rows, epsg_utm, min_x_utm, min_y_utm)