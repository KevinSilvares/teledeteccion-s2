import os
import logging
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv

import aoi_generator as aoi_gen
from copernicus_client import CopernicusClient


logging.basicConfig(level = logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _get_client_info() -> tuple[str, str] | None:
    """
    Gets the client credentials from a .env file. Returns None if not found.
    """
    load_dotenv()
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")

    if not client_id or not client_secret:
        logger.error("Copernicus credentials not found. Make sure your .env file exists and contains CLIENT_ID and CLIENT_SECRET.")
        return None
    return client_id, client_secret


def main() -> None:
    credentials = _get_client_info()
    if not credentials:
        return
    
    client_id, client_secret = credentials
    client = CopernicusClient(client_id, client_secret)

    logger.info("Initializing Area of Interest grid.")
    pois = [
        { # Leon
            "min_lon": -6.423595,
            "min_lat": 42.980884,
            "max_lon": -6.238093,
            "max_lat": 42.817713
        },
        { # Asturias
            "min_lon": -6.379602,
            "min_lat": 43.534458,
            "max_lon": -6.229893,
            "max_lat": 43.480966
        },
        { # Andalucia
            "min_lon": -6.548216,
            "min_lat": 38.000913,
            "max_lon": -6.371529,
            "max_lat": 37.877346
        },
        { # Almeria
            "min_lon": -2.069388,
            "min_lat": 37.034163,
            "max_lon": -2.017125,
            "max_lat": 37.011797
        }
    ]

    bboxes = []
    for poi in pois:
        bboxes.extend(aoi_gen.generate_grid_from_wgs84(
            min_lon = poi["min_lon"],
            min_lat = poi["min_lat"],
            max_lon = poi["max_lon"],
            max_lat = poi["max_lat"]
        ))

    dest = Path(__file__).resolve().parent.parent / "data" / "ingest"

    for patch in tqdm(bboxes, desc = "Processing regionals grids"):
        unique_id = patch["id"]
        file_name = f"S2_patch_{unique_id}.tif"

        logger.info(f"Processing {file_name}")

        payload = client.get_10_bands_s2_payload(
            poi = patch["poi"],
            date_start = "2023-07-01",
            date_end = "2023-08-31"
        )

        client.request(
            payload = payload,
            dest = dest,
            file_name = file_name
        ) 

    logger.info("=== Data Ingestion Pipeline Finished ===")


if __name__ == "__main__":
    main()