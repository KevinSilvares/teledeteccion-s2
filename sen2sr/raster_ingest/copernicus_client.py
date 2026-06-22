import time
import os
import requests
import json
import uuid
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from pyproj import Transformer


from aoi_generator import aoi_gen


class CopernicusClient:
    """
    Handles authentication and data ingestion from the Copernicus Processing API.
    """

    def __init__(self, client_id: str, client_secret: str):
        self.URL_REQUEST = "https://sh.dataspace.copernicus.eu/process/v1"
        self.EPSG_HEMISFERIO_NORTE_BASE = 32600
        self.EPSG_HEMISFERIO_SUR_BASE = 32700
        self.DEFAULT_EPSG = "3857"
        self.oauth = None

        self.client_id = client_id
        self.client_secret = client_secret

        self.logger = logging.basicConfig(level = logging.INFO, format = "%(levelname)s: %(message)s")


    def get_copernicus_authenticated_session(self) -> OAuth2Session:
        """
        Genereates or retrieves the cached Copernicus authenticated session (OAuth).
        """
        # Singleton
        if self.oauth:
            return self.oauth
        else:
            client = BackendApplicationClient(client_id = self.client_id)

            oauth_copernicus = OAuth2Session(client = client)
            oauth_copernicus.fetch_token(
                token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                client_secret = self.client_secret,
                include_client_id = True,
            )

        self.oauth = oauth_copernicus
        return self.oauth


    def request(self, payload: str, dest: Path, file_name: str, retries: int = 3) -> bool:
        """
        Makes an HTTP request. Implements error handling and automatic oauth re-generation. Saves the TIFF.
        """

        oauth = self.get_copernicus_authenticated_session()
        try:
            response = oauth.post(
                self.URL_REQUEST,
                json = payload,
                headers = {"Content-Type": "application/json", "Accept": "image/tiff"}
            )

            if response.status_code == 401:
                self.logger.warning("Token expired. Re-authenticating.")
                oauth = None
                return self.request(oauth, payload, destino, file_name)
            elif response.status_code == 429:
                self.logger.info(f"Rate limit hit. Waiting 30 seconds. Retries left: {retries - 1}")
                time.sleep(30)
                return self.request(oauth, payload, destino, file_name, retries - 1)
            elif response.status_code not in (200, 401, 429):
                self.logger.error(f"{response.text}")
                return False

            response.raise_for_status()

            # Saves the file
            filepath = destino / nombre_archivo
            destino.mkdir(parents = True, exist_ok = True)

            with open(filepath, "wb") as f:
                f.write(response.content)

            self.logger.info(f"Succesfully downloaded: {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"ERROR: {e}")
            return False


    def _get_api_bbox(self, poi: dict[str, float], height_pixels: int = 512, width_pixels: int = 512, res_m: float = 10):
        """
        Translates a center POI into a bounding box dictionary formatted for the Copernicus API.
        """
        if height_pixels % 128 != 0 or width_pixels % 128 != 0:
            raise ValueError("Height and width pixels must be a multiple of 128 (it's recommended 128, 256, 512).")

        epsg = aoigen.calc_epsg(poi["lon"], poi["lat"])

        # Calculates dimensions in meters
        height_meters = height_pixels * res_m
        width_meters = width_pixels * res_m
        half_heigth_m = height_meters / 2
        half_width_m = width_meters / 2

        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy = True)
        center_x, center_y = transformer.transform(poi["lon"], poi["lat"])

        min_x = centro_x - mitad_ancho_m
        max_x = centro_x + mitad_ancho_m
        min_y = centro_y - mitad_alto_m
        max_y = centro_y + mitad_alto_m

        return {
            "properties": {
                "crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"
            },
            "bbox": [min_x, min_y, max_x, max_y]
        }


    def get_10_bands_s2_payload(self,
        POI: POI,
        date_start: str | date,
        date_end: str | date,
        height_pixels: int = 512,
        width_pixels: int = 512,
        max_cloud_coverage: int = 10
        ) -> str:
        """
        Constructs the JSON payload for a 10-band Sentinel-2 request.
        """
        bbox_dict = self._get_api_bbox(POI, height_pixels, width_pixels)
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"],
                output: {
                    bands: 10,
                    sampleType: "FLOAT32"
                }
            }
        }

        function evaluatePixel(sample) {
            return [
                sample.B02, sample.B03, sample.B04,
                sample.B05, sample.B06, sample.B07,
                sample.B08, sample.B8A, sample.B11, sample.B12
            ]
        }
        """

        payload = {
            "input": {
                "bounds": bbox_dict,
                "data": [
                    {
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": f"{date_start}T00:00:00Z",
                                "to": f"{date_end}T23:59:59Z",
                            },
                            "maxCloudCoverage": max_cloud_coverage,
                            "mosaickingOrder": "leastCC"
                        },
                    }
                ],
            },
            "output": {
                "width": width_pixels,
                "height": height_pixels,
                "responses": [
                    {
                        "identifier": "default",
                        "format": {"type": "image/tiff"}
                    }
                ],
            },
            "evalscript": evalscript,
        }

        return payload