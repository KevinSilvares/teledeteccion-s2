import torch
import random
import rasterio
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class SRDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-2 Super-Resolution.
    Handles dynamic reading, normalization and on-the-fly corrupt file fallback.
    """
    def __init__(self, lr_path: str | Path, hr_path: str | Path, default_value: float = 0.1) -> None:
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.default_value = default_value
        self.logger = logging.getLogger(__name__)

        self.lr_files = sorted(Path(lr_path).glob("*.tif"))
        self.hr_files = sorted(Path(hr_path).glob("*.tif"))

        if len(self.lr_files) <= 0:
            raise RuntimeError(f"LR directory has no files. Check {lr_path}")
        if len(self.hr_files) <= 0:
            raise RuntimeError(f"HR directory has no files. Check {hr_path}")
        if len(self.lr_files) != len(self.hr_files):
            raise RuntimeError(f"LR files ({len(self.lr_files)}) and HR files ({len(self.hr_files)}) must have exact number of files.")


    def __len__(self) -> int:
        return len(self.lr_files)
    

    def _read(self, file_path: str | Path) -> tuple[np.ndarray, float | None]:
        """
        Reads a tif safely. Returns (image_array, nodata).
        """
        # Rasterio has a known bug where it fails to read the file like it's corrupt. This is very problematic when it comes to large amount of files reading.
        max_retries = 3
        img = None
        no_data = None

        for i in range(max_retries):
            try:
                with rasterio.open(file_path) as src:
                    img = src.read()
                    no_data = src.nodata
                return img, no_data
            except Exception as e:
                if i < max_retries - 1:
                    self.logger.warning(f"[{i} / {max_retries}] Reading failed. File: {file_path} \n {e}")
                    time.sleep(0.5)
                else:
                    raise RuntimeError(f"File {file_path} corrupt. Remove it and it's associated pair.")
        

    def _normalize(self, img: np.ndarray, no_data: float | None) -> torch.Tensor:
        """
        Normalizes and adapts the tif values and masks nodata values.
        """
        tensor = torch.from_numpy(img).float()
        tensor = torch.nan_to_num(tensor, nan = 0.0, posinf = 1.0, neginf = 0.0)

        # Mask nodata values
        if no_data is not None:
            mask = (tensor != no_data)
        else:
            mask = tensor > 0.0
        
        if mask.any():
            min_value = tensor[mask].min()
            max_value = tensor[mask].max()
        
            # Min-Max scaling
            if max_value > min_value:
                tensor = (tensor - min_value) / (max_value - min_value)
            else:
                tensor = tensor - min_value
        else:
            # If the whole patch contains no data, gets filled with zeros
            tensor = torch.zeros_like(tensor)
        
        tensor[~mask] = 0.0
        return tensor


    def _extract_4_bands(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts the R, G, B and NIR bands from a 10-band Sentinel-2 tensor.
        Band Order: [B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12]
        """
        r = tensor[0:1, :, :]
        g = tensor[1:2, :, :]
        b = tensor[2:3, :, :]
        nir = tensor[3:4, :, :]
        return r, g, b, nir


    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            # Reads the pair
            lr_img, lr_no_data = self._read(self.lr_files[i])
            hr_img, hr_no_data = self._read(self.hr_files[i])

            # Normalizes values
            lr_tensor = self._normalize(lr_img, lr_no_data)
            hr_tensor = self._normalize(hr_img, hr_no_data)

            # Extracts the 4 target bands
            lr_r, lr_g, lr_b, lr_nir = self._extract_4_bands(lr_tensor)
            hr_r, hr_g, hr_b, hr_nir = self._extract_4_bands(hr_tensor)

            # Builds the 10 band tensor using the default value for the missing bands
            _, height, width = lr_tensor.shape
            dummy_band = torch.full((1, height, width), self.default_value, dtype = torch.float32)
            
            input_10_bands = torch.cat([
                lr_b,                   # B02
                lr_g,                   # B03
                lr_r,                   # B04
                dummy_band,             # B05
                dummy_band,             # B06
                dummy_band,             # B07
                lr_nir,                 # B08
                dummy_band,             # B8A
                dummy_band,             # B11
                dummy_band              # B12
            ], dim = 0)

            # Builds the target with only the 4 target bands
            target_4_bands = torch.cat([
                hr_b, 
                hr_g, 
                hr_r, 
                hr_nir
            ], dim = 0)

            return input_10_bands, target_4_bands
        except Exception as e:
            self.logger.error(f"File corrupt ({self.lr_files[i].name}). \n {e} \n Returning random pair to avoid stopping training.")

            # Gets a random index to keep training alive
            rand_index = random.randint(0 , len(self.lr_files) - 1)
            return self.__getitem__(rand_index)