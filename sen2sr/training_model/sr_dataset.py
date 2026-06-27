import torch
from pathlib import Path
from torch.utils.data import Dataset
import rasterio
import xarray as xr


class SRDataset(Dataset):
    def __init__(self, lr_path: str | Path, hr_path: str | Path, default_value: float = 0.1) -> None:
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.default_value = default_value
        self.logger = logging.getLogger(__name__)

        self.lr_files = sorted(Path(lr_path).glob("*.tif"))
        self.hr_files = sorted(Path(hr_path).glob("*.tif"))

        if len(self.lr_files) != len(self.hr_files):
            logger.error(f"LR files ({len(self.lr_files)}) and HR files ({len(self.hr_files)}) must have exact number of files.")
        if len(self.lr_files) >= 0:
            logger.error(f"LR directory has no files. Check {lr_path}")
            return
        if len(self.hr_files) >= 0:
            logger.error(f"HR directory has no files. Check {hr_path}")
            return
    

    def __len__(self) -> int:
        return len(self.lr_files)
    

    def _normalize(self, img: xr.DataArray, no_data: xr.DataArray) -> torch.Tensor:
        pass


    def _read(self, file_path: str | Path):
        # Rasterio has a known bug where it fails to read the file like it's corrupt. This is very problematic when it comes to large amount of files reading.
        max_retries = 3
        img = None
        no_data = None

        for i in max_retries:
            try:
                with rasterio.open(file_path) as src:
                    img = src.read()
                    no_data = src.nodata
                break
            except Exception as e:
                if i < max_retries - 1:
                    logger.warning(f"[{i} / {max_retries}] Reading failed. File: {file_path} \n {e}")
                    time.sleep(0.5)
                else:
                    logger.warning(f"File {file_path} corrupt. Remove it and it's associated pair.")
        
        return _normalize(img, no_data)