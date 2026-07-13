import torch
import rasterio
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

import transformations as T 


class SegFormerDataset(Dataset):
    def __init__(self, tif_dir: str | Path, mask_dir: str | Path, is_train: bool = True) -> None:
        self.tif_dir = Path(tif_dir)
        self.mask_dir = Path(mask_dir)

        transform_train, transform_val = T.get_transformations()
        self.transformations = transform_train if is_train else transform_val
        
        self.logger = logging.getLogger(__name__)

        self.tif_files = sorted(self.tif_dir.glob("*.tif"))
        if not self.tif_files:
            raise RuntimeError(f"No .tif files found in {self.tif_dir}")

        self.logger.info(f"Initialized {'Training' if is_train else 'Validation'} Dataset with {len(self.tif_files)} files.")
    

    def __len__(self) -> int:
        return len(self.tif_files)


    def _build_rgbn(self, tif_path: str | Path) -> np.ndarray:
        """
        Reads the TIF and returns a standarized Height, Width, Channels ndarray.
        Assumes input is a (super resolved) 4-band tif [R, G, B, NIR].
        """
        with rasterio.open(tif_path) as src:
            file = src.read()

            # Creates the 4-band ndarray
            img_rgbn = np.concatenate((
                file[2:3, :, :],
                file[1:2, :, :],
                file[0:1, :, :],
                file[3:4, :, :]
            ), axis = 0)

            # Channel, Height, Width -> Height, Width, Channels
            img_rgbn = np.transpose(img_rgbn, (1, 2, 0))
            return img_rgbn


    def _clip_tif(self, img_rgbn: np.ndarray) -> np.ndarray:
        """
        Normalizes the image using 2nd and 98th percentile clipping.
        """
        p2, p98 = np.percentile(img_rgbn, (2, 98))
        img_rgbn = (img_rgbn - p2) / (p98 - p2 + 1e-8)

        img_clipped = np.clip(img_rgbn, 0.0, 1.0)
        return img_clipped.astype(np.float32)


    def _build_mask(self, mask_path: str | Path) -> np.ndarray:
        """
        Loads the mask and converts it to binary classes. (0 = background, 1 = Path).
        """
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}.")
        
        mask_pil = Image.open(mask_path).convert("L")
        mask_np = np.array(mask_pil)

        # Sets a threshold for the patlabel
        # mask = (mask_np < 128).astype(np.int64)
        return mask_np.astype(np.int64)
    

    def _apply_transforms(self, img: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies Albumentations. f no spatial transform are presents (for validation), just handles the ToTensorV" conversion.
        """
        augmented = self.transformations(image = img, mask = mask)
        return augmented["image"], augmented["mask"]


    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        try:
            tif_path = self.tif_files[i]

            mask_name = tif_path.name.replace(".tif", "_mask.png")
            mask_path = self.mask_dir / mask_name

            img_rgbn = self._build_rgbn(tif_path)
            img_rgbn = self._clip_tif(img_rgbn)
            
            mask = self._build_mask(mask_path)

            img_tensor, mask_tensor = self._apply_transforms(img_rgbn, mask)

            # Hugging-Face formatted dict
            return {
                "pixel_values": img_tensor,
                "labels": mask_tensor.long()
            }
        except Exception as e:
            self.logger.error(f"Error loading {self.tif_files[i].name}: {e}")
            raise