import torch
import rasterio
import numpy as np
from pathlib import Path
from torch.utilts.data import Dataset
from PIL import Image


class SegFormerDataset(Dataset):
    def __init__(self, tif_dir: str | Path, mask_dir: str | Path, transformation = None)