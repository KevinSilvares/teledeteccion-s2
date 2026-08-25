import torch
import logging
from datetime import datetime
from tqdm import tqdm
from pathlib import Path

from .model import Model


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main(tif_dir: Path | str, output_path: Path | str, model_path: Path | str = None) -> None:
    """
    Segments a set of tifs with a custom SegFormer model and saves them to the output path. 
    """
    if not model_path:
        logger.info("Model path not provided. Trying to load the model by the default saving path.")
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "model" / "best_model"

    model_wrapper = Model()
    model_wrapper.load_pretrained_model(model_path)

    tif_dir = Path(tif_dir)
    tif_files = sorted(tif_dir.glob("*.tif"))

    i = 1
    for tif in tqdm(tif_files, desc = "Segmentif tif files"):
        logger.info(f"Segmenting {tif}.")
        output_file = output_path / f"segmented_{tif.name}"
        model_wrapper.segment(tif, output_file)


if __name__ == "__main__":
    # The datetime will work as a control name
    dt_string = datetime.now().strftime("%Y%m%d_%H%M%S")

    base_dir = Path(__file__).resolve().parent
    test_tif_dir = base_dir / "data" / "test" / "imgs_tif"
    output_path = base_dir / "data" / f"predictions_{dt_string}"
    model_path = base_dir / "model" / "best_model"

    main(test_tif_dir, output_path, model_path)
