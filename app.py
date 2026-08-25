import torch
import gradio as gr
import numpy as np
import rasterio
import logging
from pathlib import Path

from sen2sr_pipeline.resolver.resolver_config import ResolverConfig
from sen2sr_pipeline.resolver.resolver import get_model as get_sen2sr, _process_image as run_sen2sr
from segformer import segment

from segformer.model import Model

logging.basicConfig(level = logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class App:
    def __init__(self) -> None:
        logger.info("[App] Loading models")
        
        logger.info("[App] Loading Sen2SR")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_sr = ResolverConfig()
        self.sen2sr_model = get_sen2sr(config_sr, self.device)
        logger.info("[App] Sen2SR loaded")

        logger.info("[App] Loading Segformer")
        self.segformer_model = Model(device = self.device)
        base_script = Path(__file__).resolve().parent
        segformer_path = base_script / "segformer" / "model" / "best_model"
        self.segformer_model.load_pretrained_model(segformer_path)
        logger.info("[App] Segformer loaded")

        logger.info("[App] Models loaded.")
        

    def build_ui(self) -> None:
        """
        Builds the UI using Gradio components.
        """
        with gr.Blocks(theme = gr.themes.Base()) as self.ui:
            gr.Markdown("# Sentinel-2 Road Detection")
            gr.Markdown("Upload a Sentinel-2 multispectral `.tif` file (10 bands) to run Super-Resolution and Semantic Segmentation")

            with gr.Row():
                with gr.Column():
                    input_file = gr.File(label = "Input patch (.tif)")
                    submit_button = gr.Button("Process")

                    with gr.Column():
                        output_img = gr.Image(label = "Result (Super-Resolution + Mask)")
            
            submit_button.click(
                fn = self._process_file,
                inputs = input_file,
                outputs = output_img
            )
    

    def _process_file(self, tif_file: Path | str):
        """
        Runs the process to super-resolve and segment the input TIF file.
        """
        if tif_file is None:
            return None

        input_path = tif_file.name

        try:
            with rasterio.open(input_path) as src:
                img = src.read()

            rgb_preview = np.stack([img[2], img[1], img[0]], axis = -1).astype(np.float32)

            p2, p98 = np.percentile(rgb_preview, (2, 98))
            rgb_preview = (rgb_preview - p2) / (p98 - p2 + 1e-8)
            rgb_preview = np.clip(rgb_preview, 0.0, 1.0)

            # Gradio needs an 8-bit image to display
            base_img = (rgb_preview * 255).astype(np.uint8)

            return base_img
        except Exception as e:
            logger.error(f"[App] ERROR: {e}")
            return np.zeros((512, 512, 3), dtype = np.uint8)


    def run(self) -> None:
        """
        Runs web server and builds the UI.
        """
        if not hasattr(self, "ui"):
            self.build_ui()
        
        self.ui.launch()

    
if __name__ == "__main__":
    app = App()
    app.run()