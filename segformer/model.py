import torch
import rasterio
import logging
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from rasterio.windows import Window
from transformers import SegformerForSemanticSegmentation


class Model:
    def __init__(self, device: str | torch.device = "cuda") -> None:
        self.device = torch.device(device)
        self.model = None
        self.logger = logging.getLogger(__name__)
    

    def load_base_model(self, base_model: str = "nvidia/mit-b3", num_labels: int = 2) -> torch.nn.Module:
        """
        Loads a base model from Hugging Face and adapts the input layer for 4 channels (R, G, B, NIR)
        """
        self.logger.info(f"Downloading base model from Hugging-Face: {base_model}")
        
        # Class config
        id2label = {0: "background", 1: "path"}
        label2id = {"background": 0, "path": 1}

        model = SegformerForSemanticSegmentation.from_pretrained(
            base_model,
            num_labels = num_labels,
            id2label = id2label,
            label2id = label2id,
            ignore_mismatched_sizes = True,
            use_safetensors = True
        )

        model.config.num_channels = 4

        # Changes the original 3 channel input layer for a new 4 input channels layer so Segformer can use R, G, B, NIR
        self.logger.debug("Finding the 3-channel input layer.")
        # Finds the original input layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
                original_layer = module
                layer_name = name
                break
        
        if not original_layer:
            raise RuntimeError("3-channel Conv2d input layer not found.")
        
        self.logger.debug(f"3-channels input layer found: {layer_name}")
        new_layer = torch.nn.Conv2d(
            in_channels = 4,
            out_channels = original_layer.out_channels,
            kernel_size = original_layer.kernel_size,
            stride = original_layer.stride,
            padding = original_layer.padding
        )

        # Copies the RGB weights and for the red band to the NIR band because it's the closest
        with torch.no_grad():
            # RGB bands
            new_layer.weight[:, :3, :, :] = original_layer.weight

            # NIR band
            new_layer.weight[:, 3:4, :, :] = original_layer.weight[:, 0:1, :, :]

            if original_layer.bias is not None:
                new_layer.bias.copy_(original_layer.bias)
            
            path_segments = layer_name.split(".")
            parent_module = model

            for segment in path_segments[:-1]:
                parent_module = getattr(parent_module, segment)

            # Replace the old 3-channel layer with the new 4-channel layer
            setattr(parent_module, path_segments[-1], new_layer)

        self.model = model
        self.model.to(self.device)
        self.logger.info("Model loaded.")
        return self.model
    

    def load_pretrained_model(self, model_path: str | Path = r"./segformer_pretrained", is_eval: bool = True) -> torch.nn.Module:
        """
        Loads a pre-trained model and can set it to evaluation model (it's set to evaluation mode by default).
        """

        self.logger.info(f"Loading pre-trained model from {model_path}")
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_path,
            num_channels = 4
        )
        self.model.to(self.device)
        self.logger.info("Pre-trained model loaded.")

        if is_eval:
            self.logger.debug("Model set to evaluation mode.")
            self.model.eval()

        return self.model
    

    def segment(self, tif_path: str | Path, output_path: str | Path, patch_size: int = 512) -> None:
        """
        Segments a 4-band tif file.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded or not found.")
        
        tif_path = Path(tif_path)
        output_path = Path(output_path)

        output_path.parent.mkdir(parents = True, exist_ok = True)

        self.logger.info(f"Segmenting: {tif_path.name}")

        with rasterio.open(tif_path) as src:
            metadata = src.profile.copy()
            # Converts the image to black and white
            metadata.update(
                dtype = "uint8",
                count = 1,
                compress = "lzw"
            )

            height, width = src.height, src.width

            with rasterio.open(output_path, "w", **metadata) as dst:
                for y in range(0, height, patch_size):
                    for x in range(0, width, patch_size):
                        window = Window(x, y, patch_size, patch_size)

                        patch = src.read(window = window)

                        # Skip entirely empty patches
                        if not patch.any():
                            continue
                        
                        # Band reordering
                        patch_rgbn = np.concatenate((
                            patch[2:3, :, :],
                            patch[1:2, :, :],
                            patch[0:1, :, :],
                            patch[3:4, :, :]
                        ), axis = 0)

                        # Normalization (clipping 2nd/98th percentile)
                        p2, p98 = np.percentile(patch_rgbn, (2, 98))
                        patch_rgbn = (patch_rgbn - p2) / (p98 - p2 + 1e-8)
                        patch_clipped = np.clip(patch_rgbn, 0.0, 1.0)

                        tensor = torch.from_numpy(patch_clipped).float().unsqueeze(0).to(self.device)

                        with torch.no_grad():
                            output = self.model(pixel_values = tensor)
                            logits = output.logits

                            # Resize logits to the exact patch shape
                            logits_resized = F.interpolate(
                                logits,
                                size = (patch.shape[1], patch.shape[2]),
                                mode = "bilinear",
                                align_corners = False
                            )

                            pred = logits_resized.argmax(dim = 1).squeeze().cpu().numpy()

                            dst.write(pred.astype("uint8"), 1, window = window)
        
        self.logger.info(f"Full image segmented and saved to {output_path}")