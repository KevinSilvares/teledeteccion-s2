import torch
import torch.nn as nn


class MaskedL1Loss(nn.Module):
    """
    Computes L1 loss only on valid pixels, ignoring no_data regions, only for the R, G, B and NIR bands from the 10-band prediction.
    """
    def __init__(self, no_data: float = 0.0) -> None:
        super().__init__()
        self.no_data = no_data

    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Only uses the B, G, R and NIR bands (in that order).
        pred_sliced = pred[:, [0, 1, 2, 6], :, :]

        # Masks out the non valid pixel (those that contain nodata values)
        valid_mask = (target != self.no_data)

        # Calculates the absolute difference (L1 loss) per pixel
        abs_diff = torch.abs(pred_sliced - target)

        # Avoids the nodata being predicted wrong
        mask_diff = abs_diff * valid_mask.float()

        return mask_diff.sum() / (valid_mask.sum() + 1e-8)



