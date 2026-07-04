import evaluate
import torch
import torch.nn.functional as F
import numpy as np


class IOU:
    """
    Computes Mean Intersection over Union (mIoU) for SegFormer predictions.
    This class is designed to be passed as the 'compute_metrics' argument in a Hugging-Face Trainer.
    """
    def __init__(self, num_labels: int = 2) -> None:
        self.num_labels = num_labels
        self.criterion = evaluate.load("mean_iou")

    
    def __call__(self, eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred

        # Converts numpy arrays to PyTorch tensors because I feel more comfortable with torch interpolation.
        logits_tensor = torch.from_numpy(logits)
        labels_tensor = torch.from_numpy(labels)

        # Interpolates the logits to their original resolution. SegFormer ouputs logit at 1/4 by default.
        logits_tensor = F.interpolate(
            logits_tensor,
            size = (labels_tensor.shape[1], labels_tensor.shape[2]),
            mode = "bilinear",
            align_corners = False
        ).argmax(dim = 1)

        # Converts back to numpy to evaluate
        preds_np = logits_tensor.cpu().numpy()
        labels_np = labels_tensor.cpu().numpy()

        results = self.criterion.compute(
            predictions = preds_np,
            references = labels_np,
            num_labels = self.num_labels,
            ignore_index = 255 # this ignores the black borders 
        )

        # Extracts only the 'path' class. This is the only interesting label
        iou_path = results["per_category_iou"][1]

        return {
            "iou_path": iou_path,
            "mean_iou": results["mean_iou"]
        }