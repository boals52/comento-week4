
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config import (
    MAX_DEPTH,
    MIN_DEPTH,
)


def predict_metric_depth(
    image: Image.Image,
    model,
    image_processor,
    device: torch.device,
) -> np.ndarray:
    """
    PIL RGB 이미지에 대해 metric depth를 예측합니다.

    Returns
    -------
    np.ndarray
        원본 이미지와 동일한 H×W 크기의
        float32 metric depth map
    """

    if image is None:
        raise ValueError(
            "입력 이미지가 없습니다."
        )

    image = image.convert("RGB")

    original_width, original_height = (
        image.size
    )

    inputs = image_processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs[
        "pixel_values"
    ].to(device)

    with torch.inference_mode():
        outputs = model(
            pixel_values=pixel_values
        )

        predicted_depth = (
            outputs.predicted_depth
        )

        if predicted_depth.ndim == 3:
            predicted_depth = (
                predicted_depth.unsqueeze(1)
            )

        predicted_depth = F.interpolate(
            predicted_depth,
            size=(
                original_height,
                original_width,
            ),
            mode="bicubic",
            align_corners=False,
        )

        predicted_depth = predicted_depth[
            0,
            0,
        ]

        predicted_depth = torch.clamp(
            predicted_depth,
            min=MIN_DEPTH,
            max=MAX_DEPTH,
        )

    return (
        predicted_depth
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

def predict_objects(
    image: Image.Image,
    model,
    confidence_threshold: float = 0.5,
):
    image = image.convert("RGB")

    results = model.predict(
        source=np.asarray(image),
        conf=confidence_threshold,
        verbose=False,
    )

    result = results[0]

    detections = []

    for box in result.boxes:
        bbox = (
            box.xyxy[0]
            .detach()
            .cpu()
            .numpy()
            .astype(float)
            .tolist()
        )

        confidence = float(
            box.conf[0]
            .detach()
            .cpu()
        )

        class_id = int(
            box.cls[0]
            .detach()
            .cpu()
        )

        detections.append(
            {
                "bbox": bbox,
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": confidence,
            }
        )

    return detections


def estimate_bbox_depth(
    depth_map: np.ndarray,
    bbox,
    center_ratio: float = 0.5,
):
    x1, y1, x2, y2 = [
        int(round(value))
        for value in bbox
    ]

    height, width = depth_map.shape

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        return None

    box_width = x2 - x1
    box_height = y2 - y1

    margin_x = int(
        box_width
        * (1.0 - center_ratio)
        / 2.0
    )

    margin_y = int(
        box_height
        * (1.0 - center_ratio)
        / 2.0
    )

    center_x1 = x1 + margin_x
    center_x2 = x2 - margin_x
    center_y1 = y1 + margin_y
    center_y2 = y2 - margin_y

    region = depth_map[
        center_y1:center_y2,
        center_x1:center_x2,
    ]

    valid_mask = (
        np.isfinite(region)
        & (region > 0)
    )

    if not valid_mask.any():
        return None

    return float(
        np.median(
            region[valid_mask]
        )
    )


def attach_depth_to_detections(
    detections,
    depth_map,
    center_ratio: float = 0.5,
):
    results = []

    for detection in detections:
        combined = detection.copy()

        combined["depth"] = (
            estimate_bbox_depth(
                depth_map=depth_map,
                bbox=detection["bbox"],
                center_ratio=center_ratio,
            )
        )

        results.append(
            combined
        )

    return results