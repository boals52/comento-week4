
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .config import DEVICE, MAX_DEPTH, MIN_DEPTH


# ============================================================
# 1. 이미지 배치를 모델 입력 형식으로 변환
# ============================================================

def prepare_model_inputs(
    images: torch.Tensor,
    image_processor: Any,
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """
    Convert RGB image tensors into model-ready pixel values.

    Parameters
    ----------
    images : torch.Tensor
        Image batch with shape [B, 3, H, W].
        Pixel values must be in the [0, 1] range.
    image_processor : Any
        Hugging Face image processor.
    device : torch.device
        Device on which the model will run.

    Returns
    -------
    torch.Tensor
        Processed pixel values.
    """

    if images.ndim != 4:
        raise ValueError(
            "images must have shape [B, 3, H, W]. "
            f"Received: {tuple(images.shape)}"
        )

    if images.shape[1] != 3:
        raise ValueError(
            "images must contain 3 RGB channels. "
            f"Received shape: {tuple(images.shape)}"
        )

    images = images.detach().cpu()

    image_list = []

    for image in images:
        image = image.permute(1, 2, 0).numpy()

        image = np.clip(
            image * 255.0,
            0,
            255,
        ).astype(np.uint8)

        image_list.append(image)

    processed = image_processor(
        images=image_list,
        return_tensors="pt",
        do_resize=False,
    )

    if "pixel_values" not in processed:
        raise KeyError(
            "The image processor output does not contain "
            "'pixel_values'."
        )

    pixel_values = processed["pixel_values"].to(
        device,
        non_blocking=True,
    )

    return pixel_values


# ============================================================
# 2. 모델 출력에서 predicted depth 추출
# ============================================================

def extract_predicted_depth(
    model_outputs: Any,
) -> torch.Tensor:
    """
    Extract predicted depth from model outputs.

    Returns
    -------
    torch.Tensor
        Predicted depth with shape [B, 1, H, W].
    """

    if hasattr(model_outputs, "predicted_depth"):
        predicted_depth = model_outputs.predicted_depth

    elif isinstance(model_outputs, dict):
        if "predicted_depth" not in model_outputs:
            raise KeyError(
                "Model output does not contain "
                "'predicted_depth'."
            )

        predicted_depth = model_outputs["predicted_depth"]

    else:
        raise TypeError(
            "Unsupported model output type: "
            f"{type(model_outputs)}"
        )

    if predicted_depth.ndim == 3:
        predicted_depth = predicted_depth.unsqueeze(1)

    elif predicted_depth.ndim != 4:
        raise ValueError(
            "Predicted depth must have shape "
            "[B, H, W] or [B, 1, H, W]. "
            f"Received: {tuple(predicted_depth.shape)}"
        )

    if predicted_depth.shape[1] != 1:
        raise ValueError(
            "Predicted depth must contain one channel. "
            f"Received shape: {tuple(predicted_depth.shape)}"
        )

    return predicted_depth


# ============================================================
# 3. 예측 depth 해상도 조정
# ============================================================

def resize_depth_prediction(
    predicted_depth: torch.Tensor,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Resize predicted depth to the requested spatial resolution.

    Parameters
    ----------
    predicted_depth : torch.Tensor
        Prediction with shape [B, 1, H, W].
    target_size : tuple[int, int]
        Target resolution as (height, width).
    """

    if predicted_depth.ndim != 4:
        raise ValueError(
            "predicted_depth must have shape [B, 1, H, W]."
        )

    if len(target_size) != 2:
        raise ValueError(
            "target_size must be provided as (height, width)."
        )

    target_size = (
        int(target_size[0]),
        int(target_size[1]),
    )

    if predicted_depth.shape[-2:] != target_size:
        predicted_depth = F.interpolate(
            predicted_depth,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

    return predicted_depth


# ============================================================
# 4. 이미지 배치 추론
# ============================================================

def predict_depth_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    image_processor: Any,
    device: torch.device = DEVICE,
    target_size: Optional[Tuple[int, int]] = None,
    clamp_depth: bool = False,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
) -> torch.Tensor:
    """
    Predict depth for an RGB image batch.

    This function does not modify the model's training state
    permanently. The original state is restored afterward.

    Returns
    -------
    torch.Tensor
        Predicted depth with shape [B, 1, H, W].
    """

    was_training = model.training
    model.eval()

    try:
        pixel_values = prepare_model_inputs(
            images=images,
            image_processor=image_processor,
            device=device,
        )

        with torch.no_grad():
            model_outputs = model(
                pixel_values=pixel_values
            )

            predicted_depth = extract_predicted_depth(
                model_outputs
            )

            if target_size is not None:
                predicted_depth = resize_depth_prediction(
                    predicted_depth=predicted_depth,
                    target_size=target_size,
                )

            if clamp_depth:
                predicted_depth = torch.clamp(
                    predicted_depth,
                    min=float(min_depth),
                    max=float(max_depth),
                )

    finally:
        if was_training:
            model.train()

    return predicted_depth


# ============================================================
# 5. Checkpoint 불러오기
# ============================================================

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Union[str, Path],
    device: torch.device = DEVICE,
    optimizer: Optional[
        torch.optim.Optimizer
    ] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Load a model checkpoint.

    If an optimizer is provided, its state is also restored.

    Returns
    -------
    dict
        Loaded checkpoint information.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain "
            "'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=strict,
    )

    model.to(device)

    if optimizer is not None:
        if "optimizer_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain "
                "'optimizer_state_dict'."
            )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # Optimizer 내부 tensor도 현재 device로 이동
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(device)

    return checkpoint
