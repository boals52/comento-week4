
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.optim import AdamW
from tqdm.auto import tqdm

from .config import (
    CHECKPOINT_ROOT,
    DEVICE,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from .inference import (
    extract_predicted_depth,
    prepare_model_inputs,
    resize_depth_prediction,
)


# ============================================================
# 1. Valid pixel 기반 L1 loss
# ============================================================

def masked_l1_loss(
    predicted_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mean absolute error over valid depth pixels only.

    Parameters
    ----------
    predicted_depth : torch.Tensor
        Predicted metric depth with shape [B, 1, H, W].

    target_depth : torch.Tensor
        Ground-truth metric depth with shape [B, 1, H, W].

    valid_mask : torch.Tensor
        Boolean validity mask with shape [B, 1, H, W].

    Returns
    -------
    torch.Tensor
        Scalar masked L1 loss.
    """

    if predicted_depth.shape != target_depth.shape:
        raise ValueError(
            "Prediction and target shapes must match.\n"
            f"Prediction: {tuple(predicted_depth.shape)}\n"
            f"Target    : {tuple(target_depth.shape)}"
        )

    if valid_mask.shape != target_depth.shape:
        raise ValueError(
            "Mask and target shapes must match.\n"
            f"Mask  : {tuple(valid_mask.shape)}\n"
            f"Target: {tuple(target_depth.shape)}"
        )

    valid_mask = valid_mask.bool()

    finite_mask = (
        torch.isfinite(predicted_depth)
        & torch.isfinite(target_depth)
    )

    final_mask = valid_mask & finite_mask

    valid_pixel_count = int(
        final_mask.sum().item()
    )

    if valid_pixel_count == 0:
        raise ValueError(
            "The current batch contains no valid depth pixels."
        )

    absolute_error = torch.abs(
        predicted_depth - target_depth
    )

    return absolute_error[final_mask].mean()


# ============================================================
# 2. Optimizer 생성
# ============================================================

def create_optimizer(
    model: torch.nn.Module,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> torch.optim.Optimizer:
    """
    Create an AdamW optimizer using trainable parameters only.
    """

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise ValueError(
            "The model has no trainable parameters."
        )

    return AdamW(
        trainable_parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )


# ============================================================
# 3. 학습용 batch forward
# ============================================================

def forward_depth_batch(
    model: torch.nn.Module,
    batch: Dict[str, Any],
    image_processor: Any,
    device: torch.device = DEVICE,
) -> Dict[str, torch.Tensor]:
    """
    Run one training-compatible forward pass.

    Unlike predict_depth_batch(), this function does not disable
    gradient calculation or change the model's training state.

    Returns
    -------
    dict
        pixel_values, predicted_depth, target_depth, valid_mask
    """

    required_keys = {
        "image",
        "depth",
        "valid_mask",
    }

    missing_keys = required_keys - set(
        batch.keys()
    )

    if missing_keys:
        raise KeyError(
            "Batch is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    target_depth = batch["depth"].to(
        device,
        non_blocking=True,
    )

    valid_mask = batch["valid_mask"].to(
        device,
        non_blocking=True,
    )

    pixel_values = prepare_model_inputs(
        images=batch["image"],
        image_processor=image_processor,
        device=device,
    )

    model_outputs = model(
        pixel_values=pixel_values
    )

    predicted_depth = extract_predicted_depth(
        model_outputs
    )

    predicted_depth = resize_depth_prediction(
        predicted_depth=predicted_depth,
        target_size=target_depth.shape[-2:],
    )

    return {
        "pixel_values": pixel_values,
        "predicted_depth": predicted_depth,
        "target_depth": target_depth,
        "valid_mask": valid_mask,
    }


# ============================================================
# 4. 한 epoch 학습
# ============================================================

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    image_processor: Any,
    device: torch.device = DEVICE,
    gradient_clip_norm: Optional[
        float
    ] = GRADIENT_CLIP_NORM,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    max_batches can be used for a short smoke test.
    """

    model.train()

    # Frozen backbone은 학습 중에도 evaluation mode로 유지
if hasattr(model, "backbone"):
    backbone_is_frozen = not any(
        parameter.requires_grad
        for parameter in model.backbone.parameters()
    )

    if backbone_is_frozen:
        model.backbone.eval()

    total_loss = 0.0
    total_valid_pixels = 0
    processed_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc="Training",
        leave=False,
    )

    for batch_index, batch in enumerate(
        progress_bar
    ):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        optimizer.zero_grad(
            set_to_none=True
        )

        forward_result = forward_depth_batch(
            model=model,
            batch=batch,
            image_processor=image_processor,
            device=device,
        )

        loss = masked_l1_loss(
            predicted_depth=forward_result[
                "predicted_depth"
            ],
            target_depth=forward_result[
                "target_depth"
            ],
            valid_mask=forward_result[
                "valid_mask"
            ],
        )

        loss.backward()

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                max_norm=float(
                    gradient_clip_norm
                ),
            )

        optimizer.step()

        batch_loss = float(
            loss.detach().item()
        )

        current_valid_pixels = int(
            forward_result[
                "valid_mask"
            ].sum().item()
        )

        total_loss += batch_loss
        total_valid_pixels += (
            current_valid_pixels
        )
        processed_batches += 1

        progress_bar.set_postfix(
            loss=f"{batch_loss:.4f}"
        )

    if processed_batches == 0:
        raise RuntimeError(
            "No training batches were processed."
        )

    return {
        "loss": total_loss / processed_batches,
        "batch_count": processed_batches,
        "valid_pixel_count": (
            total_valid_pixels
        ),
    }


# ============================================================
# 5. 한 epoch 검증
# ============================================================

@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    image_processor: Any,
    device: torch.device = DEVICE,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Calculate validation loss for one epoch.
    """

    model.eval()

    total_loss = 0.0
    total_valid_pixels = 0
    processed_batches = 0

    progress_bar = tqdm(
        dataloader,
        desc="Validation",
        leave=False,
    )

    for batch_index, batch in enumerate(
        progress_bar
    ):
        if (
            max_batches is not None
            and batch_index >= max_batches
        ):
            break

        forward_result = forward_depth_batch(
            model=model,
            batch=batch,
            image_processor=image_processor,
            device=device,
        )

        loss = masked_l1_loss(
            predicted_depth=forward_result[
                "predicted_depth"
            ],
            target_depth=forward_result[
                "target_depth"
            ],
            valid_mask=forward_result[
                "valid_mask"
            ],
        )

        batch_loss = float(
            loss.item()
        )

        current_valid_pixels = int(
            forward_result[
                "valid_mask"
            ].sum().item()
        )

        total_loss += batch_loss
        total_valid_pixels += (
            current_valid_pixels
        )
        processed_batches += 1

        progress_bar.set_postfix(
            loss=f"{batch_loss:.4f}"
        )

    if processed_batches == 0:
        raise RuntimeError(
            "No validation batches were processed."
        )

    return {
        "loss": total_loss / processed_batches,
        "batch_count": processed_batches,
        "valid_pixel_count": (
            total_valid_pixels
        ),
    }


# ============================================================
# 6. Checkpoint 저장
# ============================================================

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    checkpoint_path: Optional[
        Path
    ] = None,
    scheduler: Optional[Any] = None,
    extra_state: Optional[
        Dict[str, Any]
    ] = None,
) -> Path:
    """
    Save model, optimizer, and optional scheduler states.
    """

    if checkpoint_path is None:
        checkpoint_path = (
            CHECKPOINT_ROOT
            / f"depth_anything_epoch_{epoch:03d}.pt"
        )

    checkpoint_path = Path(
        checkpoint_path
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = (
            scheduler.state_dict()
        )

    if extra_state is not None:
        checkpoint["extra_state"] = (
            extra_state
        )

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    return checkpoint_path
