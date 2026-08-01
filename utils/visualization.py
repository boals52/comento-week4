
from typing import Dict, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


# ============================================================
# 1. Tensor -> NumPy 변환
# ============================================================

def tensor_to_numpy(
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    Convert a PyTorch tensor into a NumPy array.
    """

    if not torch.is_tensor(tensor):
        raise TypeError(
            "tensor must be a torch.Tensor. "
            f"Received: {type(tensor)}"
        )

    return (
        tensor
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================
# 2. RGB 이미지 정리
# ============================================================

def prepare_rgb_image(
    image: torch.Tensor,
) -> np.ndarray:
    """
    Convert an RGB tensor into an HWC NumPy image.

    Accepted input shapes
    ---------------------
    [3, H, W]
    [1, 3, H, W]
    """

    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError(
                "A batched image must contain exactly "
                "one sample."
            )

        image = image[0]

    if image.ndim != 3:
        raise ValueError(
            "image must have shape [3, H, W] "
            "or [1, 3, H, W]. "
            f"Received: {tuple(image.shape)}"
        )

    if image.shape[0] != 3:
        raise ValueError(
            "image must contain three RGB channels. "
            f"Received: {tuple(image.shape)}"
        )

    image = tensor_to_numpy(image)

    image = np.transpose(
        image,
        (1, 2, 0),
    )

    image = np.clip(
        image,
        0.0,
        1.0,
    )

    return image


# ============================================================
# 3. Depth map 정리
# ============================================================

def prepare_depth_map(
    depth: torch.Tensor,
) -> np.ndarray:
    """
    Convert a depth tensor into a two-dimensional NumPy array.

    Accepted input shapes
    ---------------------
    [H, W]
    [1, H, W]
    [1, 1, H, W]
    """

    if depth.ndim == 4:
        if (
            depth.shape[0] != 1
            or depth.shape[1] != 1
        ):
            raise ValueError(
                "A batched depth map must have shape "
                "[1, 1, H, W]."
            )

        depth = depth[0, 0]

    elif depth.ndim == 3:
        if depth.shape[0] != 1:
            raise ValueError(
                "A three-dimensional depth map must "
                "have shape [1, H, W]."
            )

        depth = depth[0]

    elif depth.ndim != 2:
        raise ValueError(
            "depth must have shape [H, W], "
            "[1, H, W], or [1, 1, H, W]. "
            f"Received: {tuple(depth.shape)}"
        )

    return tensor_to_numpy(depth)


# ============================================================
# 4. 단일 depth map 표시
# ============================================================

def plot_depth_map(
    depth: torch.Tensor,
    title: str = "Depth Map",
    colorbar_label: str = "Depth",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """
    Display one depth map.
    """

    depth_array = prepare_depth_map(
        depth
    )

    plt.figure(
        figsize=figsize
    )

    image_handle = plt.imshow(
        depth_array,
        vmin=vmin,
        vmax=vmax,
    )

    plt.title(title)
    plt.axis("off")

    colorbar = plt.colorbar(
        image_handle
    )

    colorbar.set_label(
        colorbar_label
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 5. RGB / GT / Prediction / Error 비교
# ============================================================

def plot_depth_comparison(
    image: torch.Tensor,
    target_depth: torch.Tensor,
    predicted_depth: torch.Tensor,
    valid_mask: Optional[
        torch.Tensor
    ] = None,
    title: Optional[str] = None,
    max_depth: Optional[float] = None,
    figsize: tuple[int, int] = (18, 5),
) -> None:
    """
    Display RGB image, ground-truth depth, predicted depth,
    and absolute error map.
    """

    rgb_array = prepare_rgb_image(
        image
    )

    target_array = prepare_depth_map(
        target_depth
    )

    prediction_array = prepare_depth_map(
        predicted_depth
    )

    if target_array.shape != prediction_array.shape:
        raise ValueError(
            "Target and prediction shapes must match.\n"
            f"Target    : {target_array.shape}\n"
            f"Prediction: {prediction_array.shape}"
        )

    if valid_mask is not None:
        mask_array = prepare_depth_map(
            valid_mask
        ).astype(bool)

        if mask_array.shape != target_array.shape:
            raise ValueError(
                "Mask and target shapes must match."
            )

    else:
        mask_array = np.isfinite(
            target_array
        ) & (
            target_array > 0
        )

    target_display = np.where(
        mask_array,
        target_array,
        np.nan,
    )

    prediction_display = np.where(
        mask_array,
        prediction_array,
        np.nan,
    )

    error_array = np.where(
        mask_array,
        np.abs(
            prediction_array
            - target_array
        ),
        np.nan,
    )

    if max_depth is None:
        valid_target_values = (
            target_array[
                mask_array
            ]
        )

        if valid_target_values.size > 0:
            depth_max = float(
                np.nanpercentile(
                    valid_target_values,
                    99,
                )
            )
        else:
            depth_max = None

    else:
        depth_max = float(
            max_depth
        )

    figure = plt.figure(
        figsize=figsize
    )

    if title is not None:
        figure.suptitle(
            title
        )

    plt.subplot(
        1,
        4,
        1,
    )

    plt.imshow(
        rgb_array
    )

    plt.title("RGB Image")
    plt.axis("off")

    plt.subplot(
        1,
        4,
        2,
    )

    ground_truth_handle = plt.imshow(
        target_display,
        vmin=0,
        vmax=depth_max,
    )

    plt.title("Ground Truth")
    plt.axis("off")
    plt.colorbar(
        ground_truth_handle,
        fraction=0.046,
        pad=0.04,
    )

    plt.subplot(
        1,
        4,
        3,
    )

    prediction_handle = plt.imshow(
        prediction_display,
        vmin=0,
        vmax=depth_max,
    )

    plt.title("Prediction")
    plt.axis("off")
    plt.colorbar(
        prediction_handle,
        fraction=0.046,
        pad=0.04,
    )

    plt.subplot(
        1,
        4,
        4,
    )

    error_handle = plt.imshow(
        error_array
    )

    plt.title("Absolute Error")
    plt.axis("off")
    plt.colorbar(
        error_handle,
        fraction=0.046,
        pad=0.04,
    )

    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Baseline과 Fine-tuned 결과 비교
# ============================================================

def plot_model_comparison(
    image: torch.Tensor,
    target_depth: torch.Tensor,
    baseline_depth: torch.Tensor,
    finetuned_depth: torch.Tensor,
    valid_mask: Optional[
        torch.Tensor
    ] = None,
    title: Optional[str] = None,
    max_depth: Optional[float] = None,
    figsize: tuple[int, int] = (20, 8),
    save_path=None
) -> None:
    """
    Compare baseline and fine-tuned depth predictions.
    """

    rgb_array = prepare_rgb_image(
        image
    )

    target_array = prepare_depth_map(
        target_depth
    )

    baseline_array = prepare_depth_map(
        baseline_depth
    )

    finetuned_array = prepare_depth_map(
        finetuned_depth
    )

    if not (
        target_array.shape
        == baseline_array.shape
        == finetuned_array.shape
    ):
        raise ValueError(
            "Target, baseline, and fine-tuned "
            "depth shapes must match."
        )

    if valid_mask is not None:
        mask_array = prepare_depth_map(
            valid_mask
        ).astype(bool)

    else:
        mask_array = np.isfinite(
            target_array
        ) & (
            target_array > 0
        )

    target_display = np.where(
        mask_array,
        target_array,
        np.nan,
    )

    baseline_display = np.where(
        mask_array,
        baseline_array,
        np.nan,
    )

    finetuned_display = np.where(
        mask_array,
        finetuned_array,
        np.nan,
    )

    baseline_error = np.where(
        mask_array,
        np.abs(
            baseline_array
            - target_array
        ),
        np.nan,
    )

    finetuned_error = np.where(
        mask_array,
        np.abs(
            finetuned_array
            - target_array
        ),
        np.nan,
    )

    if max_depth is None:
        valid_target_values = (
            target_array[
                mask_array
            ]
        )

        if valid_target_values.size > 0:
            depth_max = float(
                np.nanpercentile(
                    valid_target_values,
                    99,
                )
            )
        else:
            depth_max = None

    else:
        depth_max = float(
            max_depth
        )

    figure = plt.figure(
        figsize=figsize
    )

    if title is not None:
        figure.suptitle(
            title
        )

    plot_items = [
        (
            rgb_array,
            "RGB Image",
            None,
            None,
        ),
        (
            target_display,
            "Ground Truth",
            0,
            depth_max,
        ),
        (
            baseline_display,
            "Baseline Prediction",
            0,
            depth_max,
        ),
        (
            finetuned_display,
            "Fine-tuned Prediction",
            0,
            depth_max,
        ),
        (
            baseline_error,
            "Baseline Error",
            None,
            None,
        ),
        (
            finetuned_error,
            "Fine-tuned Error",
            None,
            None,
        ),
    ]

    for index, (
        array,
        subplot_title,
        vmin,
        vmax,
    ) in enumerate(
        plot_items,
        start=1,
    ):
        plt.subplot(
            2,
            3,
            index,
        )

        image_handle = plt.imshow(
            array,
            vmin=vmin,
            vmax=vmax,
        )

        plt.title(
            subplot_title
        )

        plt.axis("off")

        if index != 1:
            plt.colorbar(
                image_handle,
                fraction=0.046,
                pad=0.04,
            )

    plt.tight_layout()

    if save_path is not None:
        figure.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
      )

    plt.show()


# ============================================================
# 7. 학습 곡선 표시
# ============================================================

def plot_training_history(
    history: Dict[
        str,
        Sequence[float]
    ],
    train_key: str = "train_loss",
    val_key: str = "val_loss",
    title: str = "Training History",
    figsize: tuple[int, int] = (8, 5),
) -> None:
    """
    Plot train and validation loss histories.
    """

    if train_key not in history:
        raise KeyError(
            f"History does not contain '{train_key}'."
        )

    train_values = list(
        history[train_key]
    )

    if len(train_values) == 0:
        raise ValueError(
            "Training history is empty."
        )

    epochs = np.arange(
        1,
        len(train_values) + 1,
    )

    plt.figure(
        figsize=figsize
    )

    plt.plot(
        epochs,
        train_values,
        marker="o",
        label="Train Loss",
    )

    if val_key in history:
        val_values = list(
            history[val_key]
        )

        if len(val_values) != len(
            train_values
        ):
            raise ValueError(
                "Train and validation history lengths "
                "must match."
            )

        plt.plot(
            epochs,
            val_values,
            marker="o",
            label="Validation Loss",
        )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
