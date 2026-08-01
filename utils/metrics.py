
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from .config import MAX_DEPTH, MIN_DEPTH


# ============================================================
# 1. Prediction, Target, Mask 정리
# ============================================================

def prepare_valid_depth_values(
    predicted_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    clamp_prediction: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract valid predicted and target depth values.

    Parameters
    ----------
    predicted_depth : torch.Tensor
        Predicted depth tensor.

    target_depth : torch.Tensor
        Ground-truth depth tensor.

    valid_mask : torch.Tensor or None
        Boolean mask indicating valid target pixels.
        If None, validity is determined from the target depth.

    min_depth : float
        Minimum valid depth.

    max_depth : float
        Maximum valid depth.

    clamp_prediction : bool
        Whether to clamp predicted depth to the evaluation range.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Flattened valid prediction and target values.
    """

    if predicted_depth.shape != target_depth.shape:
        raise ValueError(
            "Prediction and target shapes must match.\n"
            f"Prediction: {tuple(predicted_depth.shape)}\n"
            f"Target    : {tuple(target_depth.shape)}"
        )

    predicted_depth = predicted_depth.float()
    target_depth = target_depth.float()

    if valid_mask is None:
        valid_mask = torch.ones_like(
            target_depth,
            dtype=torch.bool,
        )

    else:
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

    depth_range_mask = (
        (target_depth > float(min_depth))
        & (target_depth <= float(max_depth))
    )

    final_mask = (
        valid_mask
        & finite_mask
        & depth_range_mask
    )

    if final_mask.sum().item() == 0:
        raise ValueError(
            "No valid depth pixels were found."
        )

    if clamp_prediction:
        predicted_depth = torch.clamp(
            predicted_depth,
            min=float(min_depth),
            max=float(max_depth),
        )

    valid_prediction = predicted_depth[
        final_mask
    ]

    valid_target = target_depth[
        final_mask
    ]

    return valid_prediction, valid_target


# ============================================================
# 2. 단일 배치 지표 계산
# ============================================================

def calculate_depth_metrics(
    predicted_depth: torch.Tensor,
    target_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    clamp_prediction: bool = True,
) -> Dict[str, float]:
    """
    Calculate depth-estimation metrics for one batch.

    Metrics
    -------
    mae
        Mean Absolute Error.

    rmse
        Root Mean Squared Error.

    abs_rel
        Mean absolute relative error.

    sq_rel
        Mean squared relative error.

    delta_1
        Percentage of pixels satisfying max(pred/gt, gt/pred) < 1.25.

    delta_2
        Percentage satisfying the threshold 1.25^2.

    delta_3
        Percentage satisfying the threshold 1.25^3.
    """

    prediction, target = prepare_valid_depth_values(
        predicted_depth=predicted_depth,
        target_depth=target_depth,
        valid_mask=valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
        clamp_prediction=clamp_prediction,
    )

    error = prediction - target
    absolute_error = torch.abs(error)
    squared_error = error ** 2

    mae = absolute_error.mean()
    rmse = torch.sqrt(squared_error.mean())

    abs_rel = (
        absolute_error / target
    ).mean()

    sq_rel = (
        squared_error / target
    ).mean()

    ratio = torch.maximum(
        prediction / target,
        target / prediction,
    )

    delta_1 = (
        ratio < 1.25
    ).float().mean()

    delta_2 = (
        ratio < (1.25 ** 2)
    ).float().mean()

    delta_3 = (
        ratio < (1.25 ** 3)
    ).float().mean()

    return {
        "valid_pixel_count": int(
            target.numel()
        ),
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "abs_rel": float(abs_rel.item()),
        "sq_rel": float(sq_rel.item()),
        "delta_1": float(delta_1.item()),
        "delta_2": float(delta_2.item()),
        "delta_3": float(delta_3.item()),
    }


# ============================================================
# 3. 전체 데이터셋용 Metric Accumulator
# ============================================================

@dataclass
class DepthMetricAccumulator:
    """
    Accumulate depth metrics over multiple batches.

    Metrics are computed using all valid pixels rather than
    averaging batch-level metric values.
    """

    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    absolute_relative_error_sum: float = 0.0
    squared_relative_error_sum: float = 0.0

    delta_1_count: int = 0
    delta_2_count: int = 0
    delta_3_count: int = 0

    valid_pixel_count: int = 0
    sample_count: int = 0

    def reset(self) -> None:
        """
        Reset all accumulated values.
        """

        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0

        self.absolute_relative_error_sum = 0.0
        self.squared_relative_error_sum = 0.0

        self.delta_1_count = 0
        self.delta_2_count = 0
        self.delta_3_count = 0

        self.valid_pixel_count = 0
        self.sample_count = 0

    @torch.no_grad()
    def update(
        self,
        predicted_depth: torch.Tensor,
        target_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        min_depth: float = MIN_DEPTH,
        max_depth: float = MAX_DEPTH,
        clamp_prediction: bool = True,
    ) -> None:
        """
        Update accumulated metric values using one batch.
        """

        prediction, target = prepare_valid_depth_values(
            predicted_depth=predicted_depth,
            target_depth=target_depth,
            valid_mask=valid_mask,
            min_depth=min_depth,
            max_depth=max_depth,
            clamp_prediction=clamp_prediction,
        )

        error = prediction - target
        absolute_error = torch.abs(error)
        squared_error = error ** 2

        ratio = torch.maximum(
            prediction / target,
            target / prediction,
        )

        self.absolute_error_sum += float(
            absolute_error.sum().item()
        )

        self.squared_error_sum += float(
            squared_error.sum().item()
        )

        self.absolute_relative_error_sum += float(
            (
                absolute_error / target
            ).sum().item()
        )

        self.squared_relative_error_sum += float(
            (
                squared_error / target
            ).sum().item()
        )

        self.delta_1_count += int(
            (
                ratio < 1.25
            ).sum().item()
        )

        self.delta_2_count += int(
            (
                ratio < (1.25 ** 2)
            ).sum().item()
        )

        self.delta_3_count += int(
            (
                ratio < (1.25 ** 3)
            ).sum().item()
        )

        self.valid_pixel_count += int(
            target.numel()
        )

        if target_depth.ndim == 0:
            batch_size = 1
        else:
            batch_size = int(
                target_depth.shape[0]
            )

        self.sample_count += batch_size

    def compute(self) -> Dict[str, float]:
        """
        Compute final metrics from accumulated values.
        """

        if self.valid_pixel_count == 0:
            raise RuntimeError(
                "No metric values have been accumulated."
            )

        pixel_count = float(
            self.valid_pixel_count
        )

        mae = (
            self.absolute_error_sum
            / pixel_count
        )

        rmse = (
            self.squared_error_sum
            / pixel_count
        ) ** 0.5

        abs_rel = (
            self.absolute_relative_error_sum
            / pixel_count
        )

        sq_rel = (
            self.squared_relative_error_sum
            / pixel_count
        )

        delta_1 = (
            self.delta_1_count
            / pixel_count
        )

        delta_2 = (
            self.delta_2_count
            / pixel_count
        )

        delta_3 = (
            self.delta_3_count
            / pixel_count
        )

        return {
            "sample_count": int(
                self.sample_count
            ),
            "valid_pixel_count": int(
                self.valid_pixel_count
            ),
            "mae": float(mae),
            "rmse": float(rmse),
            "abs_rel": float(abs_rel),
            "sq_rel": float(sq_rel),
            "delta_1": float(delta_1),
            "delta_2": float(delta_2),
            "delta_3": float(delta_3),
        }


# ============================================================
# 4. 지표 출력
# ============================================================

def print_depth_metrics(
    metrics: Dict[str, float],
    title: Optional[str] = None,
) -> None:
    """
    Print depth-estimation metrics in a readable format.
    """

    if title is not None:
        print(title)

    metric_order = [
        "sample_count",
        "valid_pixel_count",
        "mae",
        "rmse",
        "abs_rel",
        "sq_rel",
        "delta_1",
        "delta_2",
        "delta_3",
    ]

    for metric_name in metric_order:
        if metric_name not in metrics:
            continue

        metric_value = metrics[
            metric_name
        ]

        if metric_name in {
            "sample_count",
            "valid_pixel_count",
        }:
            print(
                f"{metric_name:<20}: "
                f"{int(metric_value)}"
            )

        else:
            print(
                f"{metric_name:<20}: "
                f"{float(metric_value):.6f}"
            )
