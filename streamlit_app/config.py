
from pathlib import Path

# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/2026/comento/week4"
)

UTILS_ROOT = (
    PROJECT_ROOT
    / "utils"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "finetuning"
    / "best_model.pt"
)

# ------------------------------------------------------------
# Depth model settings
# ------------------------------------------------------------

MIN_DEPTH = 0.1
MAX_DEPTH = 80.0

HEATMAP_NAME = "inferno"

YOLO_CHECKPOINT_PATH = Path(
    "/content/drive/MyDrive/2026/comento/week3/"
    "runs/exp3_yolov8n_50epochs/weights/best.pt"
)

DEFAULT_YOLO_CONFIDENCE = 0.50