
from pathlib import Path

import torch


# ============================================================
# 1. 프로젝트 경로
# ============================================================

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/2026/comento/week4"
)

DATASET_ROOT = Path(
    "/content/kitti_eigen/kitti_eigen"
)

SPLIT_ROOT = (
    PROJECT_ROOT
    / "splits"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
)

CHECKPOINT_ROOT = (
    PROJECT_ROOT
    / "checkpoints"
)

LOG_ROOT = (
    PROJECT_ROOT
    / "logs"
)


# ============================================================
# 2. 데이터셋 파일
# ============================================================

SPLIT_JSON_PATH = (
    SPLIT_ROOT
    / "kitti_subset_splits.json"
)


# ============================================================
# 3. 모델 설정
# ============================================================

RELATIVE_MODEL_NAME = (
    "depth-anything/"
    "Depth-Anything-V2-Small-hf"
)

METRIC_MODEL_NAME = (
    "depth-anything/"
    "Depth-Anything-V2-Metric-Outdoor-Small-hf"
)


# ============================================================
# 4. Depth 범위
# ============================================================

MIN_DEPTH = 1e-3

MAX_DEPTH = 80.0


# ============================================================
# 5. 입력 이미지 크기
# ============================================================

IMAGE_HEIGHT = 518

IMAGE_WIDTH = 518

IMAGE_SIZE = (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


# ============================================================
# 6. DataLoader 설정
# ============================================================

TRAIN_BATCH_SIZE = 2

VAL_BATCH_SIZE = 2

TEST_BATCH_SIZE = 2

NUM_WORKERS = 2

PIN_MEMORY = torch.cuda.is_available()


# ============================================================
# 7. 재현성 설정
# ============================================================

RANDOM_SEED = 42


# ============================================================
# 8. 학습 기본 설정
# ============================================================

NUM_EPOCHS = 10

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-2

GRADIENT_CLIP_NORM = 1.0

EARLY_STOPPING_PATIENCE = 3


# ============================================================
# 9. Device
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 10. 출력 폴더 생성
# ============================================================

def create_project_directories():
    """
    프로젝트에서 사용하는 출력 폴더가 없으면 생성합니다.
    """

    directories = [
        DATASET_ROOT,
        SPLIT_ROOT,
        OUTPUT_ROOT,
        CHECKPOINT_ROOT,
        LOG_ROOT,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# Colab 로컬 데이터 복원 설정
# ============================================================

DATASET_ARCHIVE_PATH = (
    PROJECT_ROOT
    / "dataset"
    / "kitti_eigen.tar"
)

LOCAL_EXTRACT_ROOT = Path(
    "/content/kitti_eigen"
)

DATASET_ROOT = (
    LOCAL_EXTRACT_ROOT
    / "kitti_eigen"
)
