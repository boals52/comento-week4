
import sys

import streamlit as st
import torch

from ultralytics import YOLO

from config import (
    CHECKPOINT_PATH,
    PROJECT_ROOT,
    YOLO_CHECKPOINT_PATH,
)


if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


import utils.config as project_config
import utils.initialization as init_utils
import utils.inference as inference_utils


@st.cache_resource(
    show_spinner="Fine-tuned Depth Anything V2 모델을 불러오는 중입니다..."
)
def load_finetuned_depth_model():
    """
    Fine-tuned Depth Anything V2 모델과
    image processor를 한 번만 로드합니다.
    """

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint를 찾을 수 없습니다: "
            f"{CHECKPOINT_PATH}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, image_processor = (
        init_utils.load_depth_model(
            model_name=(
                project_config.METRIC_MODEL_NAME
            ),
            device=device,
        )
    )

    checkpoint = (
        inference_utils.load_checkpoint(
            model=model,
            checkpoint_path=CHECKPOINT_PATH,
            device=device,
            optimizer=None,
            strict=True,
        )
    )

    model.eval()

    return {
        "model": model,
        "image_processor": image_processor,
        "device": device,
        "checkpoint": checkpoint,
    }

@st.cache_resource(
    show_spinner="YOLOv8n 모델을 불러오는 중입니다..."
)
def load_yolo_model():
    if not YOLO_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "YOLO checkpoint를 찾을 수 없습니다: "
            f"{YOLO_CHECKPOINT_PATH}"
        )

    model = YOLO(
        str(YOLO_CHECKPOINT_PATH)
    )

    return model
