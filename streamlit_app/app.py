
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from config import (
    DEFAULT_YOLO_CONFIDENCE,
    MAX_DEPTH,
    MIN_DEPTH,
)

from inference import (
    attach_depth_to_detections,
    predict_metric_depth,
    predict_objects,
)

from model_loader import (
    load_finetuned_depth_model,
    load_yolo_model,
)

from visualization import (
    create_depth_histogram,
    depth_to_heatmap,
    draw_detections_with_depth,
)


st.set_page_config(
    page_title=(
        "Object-aware Depth Estimation"
    ),
    page_icon="🚗",
    layout="wide",
)


st.title(
    "Object-aware Metric Depth Estimation"
)

st.caption(
    "Fine-tuned Depth Anything V2와 "
    "YOLOv8n을 결합하여 객체별 예상 거리를 "
    "시각화합니다."
)


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "inference_result" not in st.session_state:
    st.session_state[
        "inference_result"
    ] = None

if "uploaded_file_key" not in st.session_state:
    st.session_state[
        "uploaded_file_key"
    ] = None


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header(
        "Inference Settings"
    )

    confidence_threshold = st.slider(
        "YOLO confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(
            DEFAULT_YOLO_CONFIDENCE
        ),
        step=0.05,
        help=(
            "이 값보다 신뢰도가 낮은 객체는 "
            "표시하지 않습니다."
        ),
    )

    center_ratio = st.slider(
        "Depth sampling region",
        min_value=0.20,
        max_value=1.00,
        value=0.50,
        step=0.10,
        help=(
            "bbox 중심부 중 거리 계산에 사용할 "
            "영역의 비율입니다."
        ),
    )

    st.divider()

    st.subheader(
        "Model Information"
    )

    st.write(
        "Object detection: YOLOv8n"
    )

    st.write(
        "Depth estimation: "
        "Fine-tuned Depth Anything V2"
    )

    st.write(
        f"Depth range: "
        f"{MIN_DEPTH:.1f}–"
        f"{MAX_DEPTH:.1f} m"
    )

    st.info(
        "객체 거리는 bbox 중앙 영역에 해당하는 "
        "depth 값의 중앙값으로 계산됩니다."
    )


# ------------------------------------------------------------
# File uploader
# ------------------------------------------------------------

uploaded_file = st.file_uploader(
    "분석할 이미지를 업로드하세요.",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
    accept_multiple_files=False,
)


if uploaded_file is None:
    st.info(
        "JPG, PNG 또는 WebP 이미지를 "
        "업로드하세요."
    )

    st.stop()


try:
    input_image = Image.open(
        uploaded_file
    ).convert("RGB")

except Exception as error:
    st.error(
        f"이미지를 읽을 수 없습니다: {error}"
    )
    st.stop()


# 새 이미지가 업로드되면 이전 결과 제거
current_file_key = (
    uploaded_file.name,
    uploaded_file.size,
)

if (
    st.session_state[
        "uploaded_file_key"
    ]
    != current_file_key
):
    st.session_state[
        "uploaded_file_key"
    ] = current_file_key

    st.session_state[
        "inference_result"
    ] = None


st.subheader(
    "Uploaded Image"
)

st.image(
    input_image,
    caption=(
        f"{uploaded_file.name} · "
        f"{input_image.width}×"
        f"{input_image.height}"
    ),
    use_container_width=True,
)


run_inference = st.button(
    "객체 탐지 및 Depth 추론 실행",
    type="primary",
    use_container_width=True,
)


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------

if run_inference:
    try:
        total_start_time = (
            time.perf_counter()
        )

        with st.spinner(
            "Depth Anything V2와 YOLOv8n "
            "모델을 불러오는 중입니다..."
        ):
            depth_resources = (
                load_finetuned_depth_model()
            )

            yolo_model = (
                load_yolo_model()
            )

        depth_start_time = (
            time.perf_counter()
        )

        with st.spinner(
            "Metric depth를 추론하는 중입니다..."
        ):
            depth_map = predict_metric_depth(
                image=input_image,
                model=depth_resources[
                    "model"
                ],
                image_processor=(
                    depth_resources[
                        "image_processor"
                    ]
                ),
                device=depth_resources[
                    "device"
                ],
            )

        depth_seconds = (
            time.perf_counter()
            - depth_start_time
        )

        yolo_start_time = (
            time.perf_counter()
        )

        with st.spinner(
            "객체를 탐지하는 중입니다..."
        ):
            detections = predict_objects(
                image=input_image,
                model=yolo_model,
                confidence_threshold=(
                    confidence_threshold
                ),
            )

        yolo_seconds = (
            time.perf_counter()
            - yolo_start_time
        )

        # center_ratio 설정을 적용해
        # 객체별 depth를 직접 계산
        detections_with_depth = (
        attach_depth_to_detections(
        detections=detections,
        depth_map=depth_map,
        center_ratio=center_ratio,
    )
)

        heatmap_image = (
            depth_to_heatmap(
                depth_map
            )
        )

        detection_image = (
            draw_detections_with_depth(
                image=input_image,
                detections=(
                    detections_with_depth
                ),
            )
        )

        histogram_image = (
            create_depth_histogram(
                depth_map
            )
        )

        total_seconds = (
            time.perf_counter()
            - total_start_time
        )

        st.session_state[
            "inference_result"
        ] = {
            "input_image": input_image,
            "depth_map": depth_map,
            "heatmap_image": heatmap_image,
            "detection_image": (
                detection_image
            ),
            "histogram_image": (
                histogram_image
            ),
            "detections": (
                detections_with_depth
            ),
            "depth_seconds": (
                depth_seconds
            ),
            "yolo_seconds": (
                yolo_seconds
            ),
            "total_seconds": (
                total_seconds
            ),
            "confidence_threshold": (
                confidence_threshold
            ),
            "center_ratio": center_ratio,
        }

    except Exception as error:
        st.exception(error)


# ------------------------------------------------------------
# Result display
# ------------------------------------------------------------

result = st.session_state[
    "inference_result"
]


if result is not None:
    st.success(
        f"추론 완료 · "
        f"총 {result['total_seconds']:.2f}초"
    )

    st.header(
        "Inference Results"
    )

    original_column, depth_column, detection_column = (
        st.columns(3)
    )

    with original_column:
        st.subheader(
            "1. Original"
        )

        st.image(
            result["input_image"],
            use_container_width=True,
        )

    with depth_column:
        st.subheader(
            "2. Depth Heatmap"
        )

        st.image(
            result["heatmap_image"],
            use_container_width=True,
        )

    with detection_column:
        st.subheader(
            "3. Detection + Depth"
        )

        st.image(
            result[
                "detection_image"
            ],
            use_container_width=True,
        )

    depth_map = result[
        "depth_map"
    ]

    finite_depth = depth_map[
        np.isfinite(depth_map)
    ]

    st.subheader(
        "Depth Statistics"
    )

    metric_1, metric_2, metric_3, metric_4 = (
        st.columns(4)
    )

    metric_1.metric(
        "Minimum depth",
        f"{finite_depth.min():.2f} m",
    )

    metric_2.metric(
        "Median depth",
        f"{np.median(finite_depth):.2f} m",
    )

    metric_3.metric(
        "Maximum depth",
        f"{finite_depth.max():.2f} m",
    )

    metric_4.metric(
        "Detected objects",
        len(
            result["detections"]
        ),
    )

    st.subheader(
        "Detected Object Details"
    )

    detections = result[
        "detections"
    ]

    if len(detections) == 0:
        st.warning(
            "현재 confidence threshold에서 "
            "탐지된 객체가 없습니다."
        )

    else:
        table_rows = []

        for object_index, detection in enumerate(
            detections,
            start=1,
        ):
            depth = detection.get(
                "depth"
            )

            if depth is None:
                depth_value = np.nan
            else:
                depth_value = float(
                    depth
                )

            x1, y1, x2, y2 = (
                detection["bbox"]
            )

            table_rows.append(
                {
                    "No.": object_index,
                    "Class": detection[
                        "class_name"
                    ],
                    "Confidence": float(
                        detection[
                            "confidence"
                        ]
                    ),
                    "Estimated Depth (m)": (
                        depth_value
                    ),
                    "BBox": (
                        f"({x1:.0f}, {y1:.0f}) – "
                        f"({x2:.0f}, {y2:.0f})"
                    ),
                }
            )

        object_table = pd.DataFrame(
            table_rows
        )

        st.dataframe(
            object_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence": (
                    st.column_config.NumberColumn(
                        format="%.3f",
                    )
                ),
                "Estimated Depth (m)": (
                    st.column_config.NumberColumn(
                        format="%.2f m",
                    )
                ),
            },
        )

    st.subheader(
        "Depth Distribution"
    )

    st.image(
        result["histogram_image"],
        use_container_width=True,
    )

    st.subheader(
        "Inference Time"
    )

    time_column_1, time_column_2, time_column_3 = (
        st.columns(3)
    )

    time_column_1.metric(
        "Depth estimation",
        f"{result['depth_seconds']:.2f} sec",
    )

    time_column_2.metric(
        "Object detection",
        f"{result['yolo_seconds']:.2f} sec",
    )

    time_column_3.metric(
        "Total processing",
        f"{result['total_seconds']:.2f} sec",
    )

    st.caption(
        "객체 거리는 fine-tuned Depth Anything V2의 "
        "예측값 중 YOLO bbox 중앙 영역의 median으로 "
        "계산한 추정값입니다."
    )
