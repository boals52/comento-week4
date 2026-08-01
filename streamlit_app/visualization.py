
import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
)

from config import (
    HEATMAP_NAME,
    MAX_DEPTH,
    MIN_DEPTH,
)


def figure_to_pil(
    figure,
    dpi: int = 180,
) -> Image.Image:
    """
    Matplotlib Figure를 RGB PIL 이미지로 변환합니다.
    """

    buffer = io.BytesIO()

    figure.savefig(
        buffer,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(figure)

    buffer.seek(0)

    image = Image.open(
        buffer
    ).convert("RGB")

    image.load()
    buffer.close()

    return image


def depth_to_heatmap(
    depth_map: np.ndarray,
) -> Image.Image:
    """
    Metric depth map을 컬러바가 포함된
    Heatmap 이미지로 변환합니다.
    """

    depth_map = np.asarray(
        depth_map,
        dtype=np.float32,
    )

    if depth_map.ndim != 2:
        raise ValueError(
            "depth_map은 [H, W] 형태여야 합니다."
        )

    figure, axis = plt.subplots(
        figsize=(10, 5),
    )

    heatmap = axis.imshow(
        depth_map,
        cmap=HEATMAP_NAME,
        vmin=MIN_DEPTH,
        vmax=MAX_DEPTH,
        aspect="equal",
    )

    axis.set_title(
        "Fine-tuned Depth Anything V2"
    )

    axis.axis("off")

    colorbar = figure.colorbar(
        heatmap,
        ax=axis,
        fraction=0.035,
        pad=0.025,
    )

    colorbar.set_label(
        "Predicted depth (m)"
    )

    figure.tight_layout()

    return figure_to_pil(
        figure
    )


def load_label_font(
    image_width: int,
):
    """
    이미지 크기에 맞는 라벨 폰트를 불러옵니다.
    """

    font_size = max(
        12,
        min(
            24,
            int(image_width * 0.018),
        ),
    )

    font_candidates = [
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
        "DejaVuSans.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(
                font_path,
                size=font_size,
            )
        except OSError:
            continue

    return ImageFont.load_default()


def draw_detections_with_depth(
    image: Image.Image,
    detections,
) -> Image.Image:
    """
    입력 이미지 위에 다음 정보를 표시합니다.

    - YOLO bbox
    - 클래스 이름
    - YOLO confidence
    - bbox 중앙 영역의 median depth
    """

    result_image = image.convert(
        "RGB"
    ).copy()

    draw = ImageDraw.Draw(
        result_image
    )

    image_width, image_height = (
        result_image.size
    )

    font = load_label_font(
        image_width=image_width
    )

    box_width = max(
        2,
        int(image_width * 0.004),
    )

    for detection in detections:
        x1, y1, x2, y2 = [
            int(round(value))
            for value in detection["bbox"]
        ]

        # 이미지 범위를 벗어나지 않도록 보정
        x1 = max(
            0,
            min(x1, image_width - 1),
        )

        x2 = max(
            0,
            min(x2, image_width - 1),
        )

        y1 = max(
            0,
            min(y1, image_height - 1),
        )

        y2 = max(
            0,
            min(y2, image_height - 1),
        )

        if x2 <= x1 or y2 <= y1:
            continue

        class_name = str(
            detection["class_name"]
        )

        confidence = float(
            detection["confidence"]
        )

        depth = detection.get(
            "depth"
        )

        if depth is None:
            depth_text = "N/A"
        else:
            depth_text = (
                f"{float(depth):.2f} m"
            )

        label = (
            f"{class_name} | "
            f"{confidence:.2f} | "
            f"{depth_text}"
        )

        box_color = (
            255,
            64,
            64,
        )

        text_color = (
            255,
            255,
            255,
        )

        draw.rectangle(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            outline=box_color,
            width=box_width,
        )

        text_bbox = draw.textbbox(
            (0, 0),
            label,
            font=font,
        )

        text_width = (
            text_bbox[2]
            - text_bbox[0]
        )

        text_height = (
            text_bbox[3]
            - text_bbox[1]
        )

        padding = 5

        # bbox 위쪽에 공간이 없으면 안쪽에 표시
        label_top = (
            y1
            - text_height
            - padding * 2
        )

        if label_top < 0:
            label_top = y1

        label_bottom = (
            label_top
            + text_height
            + padding * 2
        )

        label_right = min(
            image_width - 1,
            x1
            + text_width
            + padding * 2,
        )

        draw.rectangle(
            [
                x1,
                label_top,
                label_right,
                label_bottom,
            ],
            fill=box_color,
        )

        draw.text(
            (
                x1 + padding,
                label_top + padding,
            ),
            label,
            fill=text_color,
            font=font,
        )

    return result_image


def create_depth_histogram(
    depth_map: np.ndarray,
    bins: int = 40,
) -> Image.Image:
    """
    예측된 metric depth의 분포를 Histogram으로 생성합니다.
    """

    depth_map = np.asarray(
        depth_map,
        dtype=np.float32,
    )

    if depth_map.ndim != 2:
        raise ValueError(
            "depth_map은 [H, W] 형태여야 합니다."
        )

    valid_mask = (
        np.isfinite(depth_map)
        & (depth_map >= MIN_DEPTH)
        & (depth_map <= MAX_DEPTH)
    )

    valid_depth = depth_map[
        valid_mask
    ]

    if valid_depth.size == 0:
        raise ValueError(
            "Histogram을 생성할 유효 깊이값이 없습니다."
        )

    figure, axis = plt.subplots(
        figsize=(9, 4.5),
    )

    axis.hist(
        valid_depth,
        bins=bins,
        range=(
            MIN_DEPTH,
            MAX_DEPTH,
        ),
    )

    median_depth = float(
        np.median(valid_depth)
    )

    axis.axvline(
        median_depth,
        linestyle="--",
        linewidth=2,
        label=(
            f"Median: "
            f"{median_depth:.2f} m"
        ),
    )

    axis.set_xlabel(
        "Predicted depth (m)"
    )

    axis.set_ylabel(
        "Pixel count"
    )

    axis.set_title(
        "Predicted Depth Distribution"
    )

    axis.set_xlim(
        MIN_DEPTH,
        MAX_DEPTH,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    axis.legend()

    figure.tight_layout()

    return figure_to_pil(
        figure
    )
