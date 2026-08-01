
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


PathLike = Union[str, Path]
PairType = Dict[str, str]


# ============================================================
# 1. Split JSON 로드
# ============================================================

def load_split_json(
    split_json_path: PathLike,
) -> Dict[str, Any]:
    """
    Load a dataset split JSON file.

    Parameters
    ----------
    split_json_path : str or pathlib.Path
        Path to the JSON file containing train, validation,
        and test split information.

    Returns
    -------
    dict
        Parsed split information.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If the loaded JSON object is not a dictionary.
    """

    split_json_path = Path(split_json_path)

    if not split_json_path.exists():
        raise FileNotFoundError(
            f"Split JSON file was not found: "
            f"{split_json_path}"
        )

    with split_json_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        split_data = json.load(file)

    if not isinstance(split_data, dict):
        raise ValueError(
            "The split JSON root must be a dictionary."
        )

    return split_data


# ============================================================
# 2. Split 이름 조회
# ============================================================

def get_split_entries(
    split_data: Mapping[str, Any],
    split_name: str,
) -> Sequence[Any]:
    """
    Retrieve entries for a requested split.

    The function accepts common validation split names such as
    'val', 'valid', and 'validation'.

    Parameters
    ----------
    split_data : mapping
        Loaded split dictionary.
    split_name : str
        Requested split name.

    Returns
    -------
    sequence
        Entries belonging to the requested split.

    Raises
    ------
    KeyError
        If a matching split key cannot be found.
    ValueError
        If the split entries are not stored as a sequence.
    """

    normalized_name = split_name.lower().strip()

    aliases = {
        "train": [
            "train",
            "training",
        ],
        "val": [
            "val",
            "valid",
            "validation",
            "validation_set",
        ],
        "test": [
            "test",
            "testing",
        ],
    }

    requested_group = None

    for canonical_name, candidates in aliases.items():
        if normalized_name in candidates:
            requested_group = candidates
            break

    if requested_group is None:
        requested_group = [normalized_name]

    matched_key = None

    for candidate in requested_group:
        if candidate in split_data:
            matched_key = candidate
            break

    if matched_key is None:
        available_keys = list(split_data.keys())

        raise KeyError(
            f"Split '{split_name}' was not found. "
            f"Available keys: {available_keys}"
        )

    entries = split_data[matched_key]

    if not isinstance(entries, Sequence) or isinstance(
        entries,
        (str, bytes),
    ):
        raise ValueError(
            f"Split '{matched_key}' must contain a list-like "
            "sequence of samples."
        )

    return entries


# ============================================================
# 3. 경로 해석
# ============================================================

def resolve_dataset_path(
    stored_path: PathLike,
    dataset_root: PathLike,
) -> Path:
    """
    Resolve a stored dataset path.

    Absolute paths are preserved when valid. Relative paths are
    interpreted with respect to dataset_root.

    The function also attempts to recover paths stored with
    prefixes such as 'dataset/'.

    Parameters
    ----------
    stored_path : str or pathlib.Path
        Path stored in a split file.
    dataset_root : str or pathlib.Path
        Root directory of the local dataset.

    Returns
    -------
    pathlib.Path
        Resolved local path.
    """

    stored_path = Path(stored_path)
    dataset_root = Path(dataset_root)

    if stored_path.is_absolute() and stored_path.exists():
        return stored_path

    direct_candidate = dataset_root / stored_path

    if direct_candidate.exists():
        return direct_candidate

    path_parts = stored_path.parts

    if dataset_root.name in path_parts:
        dataset_index = path_parts.index(dataset_root.name)

        relative_parts = path_parts[
            dataset_index + 1:
        ]

        prefixed_candidate = dataset_root.joinpath(
            *relative_parts
        )

        if prefixed_candidate.exists():
            return prefixed_candidate

    return direct_candidate


# ============================================================
# 4. Entry 형식 표준화
# ============================================================

def normalize_pair_entry(
    entry: Any,
) -> PairType:
    """
    Convert a split entry into a standardized pair dictionary.

    Supported examples
    ------------------
    Dictionary:
        {
            "rgb_path": "...",
            "depth_path": "..."
        }

    Alternative dictionary keys:
        {
            "image": "...",
            "depth": "..."
        }

    Two-element sequence:
        ["rgb/path.png", "depth/path.png"]

    Parameters
    ----------
    entry : object
        One sample entry from the split JSON.

    Returns
    -------
    dict
        Dictionary with 'rgb_path' and 'depth_path'.

    Raises
    ------
    ValueError
        If RGB and depth paths cannot be identified.
    """

    if isinstance(entry, Mapping):
        rgb_key_candidates = [
            "rgb_path",
            "image_path",
            "image",
            "rgb",
            "input_path",
        ]

        depth_key_candidates = [
            "depth_path",
            "depth",
            "target_path",
            "label_path",
            "ground_truth_path",
            "gt_path",
        ]

        rgb_path = None
        depth_path = None

        for key in rgb_key_candidates:
            if key in entry:
                rgb_path = entry[key]
                break

        for key in depth_key_candidates:
            if key in entry:
                depth_path = entry[key]
                break

        if rgb_path is None or depth_path is None:
            raise ValueError(
                "Could not identify RGB and depth paths in "
                f"entry: {entry}"
            )

        return {
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
        }

    if isinstance(entry, Sequence) and not isinstance(
        entry,
        (str, bytes),
    ):
        if len(entry) != 2:
            raise ValueError(
                "A sequence-form pair must contain exactly "
                "two paths: RGB and depth."
            )

        return {
            "rgb_path": str(entry[0]),
            "depth_path": str(entry[1]),
        }

    raise ValueError(
        f"Unsupported split entry type: {type(entry)}"
    )


# ============================================================
# 5. 저장된 split 경로 복원
# ============================================================

def restore_pairs(
    entries: Sequence[Any],
    dataset_root: PathLike,
    check_exists: bool = True,
) -> List[PairType]:
    """
    Restore RGB-depth pairs from saved split entries.

    Parameters
    ----------
    entries : sequence
        Split entries loaded from JSON.
    dataset_root : str or pathlib.Path
        Root directory of the extracted KITTI dataset.
    check_exists : bool, default=True
        Whether to verify that every restored file exists.

    Returns
    -------
    list of dict
        Restored RGB-depth pair dictionaries.

    Raises
    ------
    FileNotFoundError
        If check_exists=True and one or more files are missing.
    """

    dataset_root = Path(dataset_root)

    restored_pairs = []
    missing_files = []

    for entry in entries:
        normalized_entry = normalize_pair_entry(entry)

        rgb_path = resolve_dataset_path(
            normalized_entry["rgb_path"],
            dataset_root,
        )

        depth_path = resolve_dataset_path(
            normalized_entry["depth_path"],
            dataset_root,
        )

        if check_exists:
            if not rgb_path.exists():
                missing_files.append(rgb_path)

            if not depth_path.exists():
                missing_files.append(depth_path)

        restored_pairs.append(
            {
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
            }
        )

    if missing_files:
        preview = "\n".join(
            str(path)
            for path in missing_files[:10]
        )

        raise FileNotFoundError(
            f"{len(missing_files)} dataset files were not "
            f"found.\nFirst missing files:\n{preview}"
        )

    return restored_pairs


# ============================================================
# 6. 전체 split 복원
# ============================================================

def load_and_restore_splits(
    split_json_path: PathLike,
    dataset_root: PathLike,
    check_exists: bool = True,
) -> Dict[str, List[PairType]]:
    """
    Load a split JSON and restore train, validation, and test
    RGB-depth pairs.

    Parameters
    ----------
    split_json_path : str or pathlib.Path
        Path to the split JSON.
    dataset_root : str or pathlib.Path
        Root directory of the dataset.
    check_exists : bool, default=True
        Whether to verify all restored file paths.

    Returns
    -------
    dict
        Dictionary containing:
        - train_pairs
        - val_pairs
        - test_pairs
    """

    split_data = load_split_json(
        split_json_path
    )

    train_entries = get_split_entries(
        split_data,
        "train",
    )

    val_entries = get_split_entries(
        split_data,
        "val",
    )

    test_entries = get_split_entries(
        split_data,
        "test",
    )

    return {
        "train_pairs": restore_pairs(
            train_entries,
            dataset_root,
            check_exists=check_exists,
        ),
        "val_pairs": restore_pairs(
            val_entries,
            dataset_root,
            check_exists=check_exists,
        ),
        "test_pairs": restore_pairs(
            test_entries,
            dataset_root,
            check_exists=check_exists,
        ),
    }


# ============================================================
# 7. Metadata 추출
# ============================================================

def extract_kitti_metadata(
    rgb_path: PathLike,
    depth_path: PathLike,
) -> Dict[str, str]:
    """
    Extract lightweight KITTI sample metadata from file paths.

    Parameters
    ----------
    rgb_path : str or pathlib.Path
        RGB image path.
    depth_path : str or pathlib.Path
        Depth image path.

    Returns
    -------
    dict
        Metadata including sequence and frame names.
    """

    rgb_path = Path(rgb_path)
    depth_path = Path(depth_path)

    frame = rgb_path.stem
    sequence = ""

    common_sequence_patterns = [
        "image_02",
        "image_03",
        "proj_depth",
    ]

    rgb_parts = rgb_path.parts

    for pattern in common_sequence_patterns:
        if pattern in rgb_parts:
            pattern_index = rgb_parts.index(pattern)

            if pattern_index > 0:
                sequence = rgb_parts[
                    pattern_index - 1
                ]
                break

    if not sequence and len(rgb_path.parents) >= 2:
        sequence = rgb_path.parent.parent.name

    return {
        "sequence": sequence,
        "frame": frame,
        "rgb_path": str(rgb_path),
        "depth_path": str(depth_path),
    }


# ============================================================
# 8. KITTI Dataset
# ============================================================

class KITTIDepthDataset(Dataset):
    """
    PyTorch Dataset for paired KITTI RGB and depth images.

    KITTI depth PNG values are converted to meters using:

        depth_in_meters = raw_depth / 256.0

    RGB images are resized using bilinear interpolation.
    Depth maps and valid masks are resized using nearest-neighbor
    interpolation to avoid creating artificial depth values.

    Parameters
    ----------
    pairs : sequence
        Sequence of RGB-depth pair dictionaries or two-element
        sequences.
    image_size : tuple of int, default=(518, 518)
        Output size in (height, width) order.
    min_depth : float, default=1e-3
        Minimum valid metric depth.
    max_depth : float, default=80.0
        Maximum valid metric depth.
    normalize_image : bool, default=True
        Whether to scale RGB values to [0, 1].
    """

    def __init__(
        self,
        pairs: Sequence[Any],
        image_size: Tuple[int, int] = (518, 518),
        min_depth: float = 1e-3,
        max_depth: float = 80.0,
        normalize_image: bool = True,
    ) -> None:
        super().__init__()

        if len(image_size) != 2:
            raise ValueError(
                "image_size must be a tuple of "
                "(height, width)."
            )

        if min_depth <= 0:
            raise ValueError(
                "min_depth must be greater than zero."
            )

        if max_depth <= min_depth:
            raise ValueError(
                "max_depth must be greater than min_depth."
            )

        self.pairs = [
            normalize_pair_entry(pair)
            for pair in pairs
        ]

        self.image_size = tuple(
            int(value)
            for value in image_size
        )

        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.normalize_image = normalize_image

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def load_rgb_image(
        rgb_path: PathLike,
    ) -> np.ndarray:
        """
        Load an RGB image as a NumPy array.
        """

        rgb_path = Path(rgb_path)

        if not rgb_path.exists():
            raise FileNotFoundError(
                f"RGB image was not found: {rgb_path}"
            )

        with Image.open(rgb_path) as image:
            rgb_image = np.asarray(
                image.convert("RGB"),
                dtype=np.float32,
            )

        return rgb_image

    @staticmethod
    def load_depth_map(
        depth_path: PathLike,
    ) -> np.ndarray:
        """
        Load a KITTI depth PNG and convert it to meters.
        """

        depth_path = Path(depth_path)

        if not depth_path.exists():
            raise FileNotFoundError(
                f"Depth image was not found: {depth_path}"
            )

        with Image.open(depth_path) as image:
            depth_raw = np.asarray(image)

        depth_map = (
            depth_raw.astype(np.float32)
            / 256.0
        )

        return depth_map

    def resize_sample(
        self,
        image_tensor: torch.Tensor,
        depth_tensor: torch.Tensor,
        valid_mask_tensor: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Resize image, depth, and mask tensors.
        """

        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0),
            size=self.image_size,
            mode="nearest",
        ).squeeze(0)

        valid_mask_tensor = F.interpolate(
            valid_mask_tensor.float().unsqueeze(0),
            size=self.image_size,
            mode="nearest",
        ).squeeze(0).bool()

        return (
            image_tensor,
            depth_tensor,
            valid_mask_tensor,
        )

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:
        pair = self.pairs[index]

        rgb_path = Path(pair["rgb_path"])
        depth_path = Path(pair["depth_path"])

        rgb_image = self.load_rgb_image(
            rgb_path
        )

        depth_map = self.load_depth_map(
            depth_path
        )

        if rgb_image.shape[:2] != depth_map.shape:
            raise ValueError(
                "RGB and depth image sizes do not match.\n"
                f"RGB shape   : {rgb_image.shape[:2]}\n"
                f"Depth shape : {depth_map.shape}\n"
                f"RGB path    : {rgb_path}\n"
                f"Depth path  : {depth_path}"
            )

        valid_mask = (
            np.isfinite(depth_map)
            & (depth_map >= self.min_depth)
            & (depth_map <= self.max_depth)
        )

        rgb_image = np.ascontiguousarray(
            rgb_image
        )

        depth_map = np.ascontiguousarray(
            depth_map
        )

        valid_mask = np.ascontiguousarray(
            valid_mask
        )

        image_tensor = torch.from_numpy(
            rgb_image
        ).permute(
            2,
            0,
            1,
        ).float()

        if self.normalize_image:
            image_tensor = image_tensor / 255.0

        depth_tensor = torch.from_numpy(
            depth_map
        ).unsqueeze(0).float()

        valid_mask_tensor = torch.from_numpy(
            valid_mask
        ).unsqueeze(0).bool()

        (
            image_tensor,
            depth_tensor,
            valid_mask_tensor,
        ) = self.resize_sample(
            image_tensor,
            depth_tensor,
            valid_mask_tensor,
        )

        depth_tensor = torch.where(
            valid_mask_tensor,
            depth_tensor,
            torch.zeros_like(depth_tensor),
        )

        metadata = extract_kitti_metadata(
            rgb_path,
            depth_path,
        )

        metadata["original_height"] = int(
            rgb_image.shape[0]
        )

        metadata["original_width"] = int(
            rgb_image.shape[1]
        )

        metadata["sample_index"] = int(index)

        return {
            "image": image_tensor,
            "depth": depth_tensor,
            "valid_mask": valid_mask_tensor,
            "metadata": metadata,
        }
