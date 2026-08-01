
import random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
)

from .config import (
    CHECKPOINT_ROOT,
    DATASET_ROOT,
    DEVICE,
    IMAGE_SIZE,
    MAX_DEPTH,
    METRIC_MODEL_NAME,
    MIN_DEPTH,
    NUM_WORKERS,
    PIN_MEMORY,
    RANDOM_SEED,
    SPLIT_JSON_PATH,
    TEST_BATCH_SIZE,
    TRAIN_BATCH_SIZE,
    VAL_BATCH_SIZE,
    create_project_directories,
)
from .dataset import (
    KITTIDepthDataset,
    load_and_restore_splits,
)


# ============================================================
# 1. 재현성 설정
# ============================================================

def set_random_seed(
    seed: int = RANDOM_SEED,
    deterministic: bool = False,
) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Parameters
    ----------
    seed : int, default=RANDOM_SEED
        Random seed used by all supported libraries.
    deterministic : bool, default=False
        Whether to request deterministic CUDA algorithms.

        Setting this to True improves reproducibility but may
        reduce performance or cause errors when an operation
        does not have a deterministic implementation.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        try:
            torch.use_deterministic_algorithms(
                True,
                warn_only=True,
            )
        except TypeError:
            torch.use_deterministic_algorithms(
                True
            )
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = (
            torch.cuda.is_available()
        )


# ============================================================
# 2. Dataset 생성
# ============================================================

def create_datasets(
    split_json_path: Path = SPLIT_JSON_PATH,
    dataset_root: Path = DATASET_ROOT,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    min_depth: float = MIN_DEPTH,
    max_depth: float = MAX_DEPTH,
    check_exists: bool = True,
) -> Dict[str, KITTIDepthDataset]:
    """
    Restore saved data splits and create KITTI datasets.

    Parameters
    ----------
    split_json_path : pathlib.Path
        Path to the persisted split JSON.
    dataset_root : pathlib.Path
        Root directory containing the KITTI data.
    image_size : tuple of int
        Target image size in (height, width) order.
    min_depth : float
        Minimum valid depth in meters.
    max_depth : float
        Maximum valid depth in meters.
    check_exists : bool, default=True
        Whether to verify all RGB and depth file paths.

    Returns
    -------
    dict
        Dictionary containing:
        - train_dataset
        - val_dataset
        - test_dataset
    """

    restored_splits = load_and_restore_splits(
        split_json_path=split_json_path,
        dataset_root=dataset_root,
        check_exists=check_exists,
    )

    train_dataset = KITTIDepthDataset(
        pairs=restored_splits["train_pairs"],
        image_size=image_size,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    val_dataset = KITTIDepthDataset(
        pairs=restored_splits["val_pairs"],
        image_size=image_size,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    test_dataset = KITTIDepthDataset(
        pairs=restored_splits["test_pairs"],
        image_size=image_size,
        min_depth=min_depth,
        max_depth=max_depth,
    )

    return {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }


# ============================================================
# 3. DataLoader 생성
# ============================================================

def create_dataloaders(
    datasets: Optional[
        Dict[str, KITTIDepthDataset]
    ] = None,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    val_batch_size: int = VAL_BATCH_SIZE,
    test_batch_size: int = TEST_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = PIN_MEMORY,
    persistent_workers: Optional[bool] = None,
    drop_last_train: bool = False,
    seed: int = RANDOM_SEED,
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders.

    Parameters
    ----------
    datasets : dict or None
        Dataset dictionary returned by create_datasets().
        When None, datasets are created automatically.
    train_batch_size : int
        Training batch size.
    val_batch_size : int
        Validation batch size.
    test_batch_size : int
        Test batch size.
    num_workers : int
        Number of subprocesses used for data loading.
    pin_memory : bool
        Whether to use pinned CPU memory.
    persistent_workers : bool or None
        Keep worker processes alive between epochs.
        When None, it is enabled automatically if
        num_workers is greater than zero.
    drop_last_train : bool
        Whether to drop an incomplete final training batch.
    seed : int
        Seed used by the shuffled training DataLoader.

    Returns
    -------
    dict
        Dictionary containing:
        - train_loader
        - val_loader
        - test_loader
        - train_dataset
        - val_dataset
        - test_dataset
    """

    if datasets is None:
        datasets = create_datasets()

    required_dataset_keys = {
        "train_dataset",
        "val_dataset",
        "test_dataset",
    }

    missing_keys = (
        required_dataset_keys
        - set(datasets.keys())
    )

    if missing_keys:
        raise KeyError(
            "Dataset dictionary is missing keys: "
            f"{sorted(missing_keys)}"
        )

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    generator = torch.Generator()
    generator.manual_seed(seed)

    common_loader_arguments = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": (
            persistent_workers
            if num_workers > 0
            else False
        ),
    }

    train_loader = DataLoader(
        datasets["train_dataset"],
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=drop_last_train,
        generator=generator,
        **common_loader_arguments,
    )

    val_loader = DataLoader(
        datasets["val_dataset"],
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_arguments,
    )

    test_loader = DataLoader(
        datasets["test_dataset"],
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        **common_loader_arguments,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": datasets["train_dataset"],
        "val_dataset": datasets["val_dataset"],
        "test_dataset": datasets["test_dataset"],
    }


# ============================================================
# 4. 모델 및 Image Processor 로드
# ============================================================

def load_depth_model(
    model_name: str = METRIC_MODEL_NAME,
    device: torch.device = DEVICE,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[
    torch.nn.Module,
    Any,
]:
    """
    Load a Hugging Face depth-estimation model and processor.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.
    device : torch.device
        Device onto which the model is moved.
    checkpoint_path : pathlib.Path or None
        Optional PyTorch checkpoint containing a model state
        dictionary. If supplied, it is loaded after the
        pretrained model is initialized.

    Returns
    -------
    tuple
        model, image_processor

    Notes
    -----
    The model is returned in evaluation mode. Before training,
    call model.train().
    """

    image_processor = (
        AutoImageProcessor.from_pretrained(
            model_name
        )
    )

    model = (
        AutoModelForDepthEstimation.from_pretrained(
            model_name
        )
    )

    if checkpoint_path is not None:
        checkpoint_path = Path(
            checkpoint_path
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Checkpoint was not found: "
                f"{checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint[
                    "model_state_dict"
                ]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint[
                    "state_dict"
                ]
            else:
                state_dict = checkpoint
        else:
            raise ValueError(
                "The loaded checkpoint must be a "
                "dictionary or state dictionary."
            )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

    model = model.to(device)
    model.eval()

    return model, image_processor


# ============================================================
# 5. 파라미터 수 확인
# ============================================================

def count_model_parameters(
    model: torch.nn.Module,
) -> Dict[str, int]:
    """
    Count total and trainable model parameters.

    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model.

    Returns
    -------
    dict
        Parameter counts:
        - total_parameters
        - trainable_parameters
        - frozen_parameters
    """

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    frozen_parameters = (
        total_parameters
        - trainable_parameters
    )

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": (
            trainable_parameters
        ),
        "frozen_parameters": frozen_parameters,
    }


# ============================================================
# 6. 전체 프로젝트 초기화
# ============================================================

def initialize_project(
    load_model: bool = False,
    model_name: str = METRIC_MODEL_NAME,
    device: torch.device = DEVICE,
    checkpoint_path: Optional[Path] = None,
    deterministic: bool = False,
    check_exists: bool = True,
    num_workers: int = NUM_WORKERS,
) -> Dict[str, Any]:
    """
    Initialize project directories, datasets, DataLoaders, and
    optionally the depth-estimation model.

    Parameters
    ----------
    load_model : bool, default=False
        Whether to load the pretrained model and processor.
    model_name : str
        Hugging Face model identifier.
    device : torch.device
        Target model device.
    checkpoint_path : pathlib.Path or None
        Optional model checkpoint.
    deterministic : bool
        Whether to request deterministic PyTorch execution.
    check_exists : bool
        Whether to verify all dataset paths.
    num_workers : int
        DataLoader worker count.

    Returns
    -------
    dict
        Initialized project resources.
    """

    create_project_directories()

    set_random_seed(
        seed=RANDOM_SEED,
        deterministic=deterministic,
    )

    datasets = create_datasets(
        check_exists=check_exists
    )

    resources = create_dataloaders(
        datasets=datasets,
        num_workers=num_workers,
    )

    resources["device"] = device

    if load_model:
        model, image_processor = (
            load_depth_model(
                model_name=model_name,
                device=device,
                checkpoint_path=checkpoint_path,
            )
        )

        resources["model"] = model
        resources["image_processor"] = (
            image_processor
        )

    return resources



# ============================================================
# Colab 로컬 데이터셋 복원
# ============================================================

def prepare_local_dataset(
    archive_path=None,
    extract_root=None,
    dataset_root=None,
    force_extract: bool = False,
):
    """
    Restore the KITTI dataset from a Drive TAR archive into
    Colab's local storage.

    Existing extracted train and test directories are reused
    unless force_extract=True.

    Returns
    -------
    pathlib.Path
        Root containing the extracted train and test folders.
    """

    import tarfile

    from .config import (
        DATASET_ARCHIVE_PATH,
        DATASET_ROOT,
        LOCAL_EXTRACT_ROOT,
    )

    if archive_path is None:
        archive_path = DATASET_ARCHIVE_PATH

    if extract_root is None:
        extract_root = LOCAL_EXTRACT_ROOT

    if dataset_root is None:
        dataset_root = DATASET_ROOT

    archive_path = Path(archive_path)
    extract_root = Path(extract_root)
    dataset_root = Path(dataset_root)

    train_root = dataset_root / "train"
    test_root = dataset_root / "test"

    dataset_is_ready = (
        train_root.is_dir()
        and test_root.is_dir()
    )

    if dataset_is_ready and not force_extract:
        return dataset_root

    if not archive_path.exists():
        raise FileNotFoundError(
            "Dataset TAR archive was not found: "
            f"{archive_path}"
        )

    extract_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    with tarfile.open(
        archive_path,
        mode="r:*",
    ) as tar:
        tar.extractall(
            path=extract_root
        )

    if not train_root.is_dir():
        raise FileNotFoundError(
            "Dataset extraction completed, but the train "
            f"directory was not found: {train_root}"
        )

    if not test_root.is_dir():
        raise FileNotFoundError(
            "Dataset extraction completed, but the test "
            f"directory was not found: {test_root}"
        )

    return dataset_root
