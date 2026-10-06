"""Configuration settings and filesystem paths for the medical compression pipeline."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "Lung Segmentation"
WORK_DIR = PROJECT_ROOT / "work"
RESULTS_DIR = PROJECT_ROOT / "results"
TESTS_DIR = PROJECT_ROOT / "tests"

# Dataset subdirectories
CXR_DIR = DATA_ROOT / "CXR_png"
MASKS_DIR = DATA_ROOT / "masks"
CLINICAL_DIR = DATA_ROOT / "ClinicalReadings"
TEST_DIR = DATA_ROOT / "test"

# Image processing constants
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".dcm")
JUNK_FILENAMES = {"thumbs.db", "desktop.ini", ".ds_store"}
TARGET_LONG_SIDE = 512

# Project data and processed paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = WORK_DIR / "processed"
PROCESSED_IMAGES_DIR = PROCESSED_DIR / "images"
PROCESSED_MASKS_DIR = PROCESSED_DIR / "masks"

# Create output directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_original_bytes(img_or_shape) -> int:
    """Compute uncompressed baseline size in bytes for an 8-bit single-channel image.

    In this pipeline, every medical X-ray is standardized to an 8-bit grayscale 2D array
    (1 byte per pixel). Thus, the uncompressed representation size is strictly:
        original_bytes = height * width

    All compression ratio (CR = original_bytes / compressed_bytes) and bits-per-pixel
    (bpp = 8 * compressed_bytes / (height * width)) calculations are derived from this.

    Args:
        img_or_shape: Either a 2D numpy array or a tuple/list of (height, width).

    Returns:
        Integer representing total raw uncompressed bytes (height * width).

    Raises:
        ValueError: If shape is invalid or has non-positive dimensions.
    """
    if hasattr(img_or_shape, "shape"):
        h, w = img_or_shape.shape[:2]
    elif isinstance(img_or_shape, (tuple, list)) and len(img_or_shape) >= 2:
        h, w = int(img_or_shape[0]), int(img_or_shape[1])
    else:
        raise ValueError(f"Expected 2D numpy array or (height, width) tuple, got {type(img_or_shape)}")

    if h <= 0 or w <= 0:
        raise ValueError(f"Dimensions must be positive, got height={h}, width={w}")

    return h * w


