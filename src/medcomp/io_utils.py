"""Image input/output utilities, padding, cropping, and dataset file listing.

Supports standard image formats (PNG, JPG, BMP, TIFF) and DICOM (.dcm).
All images are returned as 2D uint8 arrays suitable for medical compression.
"""

from pathlib import Path
from typing import Sequence, Tuple, Union
import cv2
import numpy as np
import pydicom

from medcomp.config import JUNK_FILENAMES, SUPPORTED_IMAGE_EXTENSIONS


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk as a 2D uint8 grayscale array.

    Supported formats include .png, .jpg, .jpeg, .bmp, .tif, .tiff, and .dcm (DICOM).

    Conversions applied:
    - DICOM: RescaleSlope and RescaleIntercept are applied if present. If WindowCenter
      and WindowWidth metadata are available, windowing (VOI LUT) is applied to map
      Hounsfield/intensity values into [0, 255]. Otherwise, linear min-max scaling is used.
      Inverted photometric interpretation (MONOCHROME1) is mapped to standard presentation.
    - Color (RGB/BGR/BGRA): Converted to single-channel grayscale via standard ITU-R 601-2
      luma weighting (OpenCV cvtColor).
    - 16-bit unsigned (uint16): Full-scale 16-bit range [0, 65535] is mapped linearly to
      the 8-bit dynamic range [0, 255] using:
          arr_8bit = round(arr_16bit * (255.0 / 65535.0))
      This preserves monotonic relative intensities (0 -> 0, 65535 -> 255, 32768 -> 128).
    - Float: Values in [0.0, 1.0] are scaled by 255.0; other float ranges are clamped and rounded.

    Args:
        path: Filepath to the image.

    Returns:
        2D uint8 numpy array representing grayscale pixel intensities.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file cannot be decoded or has unsupported dimensions.
    """
    file_path = Path(path).resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    ext = file_path.suffix.lower()

    if ext == ".dcm":
        return _load_dicom(file_path)

    # Standard image decoding via OpenCV
    # Note: cv2.imdecode with fromfile handles Unicode and Windows path edge cases
    raw_bytes = np.fromfile(str(file_path), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode image: {file_path}")

    # Handle multi-channel images
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Unsupported channel count {img.shape[2]} for image {file_path}")

    # Handle bit depth conversion to uint8
    if img.dtype == np.uint8:
        return img
    elif img.dtype == np.uint16:
        # Linear conversion from 16-bit [0, 65535] to 8-bit [0, 255]
        scaled = np.clip(np.round(img.astype(np.float32) * (255.0 / 65535.0)), 0, 255)
        return scaled.astype(np.uint8)
    elif np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.0 and img.min() >= 0.0:
            scaled = np.clip(np.round(img * 255.0), 0, 255)
        else:
            scaled = np.clip(np.round(img), 0, 255)
        return scaled.astype(np.uint8)
    else:
        # Generic cast with min-max fallback
        min_v = float(np.min(img))
        max_v = float(np.max(img))
        if max_v > min_v:
            scaled = np.clip(np.round((img.astype(np.float32) - min_v) / (max_v - min_v) * 255.0), 0, 255)
        else:
            scaled = np.zeros_like(img, dtype=np.float32)
        return scaled.astype(np.uint8)


def _load_dicom(file_path: Path) -> np.ndarray:
    """Load a DICOM file and window it to 2D uint8 grayscale."""
    ds = pydicom.dcmread(str(file_path))
    arr = ds.pixel_array.astype(np.float64)

    # 1. Rescale slope and intercept
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    if slope != 1.0 or intercept != 0.0:
        arr = arr * slope + intercept

    # 2. Window center / window width (VOI LUT)
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth", None)
    if wc is not None and ww is not None:
        if isinstance(wc, (Sequence, list, tuple)) and not isinstance(wc, (str, bytes)):
            wc = wc[0]
        if isinstance(ww, (Sequence, list, tuple)) and not isinstance(ww, (str, bytes)):
            ww = ww[0]
        c = float(wc)
        w = float(ww)
        lower = c - 0.5 - (w - 1.0) / 2.0
        upper = c - 0.5 + (w - 1.0) / 2.0
        if upper > lower:
            arr = np.clip(arr, lower, upper)
            arr = (arr - lower) / (upper - lower) * 255.0
        else:
            arr = np.zeros_like(arr)
    else:
        min_v = float(np.min(arr))
        max_v = float(np.max(arr))
        if max_v > min_v:
            arr = (arr - min_v) / (max_v - min_v) * 255.0
        else:
            arr = np.zeros_like(arr)

    # 3. Photometric interpretation (MONOCHROME1 means 0 is maximum brightness)
    photo_interp = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
    if photo_interp == "MONOCHROME1":
        arr = 255.0 - arr

    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)

    # Convert 3D RGB/multichannel DICOM if needed
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        elif arr.shape[0] == 3:
            arr = cv2.cvtColor(np.transpose(arr, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        else:
            arr = arr[0]

    return arr


def save_image(path: Union[str, Path], arr: np.ndarray) -> None:
    """Save a 2D uint8 grayscale array to disk.

    Args:
        path: Target file path.
        arr: 2D uint8 numpy array to write.

    Raises:
        ValueError: If arr is not a 2D uint8 array.
        IOError: If writing to disk fails.
    """
    if not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError(f"Array must be a 2D uint8 numpy array. Got shape={getattr(arr, 'shape', None)}, dtype={getattr(arr, 'dtype', None)}")

    target_path = Path(path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    ext = target_path.suffix.lower()
    success, encoded = cv2.imencode(ext if ext else ".png", arr)
    if not success:
        raise IOError(f"Failed to encode image to {target_path}")

    with open(target_path, "wb") as f:
        f.write(encoded.tobytes())


def list_images(folder: Union[str, Path]) -> list[Path]:
    """Return a sorted list of supported image filepaths in the specified directory.

    Excludes operating system junk files (e.g. Thumbs.db, desktop.ini, .DS_Store).

    Args:
        folder: Directory to inspect.

    Returns:
        Sorted list of Path objects for valid image files.

    Raises:
        FileNotFoundError: If the folder does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    folder_path = Path(folder).resolve()
    if not folder_path.exists():
        raise FileNotFoundError(f"Directory not found: {folder_path}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder_path}")

    files = []
    for item in folder_path.iterdir():
        if item.is_file():
            name_lower = item.name.lower()
            if name_lower in JUNK_FILENAMES or name_lower.startswith("."):
                continue
            if item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                files.append(item)

    return sorted(files)


def pad_to_multiple(img: np.ndarray, m: int) -> np.ndarray:
    """Pad a 2D image so that both height and width are integer multiples of m.

    Padding is added to the bottom and right edges using edge pixel replication.

    Args:
        img: 2D numpy array of shape (H, W).
        m: Multiple requirement (e.g. 8 for DCT blocks, 16 for wavelet levels).

    Returns:
        Padded 2D numpy array with dimensions divisible by m.

    Raises:
        ValueError: If m <= 0 or img is not a 2D array.
    """
    if m <= 0:
        raise ValueError(f"Multiple m must be a positive integer, got {m}")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image array, got {img.ndim}D with shape {img.shape}")

    h, w = img.shape
    pad_h = (m - (h % m)) % m
    pad_w = (m - (w % m)) % m

    if pad_h == 0 and pad_w == 0:
        return img.copy()

    # Replicate edge pixels to bottom and right
    return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REPLICATE)


def crop_to(img: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Inverse of pad_to_multiple: crop padded image back to original dimensions.

    Args:
        img: 2D numpy array of padded image.
        shape: Target original shape (height, width).

    Returns:
        Cropped 2D numpy array of shape (shape[0], shape[1]).

    Raises:
        ValueError: If target shape is larger than img dimensions or not 2D.
    """
    if len(shape) < 2:
        raise ValueError(f"Target shape must specify at least (height, width), got {shape}")
    orig_h, orig_w = shape[:2]

    if img.shape[0] < orig_h or img.shape[1] < orig_w:
        raise ValueError(f"Cannot crop image of shape {img.shape} to larger shape {(orig_h, orig_w)}")

    return img[:orig_h, :orig_w].copy()
