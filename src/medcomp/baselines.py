"""Standard compression baseline codecs: JPEG (DCT), JPEG 2000 (DWT), and PNG (lossless).

All codecs encode and decode completely in-memory using BytesIO, measuring
the true byte length of the encoded stream without estimation.
"""

import io
from typing import Tuple, Union
import numpy as np
from PIL import Image, features


def check_jpeg2000_support() -> bool:
    """Verify whether OpenJPEG codec support is available in Pillow."""
    return bool(features.check_codec("jpg_2000"))


def jpeg_codec(img: np.ndarray, quality: int) -> Tuple[bytes, np.ndarray]:
    """Encode and decode an 8-bit grayscale image using standard JPEG (DCT-based).

    Args:
        img: 2D uint8 numpy array of shape (H, W).
        quality: JPEG quality factor (1-95).

    Returns:
        Tuple of (compressed_bytes, reconstructed_array).

    Raises:
        ValueError: If input is not a 2D uint8 array or quality is out of range.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"Input must be a 2D uint8 numpy array, got shape={getattr(img, 'shape', None)}, dtype={getattr(img, 'dtype', None)}")
    if not (1 <= quality <= 100):
        raise ValueError(f"JPEG quality must be between 1 and 100, got {quality}")

    im = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    compressed_bytes = buf.getvalue()

    buf.seek(0)
    with Image.open(buf) as rec_im:
        reconstructed = np.array(rec_im, dtype=np.uint8)

    return compressed_bytes, reconstructed


def jpeg2000_codec(img: np.ndarray, ratio: Union[float, int]) -> Tuple[bytes, np.ndarray]:
    """Encode and decode an 8-bit grayscale image using JPEG 2000 (DWT-based via OpenJPEG).

    Uses irreversible 9/7 wavelet transform and target compression ratio layer.

    Args:
        img: 2D uint8 numpy array of shape (H, W).
        ratio: Target compression ratio (e.g. 5, 10, 20, 40, 80).

    Returns:
        Tuple of (compressed_bytes, reconstructed_array).

    Raises:
        RuntimeError: If Pillow build lacks JPEG 2000 (OpenJPEG) support.
        ValueError: If input is not a 2D uint8 array or ratio is non-positive.
    """
    if not check_jpeg2000_support():
        raise RuntimeError(
            "Pillow build lacks JPEG 2000 (OpenJPEG) support. "
            "To fix: install a Pillow wheel compiled with OpenJPEG, e.g.: "
            "pip install --upgrade pillow"
        )
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"Input must be a 2D uint8 numpy array, got shape={getattr(img, 'shape', None)}, dtype={getattr(img, 'dtype', None)}")
    if ratio <= 0:
        raise ValueError(f"JPEG 2000 compression ratio must be positive, got {ratio}")

    im = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="JPEG2000", quality_mode="rates", quality_layers=[float(ratio)], irreversible=True)
    compressed_bytes = buf.getvalue()

    buf.seek(0)
    with Image.open(buf) as rec_im:
        reconstructed = np.array(rec_im, dtype=np.uint8)

    return compressed_bytes, reconstructed


def png_codec(img: np.ndarray) -> Tuple[bytes, np.ndarray]:
    """Encode and decode an 8-bit grayscale image using lossless PNG (DEFLATE).

    Uses maximum compression level (compress_level=9).

    Args:
        img: 2D uint8 numpy array of shape (H, W).

    Returns:
        Tuple of (compressed_bytes, reconstructed_array).

    Raises:
        ValueError: If input is not a 2D uint8 array.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"Input must be a 2D uint8 numpy array, got shape={getattr(img, 'shape', None)}, dtype={getattr(img, 'dtype', None)}")

    im = Image.fromarray(img, mode="L")
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=9)
    compressed_bytes = buf.getvalue()

    buf.seek(0)
    with Image.open(buf) as rec_im:
        reconstructed = np.array(rec_im, dtype=np.uint8)

    return compressed_bytes, reconstructed
