"""Hybrid ROI image codec: Lossless Region of Interest (ROI), lossy background.

Architecture:
1. Mask Encoding:
   Losslessly encodes binary mask using the smaller of full row-wise RLE or bounding-box RLE.
2. Lossless ROI Encoding:
   Losslessly encodes ROI pixels using MED (LOCO-I / JPEG-LS) predictor with modular residual wrap
   and adaptive canonical Huffman coding.
3. Background Conditioning & Encoding:
   Replaces ROI pixels in a copy of the image using one of three fill strategies:
   - 'none': Leaves original pixels intact.
   - 'mean': Fills ROI with the average intensity of non-ROI background pixels.
   - 'inpaint': Inpaints the ROI smoothly using cv2.inpaint (Navier-Stokes / Telea) to avoid high-frequency edge leakage.
   Encodes conditioned background using either Custom DCT (8x8 JPEG) or Tuned Custom DWT (bior2.2, 4 levels).
4. Container Serialization:
   Combines magic bytes (MEDROI10), method identifier, quality, and exact section lengths (mask, roi, background).
5. Exact ROI Restoration:
   Decoder decodes mask, decodes lossy background, and overwrites ROI pixels with the decoded lossless ROI data,
   ensuring 100% bit-exact medical ROI reconstruction.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .dct_codec import dct_decode, dct_encode
from .dwt_codec import dwt_decode, dwt_encode
from .lossless_codec import lossless_decode, lossless_encode
from .mask_codec import mask_decode, mask_encode

ROI_MAGIC = b"MEDROI10"

METHOD_DCT = 0
METHOD_DWT = 1

METHOD_MAP: dict[str, int] = {
    "dct": METHOD_DCT,
    "dwt": METHOD_DWT,
}
INV_METHOD_MAP: dict[int, str] = {v: k for k, v in METHOD_MAP.items()}


def roi_encode(
    img: np.ndarray,
    mask: np.ndarray,
    method: str = "dwt",
    quality: int = 50,
    fill: str = "inpaint",
) -> tuple[bytes, np.ndarray, dict[str, int]]:
    """Encode an image with lossless ROI and lossy background.

    Args:
        img: 2D uint8 numpy array.
        mask: 2D boolean numpy array matching img shape.
        method: Lossy background codec, either 'dct' or 'dwt'.
        quality: Background quality setting in [1, 100].
        fill: ROI fill strategy before background compression ('none', 'mean', or 'inpaint').

    Returns:
        tuple[bytes, np.ndarray, dict[str, int]]:
            - Compressed container bytes.
            - Reconstructed image (lossless ROI, lossy background).
            - Dictionary of section byte counts ('header', 'mask', 'roi', 'background').

    Raises:
        ValueError: If mask is None, shape mismatch, or invalid method/fill.
    """
    if mask is None:
        raise ValueError("mask is required for roi_encode and cannot be None.")
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"Expected 2D uint8 array for img, got shape={getattr(img, 'shape', None)}, dtype={getattr(img, 'dtype', None)}")
    if not isinstance(mask, np.ndarray) or mask.shape != img.shape:
        raise ValueError(f"Mask shape {getattr(mask, 'shape', None)} does not match image shape {img.shape}")

    m_key = method.lower().strip()
    if m_key not in METHOD_MAP:
        raise ValueError(f"Invalid method '{method}', must be 'dct' or 'dwt'")

    fill_mode = fill.lower().strip()
    if fill_mode not in ("none", "mean", "inpaint"):
        raise ValueError(f"Invalid fill '{fill}', must be 'none', 'mean', or 'inpaint'")

    mask_bool = mask.astype(bool)
    quality = max(1, min(100, int(quality)))

    # 1. Mask encoding: Pick smaller of RLE and BBox variants
    m_rle = mask_encode(mask_bool, bbox_mode=False)
    m_bbox = mask_encode(mask_bool, bbox_mode=True)
    mask_bytes = m_bbox if len(m_bbox) < len(m_rle) else m_rle

    # 2. Lossless ROI encoding
    roi_bytes = lossless_encode(img, mask=mask_bool, context=False)

    # 3. Background conditioning
    bg_img = img.copy()
    if fill_mode == "none":
        pass
    elif fill_mode == "mean":
        non_roi = img[~mask_bool]
        mean_val = int(round(float(non_roi.mean()))) if non_roi.size > 0 else 128
        bg_img[mask_bool] = mean_val
    elif fill_mode == "inpaint":
        if np.any(mask_bool) and not np.all(mask_bool):
            mask_u8 = (mask_bool.astype(np.uint8)) * 255
            bg_img = cv2.inpaint(img, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        elif np.all(mask_bool):
            bg_img.fill(128)

    # 4. Background lossy encoding
    if m_key == "dct":
        bg_bytes, bg_recon = dct_encode(bg_img, quality=quality)
    else:
        bg_bytes, bg_recon = dwt_encode(bg_img, quality=quality)

    # 5. Build container
    # Header format:
    # Magic (8 bytes) + method (1 byte) + quality (1 byte) +
    # mask_len (4 bytes) + roi_len (4 bytes) + bg_len (4 bytes) = 22 bytes
    header = bytearray(ROI_MAGIC)
    header.extend(
        struct.pack(
            ">BBIII",
            METHOD_MAP[m_key],
            quality,
            len(mask_bytes),
            len(roi_bytes),
            len(bg_bytes),
        )
    )

    final_bytes = bytes(header) + mask_bytes + roi_bytes + bg_bytes

    # 6. Encoder reconstruction (exact ROI + lossy background)
    recon = bg_recon.copy()
    recon[mask_bool] = img[mask_bool]

    parts = {
        "header": len(header),
        "mask": len(mask_bytes),
        "roi": len(roi_bytes),
        "background": len(bg_bytes),
    }

    return final_bytes, recon, parts


def roi_decode(data: bytes) -> np.ndarray:
    """Decode a hybrid ROI bitstream using only compressed bytes.

    Args:
        data: Compressed container bytes.

    Returns:
        np.ndarray: Reconstructed 2D uint8 image with bit-exact ROI.

    Raises:
        ValueError: If bitstream header is invalid or corrupt.
    """
    if len(data) < 22:
        raise ValueError("Data too short for ROI container header.")

    magic = data[:8]
    if magic != ROI_MAGIC:
        raise ValueError(f"Invalid magic bytes '{magic}', expected '{ROI_MAGIC}'")

    method_id, quality, mask_len, roi_len, bg_len = struct.unpack_from(">BBIII", data, 8)
    offset = 22

    total_expected = offset + mask_len + roi_len + bg_len
    if len(data) < total_expected:
        raise ValueError(f"Corrupt bitstream: expected {total_expected} bytes, got {len(data)}")

    # Slice sections
    mask_bytes = data[offset : offset + mask_len]
    offset += mask_len

    roi_bytes = data[offset : offset + roi_len]
    offset += roi_len

    bg_bytes = data[offset : offset + bg_len]
    offset += bg_len

    # Decode mask
    mask = mask_decode(mask_bytes)

    # Decode background
    if method_id == METHOD_DCT:
        bg_recon = dct_decode(bg_bytes)
    elif method_id == METHOD_DWT:
        bg_recon = dwt_decode(bg_bytes)
    else:
        raise ValueError(f"Unknown method ID {method_id} in container header.")

    # Decode lossless ROI
    roi_recon = lossless_decode(roi_bytes, mask=mask)

    # Overwrite ROI pixels
    recon = bg_recon.copy()
    recon[mask] = roi_recon[mask]

    return recon
