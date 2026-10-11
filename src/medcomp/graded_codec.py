"""Graded zone image codec: Lossless/near-lossless core, near-lossless band, lossy background.

Architecture:
1. Zone Construction:
   - Core zone (1): Original medical ROI mask (e.g. lung fields).
   - Band zone (2): (Dilated mask by band_px OR optional extra_mask) AND NOT Core.
   - Background zone (0): All remaining pixels.
2. Coded Sections:
   - Mask: Compressed with mask_codec.
   - Core & Band: Compressed with near-lossless MED prediction and per-zone Huffman coding (nearlossless.py),
     strictly bounded by |x - recon| <= delta_core on core and <= delta_band on band.
   - Background: Inpainted under zones 1 & 2 to eliminate sharp boundary leakage, then compressed
     with tuned DWT (or DCT).
3. Exact Reconstruction:
   - Decoder recovers mask and band_px, rebuilds zone map, decodes background, and restores
     zones 1 and 2 from the near-lossless bitstream without any reference image.
"""

from __future__ import annotations

import struct
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .dct_codec import dct_decode, dct_encode
from .dwt_codec import dwt_decode, dwt_encode
from .mask_codec import mask_decode, mask_encode
from .nearlossless import nl_decode, nl_encode

GRADED_MAGIC = b"MEDGRD10"

METHOD_DCT = 0
METHOD_DWT = 1

METHOD_MAP: dict[str, int] = {
    "dct": METHOD_DCT,
    "dwt": METHOD_DWT,
}
INV_METHOD_MAP: dict[int, str] = {v: k for k, v in METHOD_MAP.items()}


def dilate_mask(mask: np.ndarray, d_pixels: int) -> np.ndarray:
    """Dilate binary mask by radius d_pixels with elliptical structuring element."""
    if d_pixels <= 0:
        return mask.astype(bool)
    k_size = 2 * d_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def make_zones(
    mask: np.ndarray,
    band_px: int,
    extra_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Construct 3-zone map (0=bg, 1=core, 2=band).

    Rules:
    - Core (1) = mask
    - Band (2) = (dilate(mask, band_px) OR extra_mask) AND NOT Core
    - Background (0) = everywhere else
    """
    core = mask.astype(bool)
    if band_px > 0:
        dilated = dilate_mask(core, band_px)
    else:
        dilated = core.copy()

    if extra_mask is not None:
        dilated = (dilated | extra_mask.astype(bool))

    band = dilated & (~core)

    zone_map = np.zeros(mask.shape, dtype=np.uint8)
    zone_map[core] = 1
    zone_map[band] = 2
    return zone_map


def graded_encode(
    img: np.ndarray,
    mask: np.ndarray,
    params: dict[str, Any],
    extra_mask: Optional[np.ndarray] = None,
) -> tuple[bytes, np.ndarray, dict[str, int]]:
    """Encode image with graded error guarantee zones and lossy background.

    Args:
        img: 2D uint8 numpy array.
        mask: 2D binary numpy array for core ROI.
        params: Compression parameters:
            - delta_core: int
            - delta_band: int
            - band_px: int
            - bg_quality: int
            - bg_method: str ('dwt' or 'dct')
            - fill: str ('inpaint')
        extra_mask: Optional extra mask to include into the band zone.

    Returns:
        tuple[bytes, np.ndarray, dict[str, int]]:
            - Serialized compressed bitstream bytes.
            - Reconstructed 2D uint8 image.
            - Breakdown dictionary of bytes per zone (header, mask, core, band, background).
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"Expected 2D uint8 image, got shape={getattr(img, 'shape', None)}, dtype={getattr(img, 'dtype', None)}")

    h, w = img.shape
    delta_core = int(params.get("delta_core", 0))
    delta_band = int(params.get("delta_band", 2))
    band_px = int(params.get("band_px", 4))
    bg_quality = int(params.get("bg_quality", 50))
    bg_method = str(params.get("bg_method", "dwt")).lower()
    fill = str(params.get("fill", "inpaint")).lower()

    # 1. Build zone map
    zone_map = make_zones(mask, band_px, extra_mask=extra_mask)

    # 2. Compress mask (and optional extra_mask) picking smaller of RLE and BBox
    m_rle = mask_encode(mask > 0, bbox_mode=False)
    m_bbox = mask_encode(mask > 0, bbox_mode=True)
    mask_bytes = m_bbox if len(m_bbox) < len(m_rle) else m_rle

    has_extra = 1 if extra_mask is not None else 0
    if has_extra:
        e_rle = mask_encode(extra_mask > 0, bbox_mode=False)
        e_bbox = mask_encode(extra_mask > 0, bbox_mode=True)
        extra_bytes = e_bbox if len(e_bbox) < len(e_rle) else e_rle
    else:
        extra_bytes = b""

    # 3. Near-lossless encode core (1) and band (2)
    nl_bytes, nl_recon = nl_encode(
        img,
        zone_map,
        deltas={1: delta_core, 2: delta_band},
    )

    # Extract core and band bytes from nl_bytes header
    # nl_bytes header: Magic (7B) + h (2B) + w (2B) + d1 (1B) + d2 (1B) + n1 (4B) + n2 (4B) + core_bytes (4B) + band_bytes (4B)
    core_bytes, band_bytes = struct.unpack_from(">II", nl_bytes, 21)
    nl_header_overhead = 29

    # 4. Background conditioning and compression
    inpaint_mask = (zone_map > 0).astype(np.uint8) * 255
    if fill == "inpaint":
        bg_img = cv2.inpaint(img, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    elif fill == "mean":
        bg_pixels = img[zone_map == 0]
        mean_val = int(np.mean(bg_pixels)) if len(bg_pixels) > 0 else 128
        bg_img = img.copy()
        bg_img[zone_map > 0] = mean_val
    else:
        bg_img = img.copy()

    m_key = bg_method.lower()
    if m_key == "dct":
        bg_bytes, bg_recon = dct_encode(bg_img, quality=bg_quality)
    else:
        bg_bytes, bg_recon = dwt_encode(bg_img, quality=bg_quality)

    # 5. Full reconstruction
    recon = bg_recon.copy()
    recon[zone_map == 1] = nl_recon[zone_map == 1]
    recon[zone_map == 2] = nl_recon[zone_map == 2]

    # 6. Container serialization
    # Magic (8B) + h (2B) + w (2B) + delta_core (1B) + delta_band (1B) + band_px (1B) +
    # bg_method (1B) + bg_quality (1B) + has_extra (1B) +
    # mask_len (4B) + extra_len (4B) + nl_len (4B) + bg_len (4B) = 34 bytes
    container_header = bytearray(GRADED_MAGIC)
    container_header.extend(
        struct.pack(
            ">HHBBBBBBIIII",
            h,
            w,
            delta_core,
            delta_band,
            band_px,
            METHOD_MAP[m_key],
            bg_quality,
            has_extra,
            len(mask_bytes),
            len(extra_bytes),
            len(nl_bytes),
            len(bg_bytes),
        )
    )

    final_bytes = bytes(container_header) + mask_bytes + extra_bytes + nl_bytes + bg_bytes

    # Exact breakdown of byte consumption
    parts = {
        "header": len(container_header) + nl_header_overhead,
        "mask": len(mask_bytes) + len(extra_bytes),
        "core": core_bytes,
        "band": band_bytes,
        "background": len(bg_bytes),
    }

    return final_bytes, recon, parts


def graded_decode(data: bytes) -> np.ndarray:
    """Decode a graded zone compressed container into 2D uint8 image using only bitstream bytes.

    Args:
        data: Serialized bitstream bytes.

    Returns:
        np.ndarray: Reconstructed 2D uint8 image with guaranteed core/band errors.
    """
    if len(data) < 34:
        raise ValueError("Data too short for graded container header.")

    magic = data[:8]
    if magic != GRADED_MAGIC:
        raise ValueError(f"Invalid magic bytes '{magic}', expected '{GRADED_MAGIC}'")

    (
        h,
        w,
        delta_core,
        delta_band,
        band_px,
        method_id,
        bg_quality,
        has_extra,
        mask_len,
        extra_len,
        nl_len,
        bg_len,
    ) = struct.unpack_from(">HHBBBBBBIIII", data, 8)
    offset = 34

    # Slice sections
    mask_bytes = data[offset : offset + mask_len]
    offset += mask_len

    extra_bytes = data[offset : offset + extra_len]
    offset += extra_len

    nl_bytes = data[offset : offset + nl_len]
    offset += nl_len

    bg_bytes = data[offset : offset + bg_len]

    # Reconstruct mask and zone map
    mask = mask_decode(mask_bytes)
    extra_mask = mask_decode(extra_bytes) if has_extra else None
    zone_map = make_zones(mask, band_px, extra_mask=extra_mask)

    # Decode background
    if method_id == METHOD_DWT:
        bg_recon = dwt_decode(bg_bytes)
    else:
        bg_recon = dct_decode(bg_bytes)

    # Decode near-lossless core and band
    nl_recon = nl_decode(nl_bytes, zone_map)

    # Overwrite zones 1 and 2
    recon = bg_recon.copy()
    recon[zone_map == 1] = nl_recon[zone_map == 1]
    recon[zone_map == 2] = nl_recon[zone_map == 2]

    return recon
