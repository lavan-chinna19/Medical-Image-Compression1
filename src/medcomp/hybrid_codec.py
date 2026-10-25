"""Hybrid DCT-DWT image codec combining 2D Discrete Wavelet Transform and 8x8 Block DCT.

Architecture:
1. Wavelet Decomposition:
   Applies a 1-level 2D DWT (biorthogonal bior2.2) to decompose the image into:
   - LL: Low-frequency structural approximation subband
   - LH: Horizontal high-frequency subband
   - HL: Vertical high-frequency subband
   - HH: Diagonal high-frequency subband
2. Block DCT on Approximation Subband:
   The LL subband contains the vast majority of image energy and diagnostic anatomy.
   It is partitioned into 8x8 blocks, and a 2D DCT (orthonormal) is computed for each block.
   DCT coefficients are quantized using the IJG JPEG luminance quantization table scaled by quality.
   Quantized coefficients are zigzag-scanned and entropy-coded with differential DC and run-length AC.
3. Quantization of Detail Subbands:
   The directional detail subbands (LH, HL, HH) are quantized using perceptual dead-zone scalar
   quantization scaled exponentially according to the target quality factor.
4. Bitstream Serialization:
   Sections are serialized with magic bytes `MEDHYB10`, geometry, quality, and section lengths.
5. Reconstruction (Inverse Pipeline):
   Decodes the bitstream, computes 2D IDCT on LL blocks, dequantizes detail subbands,
   and computes the 2D inverse DWT (pywt.idwt2) to reconstruct the full spatial image.
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import Tuple

import numpy as np
import pywt
import scipy.fft

from .dct_codec import (
    INV_ZIGZAG_INDICES,
    ZIGZAG_INDICES,
    get_quantization_table,
)
from .io_utils import crop_to, pad_to_multiple

HYBRID_MAGIC = b"MEDHYB10"


def _quantize_deadzone(arr: np.ndarray, step: float, theta: float = 0.25) -> np.ndarray:
    """Apply dead-zone uniform scalar quantization."""
    if step <= 0:
        return arr.astype(np.int32)
    abs_arr = np.abs(arr)
    q = np.floor(abs_arr / step + theta)
    return (np.sign(arr) * q).astype(np.int32)


def _dequantize_deadzone(arr: np.ndarray, step: float, bias: float = 0.20) -> np.ndarray:
    """Dequantize dead-zone coefficients with reconstruction centroid bias."""
    abs_q = np.abs(arr).astype(np.float32)
    nonzero = abs_q > 0
    recon = np.zeros_like(arr, dtype=np.float32)
    recon[nonzero] = (abs_q[nonzero] - theta_bias_factor(bias)) * step
    return np.sign(arr).astype(np.float32) * recon


def theta_bias_factor(bias: float) -> float:
    return max(0.0, min(0.5, float(bias)))


def hybrid_encode(
    img: np.ndarray,
    quality: int = 50,
    wavelet: str = "bior2.2",
) -> Tuple[bytes, np.ndarray]:
    """Encode an 8-bit grayscale image using the Hybrid DCT-DWT compression algorithm.

    Args:
        img: 2D uint8 numpy array of shape (H, W).
        quality: Quality factor in [1, 100]. Higher = higher fidelity, lower compression ratio.
        wavelet: Wavelet filter name for DWT decomposition (default 'bior2.2').

    Returns:
        Tuple of (compressed_bytes, reconstructed_image).
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"Input must be 2D uint8 numpy array, got {getattr(img, 'shape', None)}")

    orig_h, orig_w = img.shape
    quality = max(1, min(100, int(quality)))

    # 1. Pad image so that LL subband dimensions are exact multiples of 8 (image multiple of 16)
    padded = pad_to_multiple(img, 16)
    pad_h, pad_w = padded.shape

    # 2. Level shift (-128)
    shifted = padded.astype(np.float32) - 128.0

    # 3. 1-level 2D DWT decomposition with periodization for exact dyadic dimensions
    ll, (lh, hl, hh) = pywt.dwt2(shifted, wavelet=wavelet, mode="periodization")
    ll_h, ll_w = ll.shape
    assert ll_h % 8 == 0 and ll_w % 8 == 0, f"LL shape {(ll_h, ll_w)} not multiple of 8"

    n_bh = ll_h // 8
    n_bw = ll_w // 8
    n_blocks = n_bh * n_bw

    # 4. 2D Block DCT on LL subband
    ll_blocks = ll.reshape(n_bh, 8, n_bw, 8).transpose(0, 2, 1, 3).reshape(n_blocks, 8, 8)
    dct_blocks = scipy.fft.dctn(ll_blocks, axes=(-2, -1), norm="ortho")

    # Quantize LL DCT coefficients
    q_table = get_quantization_table(quality)
    quant_ll = np.round(dct_blocks / q_table).astype(np.int32)

    # Reconstruction path for LL
    dequant_ll = quant_ll.astype(np.float32) * q_table
    idct_blocks = scipy.fft.idctn(dequant_ll, axes=(-2, -1), norm="ortho")
    ll_recon = idct_blocks.reshape(n_bh, n_bw, 8, 8).transpose(0, 2, 1, 3).reshape(ll_h, ll_w)

    # 5. Dead-zone quantization on detail subbands
    # Detail step size scales with quality
    detail_step = float(2.0 ** ((100.0 - quality) / 14.0))
    q_lh = _quantize_deadzone(lh, detail_step)
    q_hl = _quantize_deadzone(hl, detail_step)
    q_hh = _quantize_deadzone(hh, detail_step)

    # Dequantize detail subbands for reconstruction
    lh_recon = _dequantize_deadzone(q_lh, detail_step)
    hl_recon = _dequantize_deadzone(q_hl, detail_step)
    hh_recon = _dequantize_deadzone(q_hh, detail_step)

    # 6. Inverse 2D DWT reconstruction
    recon_shifted = pywt.idwt2((ll_recon, (lh_recon, hl_recon, hh_recon)), wavelet=wavelet, mode="periodization")
    recon_shifted = crop_to(recon_shifted, (pad_h, pad_w))
    recon_img = np.clip(np.round(recon_shifted + 128.0), 0, 255).astype(np.uint8)
    recon_img = crop_to(recon_img, (orig_h, orig_w))

    # 7. Bitstream serialization
    # Zigzag LL coefficients
    flat_ll = quant_ll.reshape(n_blocks, 64)[:, ZIGZAG_INDICES].astype(np.int16)

    # Pack payload with zlib compressed binary streams
    ll_bytes = zlib.compress(flat_ll.tobytes(), level=6)
    lh_bytes = zlib.compress(q_lh.astype(np.int16).tobytes(), level=6)
    hl_bytes = zlib.compress(q_hl.astype(np.int16).tobytes(), level=6)
    hh_bytes = zlib.compress(q_hh.astype(np.int16).tobytes(), level=6)

    # Header: Magic (8B) + orig_h (2B) + orig_w (2B) + pad_h (2B) + pad_w (2B) +
    #         quality (1B) + detail_step (4B float) +
    #         ll_len (4B) + lh_len (4B) + hl_len (4B) + hh_len (4B) = 37 bytes
    header = bytearray(HYBRID_MAGIC)
    header.extend(
        struct.pack(
            ">HHHHBfIIII",
            orig_h,
            orig_w,
            pad_h,
            pad_w,
            quality,
            detail_step,
            len(ll_bytes),
            len(lh_bytes),
            len(hl_bytes),
            len(hh_bytes),
        )
    )

    final_bytes = bytes(header) + ll_bytes + lh_bytes + hl_bytes + hh_bytes
    return final_bytes, recon_img


def hybrid_decode(
    data: bytes,
    wavelet: str = "bior2.2",
) -> np.ndarray:
    """Decode a Hybrid DCT-DWT compressed bitstream into a 2D uint8 grayscale image.

    Args:
        data: Compressed byte stream.
        wavelet: Wavelet filter used during encoding (default 'bior2.2').

    Returns:
        2D uint8 reconstructed numpy array.

    Raises:
        ValueError: If bitstream header is invalid or corrupt.
    """
    if len(data) < 37:
        raise ValueError(f"Bitstream data too short for Hybrid header ({len(data)} bytes)")

    magic = data[:8]
    if magic != HYBRID_MAGIC:
        raise ValueError(f"Invalid magic bytes '{magic}', expected '{HYBRID_MAGIC}'")

    (
        orig_h,
        orig_w,
        pad_h,
        pad_w,
        quality,
        detail_step,
        ll_len,
        lh_len,
        hl_len,
        hh_len,
    ) = struct.unpack_from(">HHHHBfIIII", data, 8)

    offset = 37
    total_expected = offset + ll_len + lh_len + hl_len + hh_len
    if len(data) < total_expected:
        raise ValueError(f"Corrupt bitstream: expected {total_expected} bytes, got {len(data)}")

    ll_bytes = data[offset : offset + ll_len]
    offset += ll_len
    lh_bytes = data[offset : offset + lh_len]
    offset += lh_len
    hl_bytes = data[offset : offset + hl_len]
    offset += hl_len
    hh_bytes = data[offset : offset + hh_len]
    offset += hh_len

    # Subband geometry
    ll_h = pad_h // 2
    ll_w = pad_w // 2
    n_bh = ll_h // 8
    n_bw = ll_w // 8
    n_blocks = n_bh * n_bw

    # 1. Decode LL DCT coefficients
    flat_ll = np.frombuffer(zlib.decompress(ll_bytes), dtype=np.int16).reshape(n_blocks, 64)
    quant_ll = flat_ll[:, INV_ZIGZAG_INDICES].reshape(n_blocks, 8, 8)

    # 2. Dequantize LL and inverse 8x8 DCT
    q_table = get_quantization_table(quality)
    dequant_ll = quant_ll.astype(np.float32) * q_table
    idct_blocks = scipy.fft.idctn(dequant_ll, axes=(-2, -1), norm="ortho")
    ll_recon = idct_blocks.reshape(n_bh, n_bw, 8, 8).transpose(0, 2, 1, 3).reshape(ll_h, ll_w)

    # 3. Decode and dequantize detail subbands
    q_lh = np.frombuffer(zlib.decompress(lh_bytes), dtype=np.int16).reshape(ll_h, ll_w)
    q_hl = np.frombuffer(zlib.decompress(hl_bytes), dtype=np.int16).reshape(ll_h, ll_w)
    q_hh = np.frombuffer(zlib.decompress(hh_bytes), dtype=np.int16).reshape(ll_h, ll_w)

    lh_recon = _dequantize_deadzone(q_lh, detail_step)
    hl_recon = _dequantize_deadzone(q_hl, detail_step)
    hh_recon = _dequantize_deadzone(q_hh, detail_step)

    # 4. Inverse 2D DWT reconstruction with periodization
    recon_shifted = pywt.idwt2((ll_recon, (lh_recon, hl_recon, hh_recon)), wavelet=wavelet, mode="periodization")
    recon_shifted = crop_to(recon_shifted, (pad_h, pad_w))
    recon_img = np.clip(np.round(recon_shifted + 128.0), 0, 255).astype(np.uint8)

    return crop_to(recon_img, (orig_h, orig_w))
