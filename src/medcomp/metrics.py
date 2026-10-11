"""Distortion, rate, and fidelity evaluation metrics for medical image compression."""

from typing import Literal, Optional, Sequence, Union
import numpy as np
from skimage.metrics import structural_similarity


def mse(orig: np.ndarray, recon: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE) between original and reconstructed images.

    Args:
        orig: Original image array.
        recon: Reconstructed image array.

    Returns:
        Mean squared error as a float.

    Raises:
        ValueError: If array shapes do not match.
    """
    if orig.shape != recon.shape:
        raise ValueError(f"Shape mismatch: original {orig.shape} vs reconstructed {recon.shape}")

    diff = orig.astype(np.float64) - recon.astype(np.float64)
    return float(np.mean(diff ** 2))


def psnr(orig: np.ndarray, recon: np.ndarray, data_range: float = 255.0) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Returns float('inf') for identical images (zero MSE).

    Args:
        orig: Original image array.
        recon: Reconstructed image array.
        data_range: Peak dynamic range (default 255.0 for 8-bit images).

    Returns:
        PSNR value in dB, or inf if images are numerically identical.

    Raises:
        ValueError: If array shapes do not match or data_range is non-positive.
    """
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    err = mse(orig, recon)
    if err == 0.0:
        return float("inf")

    return float(10.0 * np.log10((data_range ** 2) / err))


def ssim(orig: np.ndarray, recon: np.ndarray, data_range: float = 255.0) -> float:
    """Compute Structural Similarity Index (SSIM) between two images.

    Uses scikit-image's structural_similarity implementation.

    Args:
        orig: Original image array.
        recon: Reconstructed image array.
        data_range: Dynamic range of input images (default 255.0 for 8-bit images).

    Returns:
        SSIM index in [-1.0, 1.0], where 1.0 indicates perfect structural match.

    Raises:
        ValueError: If array shapes do not match.
    """
    if orig.shape != recon.shape:
        raise ValueError(f"Shape mismatch: original {orig.shape} vs reconstructed {recon.shape}")

    return float(structural_similarity(orig, recon, data_range=data_range))


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    """Compute Compression Ratio (CR = uncompressed_size / compressed_size).

    Args:
        original_bytes: Number of bytes in uncompressed representation.
        compressed_bytes: Number of bytes in compressed bitstream/archive.

    Returns:
        Compression ratio (e.g. 4.0 means compressed data is 1/4 the original size).

    Raises:
        ValueError: If original_bytes < 0 or compressed_bytes <= 0.
    """
    if original_bytes < 0:
        raise ValueError(f"original_bytes must be non-negative, got {original_bytes}")
    if compressed_bytes <= 0:
        raise ValueError(f"compressed_bytes must be positive, got {compressed_bytes}")

    return float(original_bytes / compressed_bytes)


def bits_per_pixel(compressed_bytes: int, num_pixels: int) -> float:
    """Compute rate in bits per pixel (BPP = total_compressed_bits / total_pixels).

    Args:
        compressed_bytes: Total compressed size in bytes.
        num_pixels: Total number of pixels in image (H * W).

    Returns:
        Bits per pixel (bpp).

    Raises:
        ValueError: If compressed_bytes < 0 or num_pixels <= 0.
    """
    if compressed_bytes < 0:
        raise ValueError(f"compressed_bytes must be non-negative, got {compressed_bytes}")
    if num_pixels <= 0:
        raise ValueError(f"num_pixels must be positive, got {num_pixels}")

    return float((compressed_bytes * 8.0) / num_pixels)


def shannon_entropy(array: np.ndarray) -> float:
    """Compute empirical Shannon entropy in bits per symbol.

    H(X) = -sum(p_i * log2(p_i)) over all unique symbols in the array.
    A constant array has entropy 0.0. An ideal uniformly distributed 8-bit array has entropy ~8.0.

    Args:
        array: Input numpy array of discrete symbol values.

    Returns:
        Empirical entropy in bits per symbol.
    """
    if array.size == 0:
        return 0.0

    _, counts = np.unique(array, return_counts=True)
    probabilities = counts / array.size
    # Filter out zero probabilities (p * log2(p) -> 0 as p -> 0)
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def masked_psnr(
    orig: np.ndarray,
    recon: np.ndarray,
    mask: np.ndarray,
    region: Literal["roi", "background"] = "roi",
    data_range: float = 255.0,
) -> float:
    """Compute PSNR restricted to either the ROI (mask > 0) or background (mask == 0).

    Args:
        orig: Original image array.
        recon: Reconstructed image array.
        mask: Binary or label mask array matching image shape.
        region: Target region, either 'roi' or 'background'.
        data_range: Dynamic range of input images (default 255.0).

    Returns:
        PSNR value in dB over the specified region, or inf if identical in that region.

    Raises:
        ValueError: If shapes mismatch, region is invalid, or the selected region has 0 pixels.
    """
    if orig.shape != recon.shape:
        raise ValueError(f"Shape mismatch between original {orig.shape} and reconstructed {recon.shape}")
    if orig.shape != mask.shape:
        raise ValueError(f"Shape mismatch between image {orig.shape} and mask {mask.shape}")

    region_key = region.lower().strip()
    if region_key == "roi":
        selection = mask > 0
    elif region_key == "background":
        selection = mask == 0
    else:
        raise ValueError(f"Invalid region '{region}'. Must be 'roi' or 'background'.")

    num_selected = int(np.count_nonzero(selection))
    if num_selected == 0:
        raise ValueError(f"Selected region '{region}' contains 0 pixels in the provided mask.")

    diff = orig[selection].astype(np.float64) - recon[selection].astype(np.float64)
    region_mse = float(np.mean(diff ** 2))

    if region_mse == 0.0:
        return float("inf")

    return float(10.0 * np.log10((data_range ** 2) / region_mse))


def masked_ssim(
    orig: np.ndarray,
    recon: np.ndarray,
    mask: np.ndarray,
    region: Literal["roi", "background"] = "roi",
    data_range: float = 255.0,
) -> float:
    """Compute Structural Similarity Index (SSIM) restricted to either ROI or background.

    Computes full SSIM map using skimage.metrics.structural_similarity and averages
    the values over the selected region.

    Args:
        orig: Original image array.
        recon: Reconstructed image array.
        mask: Binary or label mask array matching image shape.
        region: Target region, either 'roi' or 'background'.
        data_range: Dynamic range of input images (default 255.0).

    Returns:
        Mean SSIM over the specified region, or NaN if the region contains 0 pixels.

    Raises:
        ValueError: If array shapes mismatch or region parameter is invalid.
    """
    if orig.shape != recon.shape:
        raise ValueError(f"Shape mismatch between original {orig.shape} and reconstructed {recon.shape}")
    if orig.shape != mask.shape:
        raise ValueError(f"Shape mismatch between image {orig.shape} and mask {mask.shape}")

    region_key = region.lower().strip()
    if region_key == "roi":
        selection = mask > 0
    elif region_key == "background":
        selection = mask == 0
    else:
        raise ValueError(f"Invalid region '{region}'. Must be 'roi' or 'background'.")

    num_selected = int(np.count_nonzero(selection))
    if num_selected == 0:
        return float("nan")

    _, ssim_map = structural_similarity(orig, recon, data_range=data_range, full=True)
    return float(np.mean(ssim_map[selection]))


def is_lossless(
    orig: np.ndarray,
    recon: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> bool:
    """Check whether reconstruction is strictly bit-for-bit lossless.

    If mask is provided, checks lossless reconstruction specifically within the ROI (mask > 0).
    If mask is None, checks lossless reconstruction across the entire image.

    Args:
        orig: Original image array.
        recon: Reconstructed image array.
        mask: Optional mask array where positive values denote the lossless ROI.

    Returns:
        True if all examined pixel values are strictly identical, False otherwise.

    Raises:
        ValueError: If array shapes do not match.
    """
    if orig.shape != recon.shape:
        raise ValueError(f"Shape mismatch: original {orig.shape} vs reconstructed {recon.shape}")

    if mask is not None:
        if orig.shape != mask.shape:
            raise ValueError(f"Shape mismatch between image {orig.shape} and mask {mask.shape}")
        selection = mask > 0
        return bool(np.array_equal(orig[selection], recon[selection]))

    return bool(np.array_equal(orig, recon))


def bd_psnr(
    bpp_ref: Sequence[float],
    psnr_ref: Sequence[float],
    bpp_test: Sequence[float],
    psnr_test: Sequence[float],
) -> float:
    """Compute Bjontegaard Delta Peak Signal-to-Noise Ratio (BD-PSNR).

    Fits a cubic polynomial of PSNR against log10(bpp) for both reference and
    test rate-distortion curves, then computes the average PSNR difference
    integrated over the overlapping log10(bpp) range.

    Args:
        bpp_ref: Sequence of bitrates for reference codec.
        psnr_ref: Sequence of PSNR values for reference codec.
        bpp_test: Sequence of bitrates for test codec.
        psnr_test: Sequence of PSNR values for test codec.

    Returns:
        Average delta PSNR in dB (positive means test curve has better PSNR at equal rate).
        Returns NaN if overlapping range is empty or invalid.

    Raises:
        ValueError: If fewer than 4 valid points are provided for either curve.
    """
    def _clean_curve(bpp_seq: Sequence[float], psnr_seq: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        b = np.asarray(bpp_seq, dtype=np.float64)
        p = np.asarray(psnr_seq, dtype=np.float64)
        valid = (b > 0) & np.isfinite(b) & np.isfinite(p)
        b_clean = b[valid]
        p_clean = p[valid]
        if len(b_clean) < 4:
            raise ValueError(f"BD-PSNR requires at least 4 rate-distortion points, got {len(b_clean)}")
        idx = np.argsort(b_clean)
        return np.log10(b_clean[idx]), p_clean[idx]

    x_r, y_r = _clean_curve(bpp_ref, psnr_ref)
    x_t, y_t = _clean_curve(bpp_test, psnr_test)

    x_min = max(float(x_r.min()), float(x_t.min()))
    x_max = min(float(x_r.max()), float(x_t.max()))

    if x_max <= x_min:
        return float("nan")

    poly_r = np.polyfit(x_r, y_r, 3)
    poly_t = np.polyfit(x_t, y_t, 3)

    int_r = np.polyint(poly_r)
    int_t = np.polyint(poly_t)

    val_r = float(np.polyval(int_r, x_max) - np.polyval(int_r, x_min))
    val_t = float(np.polyval(int_t, x_max) - np.polyval(int_t, x_min))

    return float((val_t - val_r) / (x_max - x_min))
