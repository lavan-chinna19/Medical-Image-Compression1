"""Comparative Analysis Engine for Medical Image Compression Techniques.

Compares standard baseline techniques (DCT, DWT, JPEG, JPEG2000, Hybrid DCT-DWT)
against the proposed Region-Aware Medical Image Compression pipeline across 9 metrics:
1. Accuracy (%) -> "N/A" (Image reconstruction is a pixel regression task without binary class boundaries)
2. Precision (%) -> "N/A"
3. Recall (%) -> "N/A"
4. R² Score -> Coefficient of determination between original and reconstructed pixel intensities
5. PSNR (dB) -> Peak Signal-to-Noise Ratio (dB), inf for identical images
6. SSIM -> Structural Similarity Index [-1.0, 1.0]
7. MSE -> Mean Squared Error
8. Compression Ratio -> Original raw bytes / Compressed bitstream bytes
9. Reconstruction Time (s) -> Exact measured decompression execution time in seconds
"""

from __future__ import annotations

from dataclasses import dataclass, field
import io
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image

from .baselines import jpeg_codec, jpeg2000_codec
from .dct_codec import dct_decode, dct_encode
from .dwt_codec import dwt_decode, dwt_encode
from .graded_codec import graded_decode, graded_encode
from .hybrid_codec import hybrid_decode, hybrid_encode
from .metrics import mse as calc_mse, psnr as calc_psnr, ssim as calc_ssim
from .roi_codec import roi_decode, roi_encode

TECHNIQUE_NAMES: dict[str, str] = {
    "dct": "DCT",
    "dwt": "DWT",
    "jpeg": "JPEG",
    "jpeg2000": "JPEG2000",
    "hybrid": "Hybrid DCT–DWT",
    "proposed": "Proposed Technique",
}

TECHNIQUE_FULL_NAMES: dict[str, str] = {
    "dct": "DCT — Discrete Cosine Transform",
    "dwt": "DWT — Discrete Wavelet Transform",
    "jpeg": "JPEG — Joint Photographic Experts Group",
    "jpeg2000": "JPEG2000 — Wavelet-Based ISO Standard",
    "hybrid": "Hybrid DCT–DWT — Combined DWT Subbands + Block DCT",
    "proposed": "Proposed Technique — Region-Aware Near-Lossless Graded Codec",
}

TECHNIQUE_ORDER: list[str] = ["dct", "dwt", "jpeg", "jpeg2000", "hybrid", "proposed"]


def compute_r2_score(orig: np.ndarray, recon: np.ndarray) -> float:
    """Compute Coefficient of Determination (R^2 Score) between original and reconstructed pixels.

    Formula:
        R^2 = 1.0 - (SS_res / SS_tot)
        where SS_res = sum((orig - recon)**2) and SS_tot = sum((orig - mean(orig))**2).

    Handles constant-valued images and identical images safely without division by zero.
    """
    if orig.shape != recon.shape:
        raise ValueError(f"Shape mismatch: original {orig.shape} vs reconstructed {recon.shape}")

    y = orig.astype(np.float64).ravel()
    y_hat = recon.astype(np.float64).ravel()

    ss_res = float(np.sum((y - y_hat) ** 2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    r2 = 1.0 - (ss_res / ss_tot)
    return float(r2)


@dataclass
class TechniqueResult:
    """Container holding experimental results for one evaluated compression technique."""

    s_no: int
    technique_key: str
    technique_name: str
    technique_full_name: str
    accuracy: str = "N/A"
    precision: str = "N/A"
    recall: str = "N/A"
    r2_score: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    mse: Optional[float] = None
    compression_ratio: Optional[float] = None
    reconstruction_time: Optional[float] = None
    original_bytes: Optional[int] = None
    compressed_bytes: Optional[int] = None
    recon_img: Optional[np.ndarray] = None
    status: str = "PENDING"
    error_message: str = ""
    settings: dict[str, Any] = field(default_factory=dict)

    def to_display_dict(self) -> dict[str, Any]:
        """Format metrics for UI display and table rendering."""
        if self.status != "SUCCESS":
            return {
                "S. No.": self.s_no,
                "Compression Technique": self.technique_name,
                "Accuracy (%)": "—",
                "Precision (%)": "—",
                "Recall (%)": "—",
                "R² Score": "—",
                "PSNR (dB)": "—",
                "SSIM": "—",
                "MSE": "—",
                "Compression Ratio": "—",
                "Reconstruction Time (s)": "—",
                "Status": self.status,
            }

        psnr_str = "∞ (Identical)" if (self.psnr is not None and np.isinf(self.psnr)) else (f"{self.psnr:.2f}" if self.psnr is not None else "—")
        r2_str = f"{self.r2_score:.4f}" if self.r2_score is not None else "—"
        ssim_str = f"{self.ssim:.4f}" if self.ssim is not None else "—"
        mse_str = f"{self.mse:.2f}" if self.mse is not None else "—"
        cr_str = f"{self.compression_ratio:.2f}:1 ({self.compression_ratio:.2f}x)" if self.compression_ratio is not None else "—"
        time_str = f"{self.reconstruction_time:.4f}" if self.reconstruction_time is not None else "—"

        return {
            "S. No.": self.s_no,
            "Compression Technique": self.technique_name,
            "Accuracy (%)": self.accuracy,
            "Precision (%)": self.precision,
            "Recall (%)": self.recall,
            "R² Score": r2_str,
            "PSNR (dB)": psnr_str,
            "SSIM": ssim_str,
            "MSE": mse_str,
            "Compression Ratio": cr_str,
            "Reconstruction Time (s)": time_str,
            "Status": self.status,
        }

    def to_numeric_dict(self) -> dict[str, Any]:
        """Format metrics with raw numeric types for sorting and export."""
        return {
            "s_no": self.s_no,
            "technique": self.technique_name,
            "technique_full": self.technique_full_name,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "r2_score": self.r2_score,
            "psnr": self.psnr,
            "ssim": self.ssim,
            "mse": self.mse,
            "compression_ratio": self.compression_ratio,
            "reconstruction_time": self.reconstruction_time,
            "original_bytes": self.original_bytes,
            "compressed_bytes": self.compressed_bytes,
            "status": self.status,
            "error": self.error_message,
        }


def get_initial_comparison_df() -> pd.DataFrame:
    """Generate the initial comparison table with dashes '—' for uncomputed metrics."""
    rows = []
    for idx, key in enumerate(TECHNIQUE_ORDER, start=1):
        rows.append({
            "S. No.": idx,
            "Compression Technique": TECHNIQUE_NAMES[key],
            "Accuracy (%)": "—",
            "Precision (%)": "—",
            "Recall (%)": "—",
            "R² Score": "—",
            "PSNR (dB)": "—",
            "SSIM": "—",
            "MSE": "—",
            "Compression Ratio": "—",
            "Reconstruction Time (s)": "—",
        })
    return pd.DataFrame(rows)


def evaluate_technique(
    technique_key: str,
    img_uint8: np.ndarray,
    mask_bool: Optional[np.ndarray] = None,
    params: Optional[dict[str, Any]] = None,
) -> TechniqueResult:
    """Evaluate a single compression technique on a medical image.

    Args:
        technique_key: One of 'dct', 'dwt', 'jpeg', 'jpeg2000', 'hybrid', 'proposed'.
        img_uint8: 2D uint8 numpy array of shape (H, W).
        mask_bool: Optional 2D boolean lung core mask. Used for proposed ROI technique.
        params: Optional dictionary of compression settings.

    Returns:
        TechniqueResult with measured metrics, compressed size, reconstruction time, and decoded image.
    """
    key = technique_key.lower().strip()
    if key not in TECHNIQUE_NAMES:
        raise ValueError(f"Unknown technique '{technique_key}'. Expected one of {TECHNIQUE_ORDER}")

    s_no = TECHNIQUE_ORDER.index(key) + 1
    tech_name = TECHNIQUE_NAMES[key]
    full_name = TECHNIQUE_FULL_NAMES[key]
    params = params or {}

    h, w = img_uint8.shape
    orig_bytes = int(h * w)  # Uncompressed 8-bit raw array representation

    res = TechniqueResult(
        s_no=s_no,
        technique_key=key,
        technique_name=tech_name,
        technique_full_name=full_name,
        original_bytes=orig_bytes,
        settings=dict(params),
    )

    try:
        comp_bytes: bytes
        recon_img: np.ndarray
        recon_time: float

        if key == "dct":
            q = int(params.get("quality", 50))
            comp_bytes, _ = dct_encode(img_uint8, quality=q)
            # Isolated reconstruction time measurement
            t0 = time.perf_counter()
            recon_img = dct_decode(comp_bytes)
            recon_time = time.perf_counter() - t0

        elif key == "dwt":
            q = int(params.get("quality", 50))
            comp_bytes, _ = dwt_encode(img_uint8, quality=q)
            t0 = time.perf_counter()
            recon_img = dwt_decode(comp_bytes)
            recon_time = time.perf_counter() - t0

        elif key == "jpeg":
            q = int(params.get("quality", 50))
            comp_bytes, _ = jpeg_codec(img_uint8, quality=q)
            t0 = time.perf_counter()
            with Image.open(io.BytesIO(comp_bytes)) as pil_im:
                recon_img = np.array(pil_im, dtype=np.uint8)
            recon_time = time.perf_counter() - t0

        elif key == "jpeg2000":
            r = float(params.get("ratio", 10.0))
            comp_bytes, _ = jpeg2000_codec(img_uint8, ratio=r)
            t0 = time.perf_counter()
            with Image.open(io.BytesIO(comp_bytes)) as pil_im:
                recon_img = np.array(pil_im, dtype=np.uint8)
            recon_time = time.perf_counter() - t0

        elif key == "hybrid":
            q = int(params.get("quality", 50))
            comp_bytes, _ = hybrid_encode(img_uint8, quality=q)
            t0 = time.perf_counter()
            recon_img = hybrid_decode(comp_bytes)
            recon_time = time.perf_counter() - t0

        elif key == "proposed":
            # Proposed Technique: Region-Aware Graded Multi-Zone Near-Lossless Codec
            # If no mask is passed, construct a central lung prior mask or full ROI
            if mask_bool is None or not np.any(mask_bool):
                # Fallback to morphological or central prior if mask not provided
                center_mask = np.zeros_like(img_uint8, dtype=bool)
                ch_start, ch_end = int(0.15 * h), int(0.85 * h)
                cw_start, cw_end = int(0.12 * w), int(0.88 * w)
                center_mask[ch_start:ch_end, cw_start:cw_end] = True
                active_mask = center_mask
            else:
                active_mask = mask_bool

            delta_core = int(params.get("delta_core", 2))
            delta_band = int(params.get("delta_band", 4))
            band_px = int(params.get("band_px", 8))
            bg_q = int(params.get("bg_quality", 50))

            graded_params = {
                "delta_core": delta_core,
                "delta_band": delta_band,
                "band_px": band_px,
                "bg_quality": bg_q,
                "bg_method": "dwt",
                "fill": "inpaint",
            }
            comp_bytes, _, _ = graded_encode(img_uint8, active_mask, graded_params)
            t0 = time.perf_counter()
            recon_img = graded_decode(comp_bytes)
            recon_time = time.perf_counter() - t0

        else:
            raise ValueError(f"Unhandled technique: {key}")

        # Compute all 9 metrics
        comp_len = len(comp_bytes)
        cr = float(orig_bytes) / float(comp_len) if comp_len > 0 else 1.0

        r2 = compute_r2_score(img_uint8, recon_img)
        cur_mse = calc_mse(img_uint8, recon_img)
        cur_psnr = calc_psnr(img_uint8, recon_img, data_range=255.0)
        cur_ssim = calc_ssim(img_uint8, recon_img, data_range=255.0)

        res.compressed_bytes = comp_len
        res.recon_img = recon_img
        res.reconstruction_time = max(0.0001, float(recon_time))
        res.compression_ratio = float(cr)
        res.r2_score = float(r2)
        res.mse = float(cur_mse)
        res.psnr = float(cur_psnr)
        res.ssim = float(cur_ssim)
        res.accuracy = "N/A"
        res.precision = "N/A"
        res.recall = "N/A"
        res.status = "SUCCESS"

    except Exception as exc:
        res.status = "FAILED"
        res.error_message = str(exc)

    return res


def run_all_techniques(
    img_uint8: np.ndarray,
    mask_bool: Optional[np.ndarray] = None,
    params: Optional[dict[str, Any]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> list[TechniqueResult]:
    """Execute all six compression techniques sequentially on the same medical image.

    Args:
        img_uint8: 2D uint8 numpy array.
        mask_bool: Optional 2D boolean mask for the lung fields.
        params: Optional dictionary of compression parameters.
        progress_callback: Optional callable(step_idx, total_steps, technique_name).

    Returns:
        List of TechniqueResult objects in canonical order (1 to 6).
    """
    results: list[TechniqueResult] = []
    total = len(TECHNIQUE_ORDER)

    for idx, key in enumerate(TECHNIQUE_ORDER):
        tech_name = TECHNIQUE_NAMES[key]
        if progress_callback:
            progress_callback(idx + 1, total, tech_name)

        res = evaluate_technique(
            technique_key=key,
            img_uint8=img_uint8,
            mask_bool=mask_bool,
            params=params,
        )
        results.append(res)

    return results
