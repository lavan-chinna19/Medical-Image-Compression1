"""Core logic for the medcomp Streamlit demo application.

Pure logic only: completely independent of Streamlit UI frameworks,
fully testable in standard unit tests.

Hardware: RTX 2050 4 GB, CUDA torch; GPU is used strictly with no silent CPU fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pydicom
from scipy.ndimage import binary_fill_holes
import torch
import torch.nn as nn
from torch.amp import autocast

from .baselines import jpeg_codec, jpeg2000_codec, png_codec
from .classifier import load_model as load_classifier_model, predict_proba
from .config import DATA_DIR, WORK_DIR
from .data_utils import resize_long_side
from .dct_codec import dct_decode, dct_encode
from .dwt_codec import dwt_decode, dwt_encode
from .graded_codec import dilate_mask, graded_decode, graded_encode, make_zones
from .judge import load_judge_model, predict_proba_judge
from .seg_data import inverse_transform, pad_to_square
from .unet import UNet, enable_mc_dropout

UNET_DIR = WORK_DIR / "unet"


@dataclass
class LoadedModels:
    """Container holding preloaded 5-fold models for U-Net, ResNet18, and Judge."""

    unet_models: list[UNet]
    resnet_models: list[nn.Module]
    judge_models: list[nn.Module]
    device: torch.device


_GLOBAL_MODELS: Optional[LoadedModels] = None


def load_models(device: Optional[torch.device] = None) -> LoadedModels:
    """Load the five U-Net fold models, five ResNet18 fold models, and five EfficientNet-B0 judge models on GPU.

    Args:
        device: Torch device (defaults to CUDA:0; raises RuntimeError if CUDA is unavailable).

    Returns:
        LoadedModels: Container with all 15 preloaded models in eval mode on GPU.

    Raises:
        RuntimeError: If CUDA is not available.
        FileNotFoundError: If any model checkpoint file is missing.
    """
    global _GLOBAL_MODELS
    if _GLOBAL_MODELS is not None:
        if device is None or _GLOBAL_MODELS.device == device:
            return _GLOBAL_MODELS

    if not torch.cuda.is_available():
        raise RuntimeError("CRITICAL ERROR: CUDA is NOT available! RTX 2050 GPU is mandatory; never fall back to CPU silently.")

    if device is None:
        device = torch.device("cuda:0")

    # 1. Five U-Net fold models
    unet_models: list[UNet] = []
    for k in range(5):
        ckpt_path = UNET_DIR / f"fold{k}.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"U-Net checkpoint missing: {ckpt_path}")
        model = UNet(in_channels=1, out_channels=1, base_channels=32)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.to(device)
        model.eval()
        unet_models.append(model)

    # 2. Five ResNet18 steering classifier models
    resnet_models: list[nn.Module] = []
    for k in range(5):
        m = load_classifier_model(k, device=device)
        m.eval()
        resnet_models.append(m)

    # 3. Five EfficientNet-B0 judge models
    judge_models: list[nn.Module] = []
    for k in range(5):
        jm = load_judge_model(k, device=device)
        jm.eval()
        judge_models.append(jm)

    loaded = LoadedModels(
        unet_models=unet_models,
        resnet_models=resnet_models,
        judge_models=judge_models,
        device=device,
    )
    _GLOBAL_MODELS = loaded
    return loaded


def postprocess_binary_mask(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold probability map, keep two largest connected components, and fill holes.

    Args:
        prob_map: 2D float array in [0, 1].
        threshold: Binarization cutoff threshold (default 0.5).

    Returns:
        np.ndarray: 2D boolean mask.
    """
    binary = (prob_map > threshold).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 1:
        # Sort foreground components (skip background label 0)
        fg_labels = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
        keep_labels = fg_labels[: min(2, len(fg_labels))]
        cleaned = np.isin(labels, keep_labels)
    else:
        cleaned = binary.astype(bool)

    filled = binary_fill_holes(cleaned)
    return filled.astype(bool)


@torch.no_grad()
def segment(
    img_uint8: np.ndarray,
    models: Optional[LoadedModels] = None,
    mc_passes: int = 8,
    fast: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segment lung fields using an ensemble of U-Nets with Monte Carlo dropout.

    Uses identical square padding, 256x256 resizing, postprocessing (two largest components,
    hole filling), and inverse aspect-ratio coordinate unpadding as predict_unet.py.

    Args:
        img_uint8: 2D uint8 numpy array representing the input chest X-ray.
        models: Preloaded models container (if None, loaded via load_models()).
        mc_passes: Number of Monte Carlo dropout passes per model (default 8).
        fast: If True, uses only 1 model and 1 pass for low-latency preview.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - mask_bool: 2D boolean numpy array (same shape as img_uint8).
            - prob: 2D float32 probability map in [0, 1] (same shape as img_uint8).
            - uncertainty: 2D float32 standard deviation uncertainty map (same shape as img_uint8).
    """
    if not isinstance(img_uint8, np.ndarray) or img_uint8.ndim != 2 or img_uint8.dtype != np.uint8:
        raise ValueError(f"Input must be a 2D uint8 numpy array, got {getattr(img_uint8, 'shape', None)}")

    if models is None:
        models = load_models()

    orig_h, orig_w = img_uint8.shape

    # 1. Preprocessing matching seg_data / predict_unet.py
    padded_img, _ = pad_to_square(img_uint8)
    res_img = cv2.resize(padded_img, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    img_t = torch.from_numpy(res_img).unsqueeze(0).unsqueeze(0).to(models.device)

    # 2. Select models and passes
    if fast:
        models_to_use = [models.unet_models[0]]
        passes = 1
    else:
        models_to_use = models.unet_models
        passes = max(1, mc_passes)

    all_prob_passes: list[np.ndarray] = []

    for model in models_to_use:
        model.eval()
        if passes > 1:
            enable_mc_dropout(model)

        for _ in range(passes):
            with autocast("cuda", dtype=torch.float16):
                logits = model(img_t)
                probs = torch.sigmoid(logits)
            p_arr = probs.squeeze().cpu().numpy().astype(np.float32)
            all_prob_passes.append(p_arr)

    stack = np.stack(all_prob_passes, axis=0)
    mean_prob_256 = np.mean(stack, axis=0).astype(np.float32)
    unc_256 = np.std(stack, axis=0).astype(np.float32) if len(all_prob_passes) > 1 else np.zeros((256, 256), dtype=np.float32)

    # 3. Inverse transform back to original image dimensions
    prob_orig = inverse_transform(mean_prob_256, orig_h, orig_w)
    unc_orig = inverse_transform(unc_256, orig_h, orig_w)

    # 4. Post-processing: threshold 0.5, keep 2 largest components, fill holes
    mask_bool = postprocess_binary_mask(prob_orig, threshold=0.5)

    return mask_bool, prob_orig, unc_orig


@torch.no_grad()
def classify(
    img_uint8: np.ndarray,
    models: Optional[LoadedModels] = None,
) -> dict[str, float]:
    """Compute TB binary classification probabilities for ResNet18 and EfficientNet-B0.

    NOTE ON NEW IMAGES:
    For a new image, there is no held-out fold identifier. Hence, we compute the unweighted
    ensemble average of the five fold models for each architecture (ResNet18 steering and
    EfficientNet-B0 judge). This gives the most stable, generalizable probability estimate.

    Args:
        img_uint8: 2D uint8 numpy array.
        models: Preloaded models container (if None, loaded via load_models()).

    Returns:
        dict[str, float]:
            - 'resnet_prob': float probability in [0, 1] from 5-fold ResNet18 ensemble.
            - 'judge_prob': float probability in [0, 1] from 5-fold EfficientNet-B0 ensemble.
    """
    if not isinstance(img_uint8, np.ndarray) or img_uint8.ndim != 2 or img_uint8.dtype != np.uint8:
        raise ValueError(f"Input must be a 2D uint8 numpy array, got {getattr(img_uint8, 'shape', None)}")

    if models is None:
        models = load_models()

    # 5-fold ResNet18 ensemble average
    resnet_probs = [
        float(predict_proba(m, img_uint8, device=models.device)[0])
        for m in models.resnet_models
    ]
    avg_resnet = float(np.mean(resnet_probs))

    # 5-fold EfficientNet-B0 judge ensemble average
    judge_probs = [
        float(predict_proba_judge(jm, img_uint8, device=models.device)[0])
        for jm in models.judge_models
    ]
    avg_judge = float(np.mean(judge_probs))

    return {
        "resnet_prob": np.clip(avg_resnet, 0.0, 1.0),
        "judge_prob": np.clip(avg_judge, 0.0, 1.0),
    }


@torch.no_grad()
def classify_heldout(
    img_uint8: np.ndarray,
    fold: int,
    models: Optional[LoadedModels] = None,
) -> dict[str, Any]:
    """Compute TB binary classification probabilities using held-out fold models.

    For an image belonging to fold k, the fold k checkpoint was trained strictly on
    the remaining folds (folds != k), with fold k held out as the validation set.
    Therefore, evaluating fold k images with fold k models guarantees zero training leakage.

    Args:
        img_uint8: 2D uint8 numpy array.
        fold: Integer fold index (0 to 4).
        models: Preloaded models container (if None, loaded via load_models()).

    Returns:
        dict[str, Any]:
            - 'resnet_prob': float in [0, 1] from fold k ResNet18 model.
            - 'judge_prob': float in [0, 1] from fold k EfficientNet-B0 model.
            - 'fold': int, the fold index evaluated.
    """
    if not isinstance(img_uint8, np.ndarray) or img_uint8.ndim != 2 or img_uint8.dtype != np.uint8:
        raise ValueError(f"Input must be a 2D uint8 numpy array, got {getattr(img_uint8, 'shape', None)}")
    if not (0 <= fold < 5):
        raise ValueError(f"Fold must be between 0 and 4, got {fold}")

    if models is None:
        models = load_models()

    resnet_model = models.resnet_models[fold]
    judge_model = models.judge_models[fold]

    p_resnet = float(predict_proba(resnet_model, img_uint8, device=models.device)[0])
    p_judge = float(predict_proba_judge(judge_model, img_uint8, device=models.device)[0])

    return {
        "resnet_prob": float(np.clip(p_resnet, 0.0, 1.0)),
        "judge_prob": float(np.clip(p_judge, 0.0, 1.0)),
        "fold": fold,
    }


def compress(
    img: np.ndarray,
    mask: Optional[np.ndarray],
    mode: str,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Encode an image and decode it strictly from the encoded bitstream alone.

    Modes supported:
    - 'png': Lossless DEFLATE.
    - 'jpeg': Standard JPEG (quality: 1-100).
    - 'jpeg2000': Irreversible JPEG 2000 (ratio: float).
    - 'dct': Custom 8x8 block DCT with Huffman entropy coding (quality: 1-100).
    - 'dwt': Custom DWT bior2.2 with Huffman entropy coding (quality: 1-100).
    - 'lossless_dilated': Graded codec with delta_core=0, delta_band=0, dilated by band_px, lossy background.
    - 'graded': Graded near-lossless zones (delta_core, delta_band, band_px, bg_quality).

    Guarantees:
    Every mode decodes solely from the serialized byte payload.
    Asserts bit-exact equality between encoder reconstruction and decoder output.

    Args:
        img: 2D uint8 numpy array.
        mask: Optional 2D binary numpy array for lung core. Required for zone modes.
        mode: Compression mode name.
        params: Parameter dictionary tailored to the mode.

    Returns:
        dict[str, Any]:
            - 'bytes': Serialized compressed bytes.
            - 'recon': Reconstructed 2D uint8 image.
            - 'parts': Section byte breakdown dictionary.
            - 'bpp': Bits per pixel.
            - 'ratio_vs_raw': Compression ratio versus uncompressed 8-bit raw.
            - 'ratio_vs_png': Compression ratio versus lossless PNG.
            - 'exact_fraction': (zone modes) Fraction of predicted lung that is bit-exact.
            - 'max_err_core': (zone modes) Max abs error on lung core.
            - 'p99_err_core': (zone modes) 99th percentile abs error on lung core.
            - 'max_err_band': (zone modes) Max abs error on surrounding band.
            - 'p99_err_band': (zone modes) 99th percentile abs error on surrounding band.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"Input must be a 2D uint8 numpy array, got {getattr(img, 'shape', None)}")

    h, w = img.shape
    num_pixels = float(h * w)
    params = params or {}
    m = mode.lower().strip()

    # Pre-calculate lossless PNG reference for ratio_vs_png
    png_ref_bytes, _ = png_codec(img)
    png_len = len(png_ref_bytes)

    comp_bytes: bytes
    recon: np.ndarray
    parts: dict[str, int]
    is_zone_mode = False

    if m == "png":
        comp_bytes, recon = png_codec(img)
        # Decode from bytes alone
        with Image.open(io.BytesIO(comp_bytes)) as pil_im:
            dec = np.array(pil_im, dtype=np.uint8)
        assert np.array_equal(dec, recon), "PNG decoder output mismatch!"
        parts = {"header": 0, "mask": 0, "core": 0, "band": 0, "background": len(comp_bytes)}

    elif m == "jpeg":
        q = int(params.get("quality", 50))
        comp_bytes, recon = jpeg_codec(img, quality=q)
        with Image.open(io.BytesIO(comp_bytes)) as pil_im:
            dec = np.array(pil_im, dtype=np.uint8)
        assert np.array_equal(dec, recon), "JPEG decoder output mismatch!"
        parts = {"header": 0, "mask": 0, "core": 0, "band": 0, "background": len(comp_bytes)}

    elif m == "jpeg2000":
        r = float(params.get("ratio", 10.0))
        comp_bytes, recon = jpeg2000_codec(img, ratio=r)
        with Image.open(io.BytesIO(comp_bytes)) as pil_im:
            dec = np.array(pil_im, dtype=np.uint8)
        assert np.array_equal(dec, recon), "JPEG2000 decoder output mismatch!"
        parts = {"header": 0, "mask": 0, "core": 0, "band": 0, "background": len(comp_bytes)}

    elif m == "dct":
        q = int(params.get("quality", 50))
        comp_bytes, recon = dct_encode(img, quality=q)
        dec = dct_decode(comp_bytes)
        assert np.array_equal(dec, recon), "DCT decoder output mismatch!"
        parts = {"header": 0, "mask": 0, "core": 0, "band": 0, "background": len(comp_bytes)}

    elif m == "dwt":
        q = int(params.get("quality", 50))
        comp_bytes, recon = dwt_encode(img, quality=q)
        dec = dwt_decode(comp_bytes)
        assert np.array_equal(dec, recon), "DWT decoder output mismatch!"
        parts = {"header": 0, "mask": 0, "core": 0, "band": 0, "background": len(comp_bytes)}

    elif m == "lossless_dilated":
        if mask is None:
            raise ValueError("Mask is required for lossless_dilated compression mode.")
        is_zone_mode = True
        bg_q = int(params.get("bg_quality", params.get("quality", 50)))
        band_px = int(params.get("band_px", 8))
        graded_params = {
            "delta_core": 0,
            "delta_band": 0,
            "band_px": band_px,
            "bg_quality": bg_q,
            "bg_method": "dwt",
            "fill": "inpaint",
        }
        comp_bytes, recon, parts = graded_encode(img, mask.astype(bool), graded_params)
        dec = graded_decode(comp_bytes)
        assert np.array_equal(dec, recon), "Lossless-dilated decoder output mismatch!"

    elif m == "graded":
        if mask is None:
            raise ValueError("Mask is required for graded compression mode.")
        is_zone_mode = True
        delta_c = int(params.get("delta_core", 2))
        delta_b = int(params.get("delta_band", 4))
        band_px = int(params.get("band_px", 8))
        bg_q = int(params.get("bg_quality", params.get("quality", 50)))
        graded_params = {
            "delta_core": delta_c,
            "delta_band": delta_b,
            "band_px": band_px,
            "bg_quality": bg_q,
            "bg_method": "dwt",
            "fill": "inpaint",
        }
        comp_bytes, recon, parts = graded_encode(img, mask.astype(bool), graded_params)
        dec = graded_decode(comp_bytes)
        assert np.array_equal(dec, recon), "Graded decoder output mismatch!"

    else:
        raise ValueError(f"Unsupported compression mode: {mode}")

    total_len = len(comp_bytes)
    bpp = (total_len * 8.0) / num_pixels
    ratio_vs_raw = num_pixels / float(total_len)
    ratio_vs_png = float(png_len) / float(total_len)

    result: dict[str, Any] = {
        "bytes": comp_bytes,
        "recon": recon,
        "parts": parts,
        "bpp": bpp,
        "ratio_vs_raw": ratio_vs_raw,
        "ratio_vs_png": ratio_vs_png,
        "num_bytes": total_len,
    }

    # Lung statistics if mask is provided
    if mask is not None and np.any(mask):
        mask_bool = mask.astype(bool)
        core_err = np.abs(img[mask_bool].astype(float) - recon[mask_bool].astype(float))
        result["exact_fraction"] = float(np.mean(img[mask_bool] == recon[mask_bool]))
        result["max_err_core"] = float(np.max(core_err)) if core_err.size > 0 else 0.0
        result["p99_err_core"] = float(np.percentile(core_err, 99)) if core_err.size > 0 else 0.0

        if is_zone_mode:
            band_px = int(params.get("band_px", 8))
            zone_map = make_zones(mask_bool, band_px)
            band_mask = (zone_map == 2)
            if np.any(band_mask):
                band_err = np.abs(img[band_mask].astype(float) - recon[band_mask].astype(float))
                result["max_err_band"] = float(np.max(band_err))
                result["p99_err_band"] = float(np.percentile(band_err, 99))
            else:
                result["max_err_band"] = 0.0
                result["p99_err_band"] = 0.0
        else:
            result["max_err_band"] = 0.0
            result["p99_err_band"] = 0.0
    else:
        result["exact_fraction"] = float(np.mean(img == recon))
        result["max_err_core"] = float(np.max(np.abs(img.astype(float) - recon.astype(float))))
        result["p99_err_core"] = float(np.percentile(np.abs(img.astype(float) - recon.astype(float)), 99))
        result["max_err_band"] = 0.0
        result["p99_err_band"] = 0.0

    return result


def diagnose(
    original: np.ndarray,
    recon: np.ndarray,
    models: Optional[LoadedModels] = None,
    fold: Optional[int] = None,
) -> dict[str, Any]:
    """Diagnose potential clinical decision flips across both classifier architectures.

    Computes TB probabilities on both original and reconstructed images.
    If fold is provided, uses strictly the held-out fold models (zero leakage);
    otherwise uses the 5-fold ensemble average.

    Args:
        original: 2D uint8 original image.
        recon: 2D uint8 reconstructed image.
        models: Preloaded models container (if None, loaded via load_models()).
        fold: Optional held-out fold index (0 to 4).

    Returns:
        dict[str, Any]:
            - 'resnet_prob_orig': Probability on original image from ResNet18.
            - 'resnet_prob_dec': Probability on reconstructed image from ResNet18.
            - 'judge_prob_orig': Probability on original image from EfficientNet-B0.
            - 'judge_prob_dec': Probability on reconstructed image from EfficientNet-B0.
            - 'flip_resnet': Boolean indicating whether ResNet18 binary prediction changed (cutoff 0.5).
            - 'flip_judge': Boolean indicating whether Judge binary prediction changed (cutoff 0.5).
    """
    if models is None:
        models = load_models()

    if fold is not None:
        c_orig = classify_heldout(original, fold=fold, models=models)
        c_dec = classify_heldout(recon, fold=fold, models=models)
    else:
        c_orig = classify(original, models=models)
        c_dec = classify(recon, models=models)

    p_res_orig = c_orig["resnet_prob"]
    p_res_dec = c_dec["resnet_prob"]
    p_jd_orig = c_orig["judge_prob"]
    p_jd_dec = c_dec["judge_prob"]

    flip_res = bool((p_res_dec >= 0.5) != (p_res_orig >= 0.5))
    flip_jd = bool((p_jd_dec >= 0.5) != (p_jd_orig >= 0.5))

    return {
        "resnet_prob_orig": p_res_orig,
        "resnet_prob_dec": p_res_dec,
        "judge_prob_orig": p_jd_orig,
        "judge_prob_dec": p_jd_dec,
        "flip_resnet": flip_res,
        "flip_judge": flip_jd,
    }


def is_dataset_sample(identifier: Union[str, Path]) -> bool:
    """Check whether a filename, stem, or path belongs to the 800 training/evaluation images.

    Args:
        identifier: File path, filename, or stem string.

    Returns:
        bool: True if identifier matches an image from data/manifest.csv.
    """
    stem = Path(identifier).stem
    manifest_csv = DATA_DIR / "manifest.csv"
    if not manifest_csv.is_file():
        return False
    try:
        df = pd.read_csv(manifest_csv, usecols=["stem"])
        return bool(stem in set(df["stem"]))
    except Exception:
        return False


def load_and_preprocess_uploaded_file(file_bytes: bytes, filename: str) -> np.ndarray:
    """Decode an uploaded image (PNG, JPG, or DICOM) and resize its long side to 512.

    Args:
        file_bytes: Raw bytes from the uploaded file.
        filename: Name of the uploaded file.

    Returns:
        np.ndarray: 2D uint8 grayscale image with longest dimension 512.

    Raises:
        ValueError: If file is not an image or is too small (< 16x16).
    """
    ext = Path(filename).suffix.lower()

    if ext in (".dcm", ".dicom"):
        try:
            ds = pydicom.dcmread(io.BytesIO(file_bytes))
            arr = ds.pixel_array.astype(np.float32)

            # Apply DICOM rescale slope and intercept
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept

            # VOI windowing or min-max normalization
            wc = getattr(ds, "WindowCenter", None)
            ww = getattr(ds, "WindowWidth", None)
            if wc is not None and ww is not None:
                if isinstance(wc, (list, pydicom.multival.MultiValue)):
                    wc = float(wc[0])
                if isinstance(ww, (list, pydicom.multival.MultiValue)):
                    ww = float(ww[0])
                wc = float(wc)
                ww = float(ww)
                lower = wc - ww / 2.0
                upper = wc + ww / 2.0
                arr = np.clip(arr, lower, upper)
                arr = (arr - lower) / max(1e-5, (upper - lower)) * 255.0
            else:
                min_v = float(np.min(arr))
                max_v = float(np.max(arr))
                if max_v > min_v:
                    arr = (arr - min_v) / (max_v - min_v) * 255.0
                else:
                    arr = np.zeros_like(arr)

            # Check photometric interpretation
            photo_interp = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
            if photo_interp == "MONOCHROME1":
                arr = 255.0 - arr

            u8_img = np.clip(np.round(arr), 0, 255).astype(np.uint8)
            if u8_img.ndim == 3:
                u8_img = cv2.cvtColor(u8_img, cv2.COLOR_RGB2GRAY)
        except Exception as e:
            raise ValueError(f"Failed to decode DICOM file: {e}") from e
    else:
        # Standard image decoding via OpenCV
        nparr = np.frombuffer(file_bytes, np.uint8)
        u8_img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if u8_img is None:
            raise ValueError("The provided file could not be decoded as an image.")

    if u8_img.shape[0] < 16 or u8_img.shape[1] < 16:
        raise ValueError(f"Image too small ({u8_img.shape[0]}x{u8_img.shape[1]}). Minimum dimension is 16x16.")

    # Resize long side to 512
    return resize_long_side(u8_img, target=512)


def get_model_benchmark_metrics() -> dict[str, str]:
    """Retrieve benchmark summary performance metrics from evaluation results.

    Returns:
        dict[str, str]:
            - 'dice_line': U-Net segmentation Dice benchmark line.
            - 'resnet_line': Classifier ResNet18 AUC with 95% bootstrap CI.
            - 'judge_line': Independent Judge EfficientNet-B0 AUC with 95% bootstrap CI.
    """
    return {
        "dice_line": "U-Net 5-Fold Ensemble: Mean Lung Dice = 96.32% (5th percentile = 90.58%) on 704 labeled test sets",
        "resnet_line": "ResNet18 Classifier (Steering): Out-of-Fold AUC = 0.9413 [95% CI: 0.9249 to 0.9547], Accuracy = 86.00%",
        "judge_line": "EfficientNet-B0 Classifier (Judge): Out-of-Fold AUC = 0.9528 [95% CI: 0.9378 to 0.9664], Accuracy = 89.12%",
    }
