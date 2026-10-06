"""Dataset parsing, mask manipulation, resizing, and metadata utilities."""

import math
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from medcomp.config import TARGET_LONG_SIDE


def source_of(stem: str) -> str:
    """Identify the originating hospital/dataset source from an image or reading stem.

    Args:
        stem: Filename stem (e.g. 'MCUCXR_0001_0' or 'CHNCXR_0001_0').

    Returns:
        'Montgomery' for MCUCXR prefix, 'Shenzhen' for CHNCXR prefix.

    Raises:
        ValueError: If the stem does not match known source prefixes.
    """
    if stem.startswith("MCUCXR"):
        return "Montgomery"
    elif stem.startswith("CHNCXR"):
        return "Shenzhen"
    else:
        raise ValueError(f"Unknown dataset source for stem: {stem}")


def mask_path_for(stem: str, dataset_root: Union[str, Path]) -> Optional[Path]:
    """Resolve the filesystem path of the segmentation mask for a given image stem.

    Handles naming conventions for both Montgomery and Shenzhen datasets:
    - Montgomery: masks/<stem>.png
    - Shenzhen:   masks/<stem>_mask.png

    Args:
        stem: Filename stem of the chest X-ray image.
        dataset_root: Root directory of the dataset.

    Returns:
        Path to the mask file if it exists, otherwise None.
    """
    root = Path(dataset_root)
    masks_dir = root / "masks"

    src = source_of(stem) if stem.startswith(("MCUCXR", "CHNCXR")) else ""

    if src == "Montgomery":
        candidate = masks_dir / f"{stem}.png"
        if candidate.is_file():
            return candidate
    elif src == "Shenzhen":
        candidate = masks_dir / f"{stem}_mask.png"
        if candidate.is_file():
            return candidate

    # Fallback search if source prefix is custom or non-standard
    cand1 = masks_dir / f"{stem}.png"
    if cand1.is_file():
        return cand1
    cand2 = masks_dir / f"{stem}_mask.png"
    if cand2.is_file():
        return cand2

    return None


def parse_reading(path: Union[str, Path]) -> Dict[str, Any]:
    """Parse demographic and clinical findings from a clinical reading text file.

    Handles Montgomery and Shenzhen formats gracefully without crashing:
    - Montgomery:
        Patient's Sex: F
        Patient's Age: 027Y
        <findings>
    - Shenzhen:
        <sex> <age>
        <findings>
      (Includes irregular formats: e.g. 'femal 32yrs', 'male 16month', 'female 64days', 'Female, 28yrs')

    Ages with month or day units are converted to years (months / 12.0, days / 365.25).
    Missing or unparseable fields are returned as None (sex) or NaN (age_years).

    Args:
        path: Path to the clinical reading text file.

    Returns:
        Dictionary with keys:
            - 'sex': 'M', 'F', 'O', or None
            - 'age_years': float (in years) or float('nan')
            - 'findings_text': str (cleaned findings text)
    """
    result: Dict[str, Any] = {
        "sex": None,
        "age_years": float("nan"),
        "findings_text": "",
    }

    reading_path = Path(path)
    if not reading_path.is_file():
        return result

    try:
        content = reading_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return result

    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return result

    filename = reading_path.name

    if filename.startswith("MCUCXR"):
        # Montgomery format
        findings_lines = []
        for line in lines:
            if "Sex:" in line:
                raw_sex = line.split("Sex:", 1)[1].strip().upper()
                if "M" in raw_sex:
                    result["sex"] = "M"
                elif "F" in raw_sex:
                    result["sex"] = "F"
                elif raw_sex:
                    result["sex"] = raw_sex[0]
            elif "Age:" in line:
                raw_age = line.split("Age:", 1)[1].strip()
                result["age_years"] = _parse_age_string(raw_age)
            else:
                findings_lines.append(line)
        result["findings_text"] = " ".join(findings_lines).strip()

    else:
        # Shenzhen format (first line has demographics, subsequent lines have findings)
        first_line = lines[0]
        # Match sex and age from first line
        sex_match = re.search(r"\b(male|female|femal|m|f)\b", first_line, re.IGNORECASE)
        if sex_match:
            val = sex_match.group(1).lower()
            if val in ("male", "m"):
                result["sex"] = "M"
            elif val in ("female", "femal", "f"):
                result["sex"] = "F"

        # Match age and potential units (yrs, month, days, etc.)
        age_match = re.search(r"(\d+(?:\.\d+)?)\s*(yrs|yr|years|year|month|months|days|day|y|m|d)?\b", first_line, re.IGNORECASE)
        if age_match:
            num_str = age_match.group(1)
            unit_str = age_match.group(2) or ""
            result["age_years"] = _convert_age(num_str, unit_str)

        findings_lines = lines[1:] if len(lines) > 1 else []
        result["findings_text"] = " ".join(findings_lines).strip()

    return result


def _parse_age_string(raw_age: str) -> float:
    """Helper to parse Montgomery age string like '027Y' or '10M'."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]*)", raw_age.strip())
    if not match:
        return float("nan")
    num_str = match.group(1)
    unit_str = match.group(2)
    return _convert_age(num_str, unit_str)


def _convert_age(num_str: str, unit_str: str) -> float:
    """Convert age value with unit to years."""
    try:
        val = float(num_str)
    except (ValueError, TypeError):
        return float("nan")

    unit = unit_str.lower().strip()
    if "month" in unit:
        return val / 12.0
    elif "day" in unit:
        return val / 365.25
    else:
        # Default unit is years
        return val


def load_mask(path: Union[str, Path]) -> np.ndarray:
    """Load a binary segmentation mask as a 2D boolean numpy array.

    Pixels with intensity > 127 are considered foreground (True).

    Args:
        path: Filepath to mask image.

    Returns:
        2D boolean numpy array where True denotes lung ROI.

    Raises:
        FileNotFoundError: If the mask file does not exist.
        ValueError: If mask decoding fails.
    """
    mask_path = Path(path).resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    raw_bytes = np.fromfile(str(mask_path), dtype=np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode mask: {mask_path}")

    if img.ndim == 3:
        img = img[:, :, 0]

    return (img > 127).astype(bool)


def resize_mask_nearest(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask to the specified shape using nearest-neighbor interpolation.

    Preserves strict binary boolean values without creating interpolated shades.

    Args:
        mask: 2D boolean or uint8 mask array.
        shape: Target shape as (height, width).

    Returns:
        2D boolean array of shape (height, width).

    Raises:
        ValueError: If shape is invalid or mask is not 2D.
    """
    if len(shape) < 2:
        raise ValueError(f"Shape must be (height, width), got {shape}")
    target_h, target_w = int(shape[0]), int(shape[1])

    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got ndim={mask.ndim}")

    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    resized = cv2.resize(mask_uint8, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return (resized > 127).astype(bool)


def roi_fraction(mask: np.ndarray) -> float:
    """Compute the foreground fraction of the mask (foreground_pixels / total_pixels).

    Args:
        mask: 2D boolean or binary numpy array.

    Returns:
        Fraction of pixels in foreground as a float in [0.0, 1.0].
    """
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask) / mask.size)


def bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute the tight bounding box around the foreground mask.

    Args:
        mask: 2D boolean or binary numpy array.

    Returns:
        Tuple of (min_row, min_col, max_row, max_col) coordinates.
        Returns (0, 0, 0, 0) if mask has no foreground pixels.
    """
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        return (0, 0, 0, 0)

    return (
        int(np.min(rows)),
        int(np.min(cols)),
        int(np.max(rows)),
        int(np.max(cols)),
    )


def resize_long_side(img: np.ndarray, target: int = TARGET_LONG_SIDE) -> np.ndarray:
    """Resize an image so its longest dimension equals target while preserving aspect ratio.

    Uses cv2.INTER_AREA for high-quality downsampling.

    Args:
        img: 2D numpy array of image.
        target: Length of the longer dimension (default 512).

    Returns:
        Resized 2D numpy array.

    Raises:
        ValueError: If target <= 0 or image is not 2D.
    """
    if target <= 0:
        raise ValueError(f"Target dimension must be positive, got {target}")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image array, got shape {img.shape}")

    h, w = img.shape
    if h >= w:
        new_h = target
        new_w = max(1, int(round(w * (target / h))))
    else:
        new_w = target
        new_h = max(1, int(round(h * (target / w))))

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
