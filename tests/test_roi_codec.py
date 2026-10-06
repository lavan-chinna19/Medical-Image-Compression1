"""Tests for hybrid ROI codec (lossless ROI, lossy background)."""

from pathlib import Path
import numpy as np
import pytest

from medcomp.config import PROCESSED_IMAGES_DIR, PROCESSED_MASKS_DIR
from medcomp.io_utils import load_image
from medcomp.metrics import is_lossless
from medcomp.roi_codec import roi_decode, roi_encode


def make_synthetic_test_case(shape: tuple[int, int] = (128, 128), seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible synthetic image and circular ROI mask."""
    rng = np.random.default_rng(seed)
    h, w = shape
    y, x = np.ogrid[:h, :w]
    img = np.clip(y * 1.5 + x * 0.8 + rng.normal(0, 10, size=shape), 0, 255).astype(np.uint8)

    # Circular mask in center
    cy, cx = h // 2, w // 2
    r_val = min(h, w) // 3
    mask = ((y - cy) ** 2 + (x - cx) ** 2) <= (r_val ** 2)
    return img, mask


@pytest.mark.parametrize("method", ["dct", "dwt"])
def test_roi_lossless_roundtrip_basic(method: str):
    """Verify bit-exact ROI reconstruction for both DCT and DWT background codecs."""
    img, mask = make_synthetic_test_case((128, 128), seed=1)
    data, recon, parts = roi_encode(img, mask, method=method, quality=50, fill="inpaint")

    # Decoder uses only bitstream bytes
    recon_dec = roi_decode(data)

    # Exact equality on ROI pixels
    assert np.array_equal(recon_dec[mask], img[mask])
    assert np.array_equal(recon[mask], img[mask])
    assert is_lossless(img, recon_dec, mask=mask)

    # Size consistency
    assert len(data) == sum(parts.values())
    assert parts["header"] == 22


@pytest.mark.parametrize("method", ["dct", "dwt"])
def test_roi_codec_odd_shapes(method: str):
    """Test odd non-multiple shapes (67x93, 93x67)."""
    for shape in [(67, 93), (93, 67)]:
        img, mask = make_synthetic_test_case(shape, seed=2)
        data, recon, _ = roi_encode(img, mask, method=method, quality=60, fill="inpaint")
        recon_dec = roi_decode(data)

        assert recon_dec.shape == shape
        assert np.array_equal(recon_dec[mask], img[mask])


@pytest.mark.parametrize("method", ["dct", "dwt"])
def test_roi_codec_empty_full_and_tiny_masks(method: str):
    """Test empty mask, full mask, and tiny 1-pixel mask."""
    img = np.full((64, 64), 100, dtype=np.uint8)

    # 1. Empty mask
    empty_mask = np.zeros(img.shape, dtype=bool)
    d_empty, _, _ = roi_encode(img, empty_mask, method=method, quality=40)
    r_empty = roi_decode(d_empty)
    assert r_empty.shape == img.shape

    # 2. Full mask
    full_mask = np.ones(img.shape, dtype=bool)
    d_full, _, _ = roi_encode(img, full_mask, method=method, quality=40)
    r_full = roi_decode(d_full)
    assert np.array_equal(r_full, img)

    # 3. Tiny 1-pixel mask
    tiny_mask = np.zeros(img.shape, dtype=bool)
    tiny_mask[20, 20] = True
    d_tiny, _, _ = roi_encode(img, tiny_mask, method=method, quality=40)
    r_tiny = roi_decode(d_tiny)
    assert r_tiny[20, 20] == img[20, 20]


def test_missing_mask_raises_error():
    """Missing or None mask must raise a clear ValueError."""
    img = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="mask is required"):
        roi_encode(img, None)  # type: ignore


def test_deterministic_output():
    """Encoding twice with identical parameters yields identical bitstreams."""
    img, mask = make_synthetic_test_case((64, 64), seed=3)
    d1, r1, _ = roi_encode(img, mask, method="dwt", quality=50, fill="inpaint")
    d2, r2, _ = roi_encode(img, mask, method="dwt", quality=50, fill="inpaint")

    assert d1 == d2
    assert np.array_equal(r1, r2)


def test_real_image_and_mask_roundtrip():
    """Real chest X-ray and lung mask achieves bit-exact ROI reconstruction."""
    if not PROCESSED_IMAGES_DIR.is_dir() or not PROCESSED_MASKS_DIR.is_dir():
        pytest.skip("Processed images/masks directory not found.")

    candidates = sorted(PROCESSED_MASKS_DIR.glob("*.png"))
    if not candidates:
        pytest.skip("No masks found.")

    stem = candidates[0].stem
    img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
    mask_path = PROCESSED_MASKS_DIR / f"{stem}.png"
    if not img_path.is_file():
        pytest.skip(f"Image {img_path} not found.")

    img = load_image(img_path)
    mask = (load_image(mask_path) > 127)

    for method in ["dct", "dwt"]:
        data, recon, _ = roi_encode(img, mask, method=method, quality=50, fill="inpaint")
        recon_dec = roi_decode(data)
        assert np.array_equal(recon_dec[mask], img[mask])
        assert is_lossless(img, recon_dec, mask=mask)
