"""Unit tests for image input/output, padding, cropping, and directory filtering."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from medcomp.io_utils import (
    crop_to,
    list_images,
    load_image,
    pad_to_multiple,
    save_image,
)


def test_pad_and_crop_round_trip():
    """Verify pad_to_multiple and crop_to round trip, including exact multiples."""
    # Case 1: Image size is already a multiple of 8
    img_mult = np.arange(64, dtype=np.uint8).reshape((8, 8))
    padded_mult = pad_to_multiple(img_mult, 8)
    assert padded_mult.shape == (8, 8)
    assert np.array_equal(padded_mult, img_mult)
    cropped_mult = crop_to(padded_mult, img_mult.shape)
    assert np.array_equal(cropped_mult, img_mult)

    # Case 2: Image size is NOT a multiple of 8 (e.g., 13 x 27)
    rng = np.random.default_rng(99)
    img_odd = rng.integers(0, 256, size=(13, 27), dtype=np.uint8)
    padded_odd = pad_to_multiple(img_odd, 8)
    # Next multiples: 13 -> 16, 27 -> 32
    assert padded_odd.shape == (16, 32)
    # Replicated border check: last row of original replicated down
    assert np.array_equal(padded_odd[12, :27], padded_odd[13, :27])
    assert np.array_equal(padded_odd[12, :27], padded_odd[15, :27])

    cropped_odd = crop_to(padded_odd, img_odd.shape)
    assert cropped_odd.shape == img_odd.shape
    assert np.array_equal(cropped_odd, img_odd)

    # Case 3: Larger multiple (e.g. 16)
    padded_16 = pad_to_multiple(img_odd, 16)
    assert padded_16.shape == (16, 32)
    assert np.array_equal(crop_to(padded_16, img_odd.shape), img_odd)


def test_save_load_round_trip(tmp_path: Path):
    """Verify lossless save and load round-trip for 8-bit grayscale PNG."""
    rng = np.random.default_rng(101)
    orig_img = rng.integers(0, 256, size=(45, 55), dtype=np.uint8)

    save_path = tmp_path / "test_roundtrip.png"
    save_image(save_path, orig_img)
    loaded_img = load_image(save_path)

    assert loaded_img.shape == orig_img.shape
    assert loaded_img.dtype == np.uint8
    assert np.array_equal(loaded_img, orig_img)


def test_rgb_to_gray_conversion(tmp_path: Path):
    """Verify RGB images are correctly converted to 2D uint8 grayscale on load."""
    h, w = 30, 40
    # Synthetic RGB image with known color values
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_img[:, :, 0] = 50   # Blue
    rgb_img[:, :, 1] = 100  # Green
    rgb_img[:, :, 2] = 150  # Red

    save_path = tmp_path / "color_sample.png"
    cv2.imwrite(str(save_path), rgb_img)

    loaded_gray = load_image(save_path)
    assert loaded_gray.ndim == 2
    assert loaded_gray.shape == (h, w)
    assert loaded_gray.dtype == np.uint8

    # ITU-R 601 formula: Y = 0.299*R + 0.587*G + 0.114*B
    expected_y = int(round(0.299 * 150 + 0.587 * 100 + 0.114 * 50))
    # Loaded value should match expected within rounding tolerance
    assert abs(int(loaded_gray[0, 0]) - expected_y) <= 1


def test_16bit_to_8bit_conversion(tmp_path: Path):
    """Verify 16-bit unsigned images are linearly mapped down to 8-bit."""
    h, w = 10, 10
    # Construct 16-bit image with 0, mid-point (32768), and max (65535)
    img_16 = np.zeros((h, w), dtype=np.uint16)
    img_16[0, 0] = 0
    img_16[5, 5] = 32768
    img_16[9, 9] = 65535

    save_path = tmp_path / "sample_16bit.png"
    cv2.imwrite(str(save_path), img_16)

    loaded_8 = load_image(save_path)
    assert loaded_8.ndim == 2
    assert loaded_8.dtype == np.uint8
    assert loaded_8[0, 0] == 0
    assert loaded_8[5, 5] in (127, 128)  # 32768 * 255 / 65535 = 127.5
    assert loaded_8[9, 9] == 255


def test_list_images_ignores_junk_files(tmp_path: Path):
    """Verify list_images finds supported formats and filters out Thumbs.db, desktop.ini, etc."""
    # Create valid images
    dummy_pixel = np.zeros((4, 4), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "scan_b.png"), dummy_pixel)
    cv2.imwrite(str(tmp_path / "scan_a.jpg"), dummy_pixel)
    cv2.imwrite(str(tmp_path / "scan_c.bmp"), dummy_pixel)
    cv2.imwrite(str(tmp_path / "scan_d.tif"), dummy_pixel)

    # Create junk files
    (tmp_path / "Thumbs.db").write_bytes(b"junk")
    (tmp_path / "thumbs.db").write_bytes(b"junk")
    (tmp_path / "desktop.ini").write_bytes(b"junk")
    (tmp_path / ".DS_Store").write_bytes(b"junk")
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "data.csv").write_text("col1,col2")

    img_list = list_images(tmp_path)
    filenames = [f.name for f in img_list]

    # Verify only valid image files are returned in sorted order
    assert filenames == ["scan_a.jpg", "scan_b.png", "scan_c.bmp", "scan_d.tif"]
