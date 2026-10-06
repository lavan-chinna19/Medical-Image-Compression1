"""Unit and integration tests for custom DWT image codec."""

from pathlib import Path
import numpy as np
import pytest

from medcomp.config import PROCESSED_IMAGES_DIR
from medcomp.dwt_codec import dwt_decode, dwt_encode, dwt_encode_with_stats
from medcomp.io_utils import load_image
from medcomp.metrics import psnr


def make_synthetic_texture(shape: tuple[int, int] = (128, 128), seed: int = 42) -> np.ndarray:
    """Generate a reproducible noisy texture gradient image."""
    rng = np.random.default_rng(seed)
    h, w = shape
    y = np.linspace(0, 180, h)[:, None]
    x = np.linspace(0, 70, w)[None, :]
    base = y + x
    noise = rng.normal(0, 15, size=shape)
    return np.clip(np.round(base + noise), 0, 255).astype(np.uint8)


def test_decode_equals_encoder_recon():
    """Decoder output must bit-for-bit equal the encoder's internal reconstruction."""
    img = make_synthetic_texture((128, 128), seed=1)
    comp_bytes, recon_enc = dwt_encode(img, quality=50, wavelet="bior4.4", levels=4)
    recon_dec = dwt_decode(comp_bytes)

    assert recon_dec.shape == img.shape
    assert np.array_equal(recon_dec, recon_enc)


def test_non_multiple_odd_sizes():
    """Codec must support arbitrary non-multiple odd dimensions like 67x93 and 93x67."""
    for shape in [(67, 93), (93, 67)]:
        img = make_synthetic_texture(shape, seed=2)
        comp_bytes, recon_enc = dwt_encode(img, quality=60, wavelet="bior4.4", levels=4)
        recon_dec = dwt_decode(comp_bytes)

        assert recon_dec.shape == shape
        assert np.array_equal(recon_dec, recon_enc)


def test_constant_and_all_zero_images():
    """All-zero and constant flat images must encode and decode cleanly."""
    for img in [
        np.zeros((64, 64), dtype=np.uint8),
        np.full((64, 64), 128, dtype=np.uint8),
        np.full((80, 80), 255, dtype=np.uint8),
    ]:
        comp_bytes, recon_enc = dwt_encode(img, quality=50, wavelet="bior4.4", levels=3)
        recon_dec = dwt_decode(comp_bytes)

        assert recon_dec.shape == img.shape
        assert np.array_equal(recon_dec, recon_enc)


def test_size_decreases_monotonically_with_quality():
    """Compressed size must decrease monotonically as quality decreases."""
    img = make_synthetic_texture((128, 128), seed=3)
    qualities = [90, 70, 50, 30, 20, 10]
    sizes = []

    for q in qualities:
        comp_bytes, _ = dwt_encode(img, quality=q, wavelet="bior4.4", levels=4)
        sizes.append(len(comp_bytes))

    # Strict monotonicity across standard quality steps
    for i in range(len(sizes) - 1):
        assert sizes[i] > sizes[i + 1], f"Size at Q={qualities[i]} ({sizes[i]}) not > Q={qualities[i+1]} ({sizes[i+1]})"


def test_real_image_quality_90_psnr_above_38db():
    """On a real processed chest X-ray, Q=90 PSNR must exceed 38 dB and decodes identically."""
    if not PROCESSED_IMAGES_DIR.is_dir():
        pytest.skip("Processed images directory work/processed/images not found.")

    candidates = sorted(PROCESSED_IMAGES_DIR.glob("*.png"))
    if not candidates:
        pytest.skip("No processed PNG images found.")

    img = load_image(candidates[0])
    comp_bytes, recon_enc, stats = dwt_encode_with_stats(img, quality=90, wavelet="bior4.4", levels=4)
    recon_dec = dwt_decode(comp_bytes)

    # Decode bit-for-bit equality
    assert np.array_equal(recon_dec, recon_enc)

    # PSNR threshold verification
    p = psnr(img, recon_enc)
    assert p > 38.0, f"Expected PSNR > 38.0 dB at Q=90, got {p:.2f} dB"

    # Entropy stats sanity check
    assert stats["ll_avg_code_length"] >= stats["ll_entropy"]
    assert stats["detail_avg_code_length"] >= stats["detail_entropy"]


def test_deterministic_output():
    """Encoding the same image with the same parameters must produce identical bitstreams."""
    img = make_synthetic_texture((128, 128), seed=4)
    bytes1, recon1 = dwt_encode(img, quality=50, wavelet="bior4.4", levels=4)
    bytes2, recon2 = dwt_encode(img, quality=50, wavelet="bior4.4", levels=4)

    assert bytes1 == bytes2
    assert np.array_equal(recon1, recon2)


def test_all_four_wavelets_and_levels():
    """Verify all four wavelets ('haar', 'bior2.2', 'bior4.4', 'db2') across levels 3, 4, 5."""
    img = make_synthetic_texture((96, 96), seed=5)
    wavelets = ["haar", "bior2.2", "bior4.4", "db2"]
    levels_list = [3, 4, 5]

    for w in wavelets:
        for lvl in levels_list:
            comp_bytes, recon_enc = dwt_encode(img, quality=50, wavelet=w, levels=lvl)
            recon_dec = dwt_decode(comp_bytes)

            assert recon_dec.shape == img.shape, f"Failed shape check for {w} level {lvl}"
            assert np.array_equal(recon_dec, recon_enc), f"Recon mismatch for {w} level {lvl}"


def test_per_level_tables_option():
    """Verify per_level_tables=True flag encodes and decodes identically."""
    img = make_synthetic_texture((128, 128), seed=6)
    comp_bytes, recon_enc, _ = dwt_encode_with_stats(
        img, quality=50, wavelet="bior4.4", levels=4, per_level_tables=True, legacy=True
    )
    recon_dec = dwt_decode(comp_bytes)

    assert np.array_equal(recon_dec, recon_enc)


@pytest.mark.parametrize("scheme", ["default", "uniform", "fine_1.15", "fine_0.9"])
def test_parameter_combinations_weights_roundtrip(scheme: str):
    """Test exact decode(encode) round trip for each subband weight scheme."""
    img = make_synthetic_texture((64, 64), seed=7)
    comp_bytes, recon_enc = dwt_encode(img, quality=40, weight_scheme=scheme)
    recon_dec = dwt_decode(comp_bytes)
    assert np.array_equal(recon_dec, recon_enc)


@pytest.mark.parametrize("theta", [0.5, 0.4, 0.33, 0.25])
@pytest.mark.parametrize("bias", [0.0, 0.1, 0.2, 0.3])
def test_parameter_combinations_deadzone_roundtrip(theta: float, bias: float):
    """Test exact decode(encode) round trip for dead zone theta and reconstruction bias."""
    img = make_synthetic_texture((64, 64), seed=8)
    comp_bytes, recon_enc = dwt_encode(img, quality=50, theta=theta, bias=bias)
    recon_dec = dwt_decode(comp_bytes)
    assert np.array_equal(recon_dec, recon_enc)


@pytest.mark.parametrize("ctx_mode", ["run", "direct", "none"])
def test_parameter_combinations_context_modes_roundtrip(ctx_mode: str):
    """Test exact decode(encode) round trip for context modes."""
    img = make_synthetic_texture((64, 64), seed=9)
    comp_bytes, recon_enc = dwt_encode(img, quality=50, context_mode=ctx_mode)
    recon_dec = dwt_decode(comp_bytes)
    assert np.array_equal(recon_dec, recon_enc)


@pytest.mark.parametrize("wavelet", ["bior2.2", "bior4.4", "db2"])
@pytest.mark.parametrize("levels", [3, 4, 5])
def test_parameter_combinations_wavelets_levels_roundtrip(wavelet: str, levels: int):
    """Test exact decode(encode) round trip across wavelets and decomposition levels."""
    img = make_synthetic_texture((96, 96), seed=10)
    comp_bytes, recon_enc = dwt_encode(img, quality=50, wavelet=wavelet, levels=levels)
    recon_dec = dwt_decode(comp_bytes)
    assert np.array_equal(recon_dec, recon_enc)


def test_context_tables_stored_in_header_decoder_bytes_only():
    """Verify context Huffman tables are embedded in header and decoder needs only bitstream bytes."""
    img = make_synthetic_texture((128, 128), seed=11)
    for mode in ["run", "direct"]:
        comp_bytes, recon_enc = dwt_encode(img, quality=60, context_mode=mode)
        # Header must be MEDDWT20
        assert comp_bytes.startswith(b"MEDDWT20")
        # Decoder must decode with no external side-information or custom tables
        recon_dec = dwt_decode(comp_bytes)
        assert np.array_equal(recon_dec, recon_enc)


def test_legacy_reproduces_old_byte_counts_on_real_images():
    """Verify legacy=True exactly reproduces original v1 byte counts on two real images."""
    if not PROCESSED_IMAGES_DIR.is_dir():
        pytest.skip("Processed images directory work/processed/images not found.")

    path1 = PROCESSED_IMAGES_DIR / "CHNCXR_0001_0.png"
    path2 = PROCESSED_IMAGES_DIR / "CHNCXR_0002_0.png"
    if not path1.is_file() or not path2.is_file():
        pytest.skip("Reference images CHNCXR_0001_0 / CHNCXR_0002_0 not found.")

    img1 = load_image(path1)
    img2 = load_image(path2)

    qualities = [10, 20, 30, 50, 70, 90]
    expected_bytes_1 = [1063, 1673, 2353, 5906, 13490, 29935]
    expected_bytes_2 = [1091, 1713, 2530, 6146, 13637, 30166]

    for q, exp_len in zip(qualities, expected_bytes_1):
        comp_bytes, recon_enc = dwt_encode(img1, quality=q, legacy=True)
        assert comp_bytes.startswith(b"MEDDWT10")
        assert len(comp_bytes) == exp_len, f"Image 1 at Q={q}: got {len(comp_bytes)}, expected {exp_len}"
        recon_dec = dwt_decode(comp_bytes)
        assert np.array_equal(recon_dec, recon_enc)

    for q, exp_len in zip(qualities, expected_bytes_2):
        comp_bytes, recon_enc = dwt_encode(img2, quality=q, legacy=True)
        assert comp_bytes.startswith(b"MEDDWT10")
        assert len(comp_bytes) == exp_len, f"Image 2 at Q={q}: got {len(comp_bytes)}, expected {exp_len}"
        recon_dec = dwt_decode(comp_bytes)
        assert np.array_equal(recon_dec, recon_enc)

