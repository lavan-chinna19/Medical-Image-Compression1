"""Unit tests for Hybrid DCT-DWT Image Codec."""

from __future__ import annotations

import numpy as np
import pytest

from medcomp.hybrid_codec import hybrid_decode, hybrid_encode


def test_hybrid_codec_roundtrip():
    """Verify hybrid encode and decode roundtrip with exact reconstruction match."""
    # Synthetic gradient image
    y, x = np.mgrid[:128, :128]
    img = np.clip(x + y, 0, 255).astype(np.uint8)

    comp_bytes, recon_enc = hybrid_encode(img, quality=60)
    assert len(comp_bytes) > 0
    assert comp_bytes[:8] == b"MEDHYB10"

    recon_dec = hybrid_decode(comp_bytes)
    assert recon_dec.shape == img.shape
    assert recon_dec.dtype == np.uint8
    # Decoder output must match encoder's reconstruction exactly
    assert np.array_equal(recon_enc, recon_dec)


def test_hybrid_codec_different_qualities():
    """Higher quality must produce larger bitstream and lower MSE."""
    img = np.random.randint(40, 220, (128, 128), dtype=np.uint8)
    b_low, rec_low = hybrid_encode(img, quality=20)
    b_high, rec_high = hybrid_encode(img, quality=80)

    assert len(b_high) >= len(b_low)
    mse_low = np.mean((img.astype(float) - rec_low.astype(float)) ** 2)
    mse_high = np.mean((img.astype(float) - rec_high.astype(float)) ** 2)
    assert mse_high <= mse_low


def test_hybrid_codec_arbitrary_dimensions():
    """Image dimensions not multiples of 16 must be handled cleanly."""
    img = np.random.randint(0, 255, (137, 143), dtype=np.uint8)
    comp_bytes, recon = hybrid_encode(img, quality=50)
    decoded = hybrid_decode(comp_bytes)
    assert decoded.shape == (137, 143)
    assert np.array_equal(recon, decoded)


def test_hybrid_codec_invalid_inputs():
    """Invalid inputs must raise ValueError."""
    with pytest.raises(ValueError, match="2D uint8"):
        hybrid_encode(np.zeros((10, 10, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match="too short"):
        hybrid_decode(b"SHORT")

    with pytest.raises(ValueError, match="Invalid magic bytes"):
        hybrid_decode(b"INVALID_MAGIC_HEADER_DATA_PADDING_BYTES_37")
