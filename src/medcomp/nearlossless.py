"""Near-lossless image coder with bounded-error guarantees for medical image zones.

Features:
- MED (LOCO-I / JPEG-LS) predictor:
    pred = min(W, N)  if NW >= max(W, N)
    pred = max(W, N)  if NW <= min(W, N)
    pred = W + N - NW otherwise
- Fully causal, deterministic fallback chain for ROI boundary pixels identical to lossless_codec.
- Bounded-error near-lossless quantization:
    Residual: e = x - pred
    Quantized: q = sign(e) * floor((|e| + delta) / (2 * delta + 1))
    Unclamped reconstruction: recon_val = pred + q * (2 * delta + 1)
    Clamped: recon = clip(recon_val, 0, 255)
  Because 0 <= x <= 255, clamping an out-of-range recon_val to [0, 255] strictly decreases
  the error |x - recon|, preserving the JPEG-LS guarantee |x - recon| <= delta for every pixel.
  When delta = 0, q = e, recon_val = x, reducing bit-exactly to lossless coding.
- Per-zone adaptive canonical Huffman coding using src/medcomp/entropy.py.
- Both zones (1 = core, 2 = band) are interleaved in raster order, with dedicated Huffman tables
  stored in the bitstream header.
"""

from __future__ import annotations

from collections import Counter
import struct
from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .entropy import (
    BitReader,
    BitWriter,
    build_canonical_decode_trie,
    build_huffman_code_lengths,
    canonical_codes,
    compute_category,
    decode_amplitude,
    decode_huffman_symbol,
    deserialize_code_lengths,
    encode_amplitude,
    serialize_code_lengths,
)

NEARLOSSLESS_MAGIC = b"MEDNL10"


def med_predict(w: int, n: int, nw: int) -> int:
    """Compute LOCO-I / JPEG-LS Median Edge Detector (MED) prediction.

    Args:
        w: West neighbor value [0, 255].
        n: North neighbor value [0, 255].
        nw: Northwest neighbor value [0, 255].

    Returns:
        int: Clamped predicted pixel value in [0, 255].
    """
    max_wn = max(w, n)
    min_wn = min(w, n)
    if nw >= max_wn:
        val = min_wn
    elif nw <= min_wn:
        val = max_wn
    else:
        val = w + n - nw
    return max(0, min(255, val))


def nl_encode(
    img: np.ndarray,
    zone_map: np.ndarray,
    deltas: dict[int, int],
) -> tuple[bytes, np.ndarray]:
    """Encode image pixels in zones 1 and 2 with near-lossless error guarantees.

    Args:
        img: 2D uint8 numpy array.
        zone_map: 2D uint8 array matching img shape (0 = uncompressed background, 1 = core, 2 = band).
        deltas: Error bound per zone, e.g. {1: delta_core, 2: delta_band}.

    Returns:
        tuple[bytes, np.ndarray]:
            - Serialized compressed bitstream bytes including headers.
            - Reconstructed 2D uint8 array for the coded zones (0 elsewhere).
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"Expected 2D uint8 image, got shape={getattr(img, 'shape', None)}, dtype={getattr(img, 'dtype', None)}")
    if zone_map.shape != img.shape:
        raise ValueError(f"Zone map shape {zone_map.shape} does not match image {img.shape}")

    h, w_img = img.shape
    delta1 = int(deltas.get(1, 0))
    delta2 = int(deltas.get(2, 0))

    recon = np.zeros((h, w_img), dtype=np.uint8)

    # Coordinates of pixels to code in raster order
    coords = [(r, c) for r in range(h) for c in range(w_img) if zone_map[r, c] in (1, 2)]
    n_total = len(coords)

    # Handle empty coded region
    if n_total == 0:
        header = bytearray(NEARLOSSLESS_MAGIC)
        header.extend(struct.pack(">HHBBIIII", h, w_img, delta1, delta2, 0, 0, 0, 0))
        return bytes(header), recon

    # Pass 1: compute predictions, quantized residuals, size categories per pixel
    zone_cats: dict[int, list[int]] = {1: [], 2: []}
    pixel_records: list[tuple[int, int, int]] = []  # (zone, category, quantized_residual)

    prev_val = 128

    for r, c in coords:
        z = int(zone_map[r, c])
        delta = delta1 if z == 1 else delta2

        w_avail = (c > 0) and (zone_map[r, c - 1] > 0)
        n_avail = (r > 0) and (zone_map[r - 1, c] > 0)
        nw_avail = (r > 0 and c > 0) and (zone_map[r - 1, c - 1] > 0)

        # Causal neighbor fallback chain identical to lossless_codec
        if w_avail:
            w_val = int(recon[r, c - 1])
        elif n_avail:
            w_val = int(recon[r - 1, c])
        else:
            w_val = prev_val

        if n_avail:
            n_val = int(recon[r - 1, c])
        elif w_avail:
            n_val = int(recon[r, c - 1])
        else:
            n_val = prev_val

        if nw_avail:
            nw_val = int(recon[r - 1, c - 1])
        elif w_avail:
            nw_val = int(recon[r, c - 1])
        elif n_avail:
            nw_val = int(recon[r - 1, c])
        else:
            nw_val = prev_val

        pred = med_predict(w_val, n_val, nw_val)
        pix = int(img[r, c])
        e = pix - pred

        # Bounded-error quantization rule
        if delta == 0:
            q = e
            recon_val = pred + q
        else:
            step = 2 * delta + 1
            sign_e = 1 if e > 0 else (-1 if e < 0 else 0)
            q = sign_e * ((abs(e) + delta) // step)
            recon_val = pred + q * step

        # Clamping rule preserving |pix - recon_pix| <= delta
        recon_pix = max(0, min(255, recon_val))
        recon[r, c] = recon_pix
        prev_val = recon_pix

        cat = compute_category(q)
        zone_cats[z].append(cat)
        pixel_records.append((z, cat, q))

    n_z1 = len(zone_cats[1])
    n_z2 = len(zone_cats[2])

    # Build Huffman tables per zone
    codes: dict[int, dict[int, tuple[int, int]]] = {}
    serialized_tables: dict[int, bytes] = {}

    for z in (1, 2):
        if len(zone_cats[z]) > 0:
            freqs = Counter(zone_cats[z])
            lengths = build_huffman_code_lengths(freqs)
            codes[z] = canonical_codes(lengths)
            serialized_tables[z] = serialize_code_lengths(lengths)
        else:
            codes[z] = {}
            serialized_tables[z] = b""

    # Encode bitstream in raster order
    writer = BitWriter()
    z1_bits = 0
    z2_bits = 0

    for z, cat, q in pixel_records:
        c_code, c_len = codes[z][cat]
        writer.write_bits(c_code, c_len)
        bits_added = c_len

        if cat > 0:
            amp = encode_amplitude(q, cat)
            writer.write_bits(amp, cat)
            bits_added += cat

        if z == 1:
            z1_bits += bits_added
        else:
            z2_bits += bits_added

    payload = writer.pad_and_flush()

    # Calculate allocated bytes per zone
    table1_bytes = len(serialized_tables[1])
    table2_bytes = len(serialized_tables[2])
    core_bytes = table1_bytes + int(np.ceil(z1_bits / 8.0))
    band_bytes = table2_bytes + int(np.ceil(z2_bits / 8.0))

    # Build header:
    # Magic (7B) + h (2B) + w (2B) + delta1 (1B) + delta2 (1B) +
    # n_z1 (4B) + n_z2 (4B) + core_bytes (4B) + band_bytes (4B) = 29 bytes
    header = bytearray(NEARLOSSLESS_MAGIC)
    header.extend(struct.pack(">HHBBIIII", h, w_img, delta1, delta2, n_z1, n_z2, core_bytes, band_bytes))
    header.extend(serialized_tables[1])
    header.extend(serialized_tables[2])

    final_bytes = bytes(header) + payload
    return final_bytes, recon


def nl_decode(data: bytes, zone_map: np.ndarray) -> np.ndarray:
    """Decode near-lossless bitstream into 2D uint8 image using supplied zone_map.

    Args:
        data: Serialized compressed bytes.
        zone_map: 2D uint8 array matching image shape.

    Returns:
        np.ndarray: Reconstructed 2D uint8 image with decoded values in zones 1 and 2.
    """
    if len(data) < 29:
        raise ValueError("Data too short for near-lossless header.")

    magic = data[:7]
    if magic != NEARLOSSLESS_MAGIC:
        raise ValueError(f"Invalid magic bytes '{magic}', expected '{NEARLOSSLESS_MAGIC}'")

    h, w_img, delta1, delta2, n_z1, n_z2, core_bytes, band_bytes = struct.unpack_from(">HHBBIIII", data, 7)
    offset = 29

    if zone_map.shape != (h, w_img):
        raise ValueError(f"Zone map shape {zone_map.shape} does not match header dimensions ({h}, {w_img})")

    recon = np.zeros((h, w_img), dtype=np.uint8)

    # Deserialize Huffman tables
    tries: dict[int, dict] = {}
    if n_z1 > 0:
        lengths1, n_read1 = deserialize_code_lengths(data, offset)
        offset += n_read1
        tries[1] = build_canonical_decode_trie(lengths1)
    else:
        tries[1] = {}

    if n_z2 > 0:
        lengths2, n_read2 = deserialize_code_lengths(data, offset)
        offset += n_read2
        tries[2] = build_canonical_decode_trie(lengths2)
    else:
        tries[2] = {}

    coords = [(r, c) for r in range(h) for c in range(w_img) if zone_map[r, c] in (1, 2)]
    if len(coords) != (n_z1 + n_z2):
        raise ValueError(f"Zone map pixels ({len(coords)}) do not match header counts ({n_z1 + n_z2})")

    if len(coords) == 0:
        return recon

    reader = BitReader(data, byte_offset=offset)
    prev_val = 128

    for r, c in coords:
        z = int(zone_map[r, c])
        delta = delta1 if z == 1 else delta2

        w_avail = (c > 0) and (zone_map[r, c - 1] > 0)
        n_avail = (r > 0) and (zone_map[r - 1, c] > 0)
        nw_avail = (r > 0 and c > 0) and (zone_map[r - 1, c - 1] > 0)

        # Causal neighbor fallback chain identical to encoder
        if w_avail:
            w_val = int(recon[r, c - 1])
        elif n_avail:
            w_val = int(recon[r - 1, c])
        else:
            w_val = prev_val

        if n_avail:
            n_val = int(recon[r - 1, c])
        elif w_avail:
            n_val = int(recon[r, c - 1])
        else:
            n_val = prev_val

        if nw_avail:
            nw_val = int(recon[r - 1, c - 1])
        elif w_avail:
            nw_val = int(recon[r, c - 1])
        elif n_avail:
            nw_val = int(recon[r - 1, c])
        else:
            nw_val = prev_val

        pred = med_predict(w_val, n_val, nw_val)

        # Decode symbol
        cat = decode_huffman_symbol(reader, tries[z])
        if cat > 0:
            code_val = reader.read_bits(cat)
            q = decode_amplitude(code_val, cat)
        else:
            q = 0

        if delta == 0:
            recon_val = pred + q
        else:
            step = 2 * delta + 1
            recon_val = pred + q * step

        recon_pix = max(0, min(255, recon_val))
        recon[r, c] = recon_pix
        prev_val = recon_pix

    return recon
