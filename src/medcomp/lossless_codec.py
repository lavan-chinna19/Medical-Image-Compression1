"""Custom lossless image codec for region of interest (ROI) and full-frame images.

Features:
- MED (Mathematical Edge Detection / LOCO-I / JPEG-LS) predictor:
    pred = min(W, N)  if NW >= max(W, N)
    pred = max(W, N)  if NW <= min(W, N)
    pred = W + N - NW otherwise
- Fully causal, deterministic fallback chain for ROI boundary pixels where
  W, N, NW, NE are outside the image or outside the ROI:
  1. For W: if (c > 0 and in_roi(r, c-1)), use W; elif (r > 0 and in_roi(r-1, c)), use N; else prev_roi_pixel (init 128).
  2. For N: if (r > 0 and in_roi(r-1, c)), use N; elif (c > 0 and in_roi(r, c-1)), use W; else prev_roi_pixel.
  3. For NW: if (r > 0 and c > 0 and in_roi(r-1, c-1)), use NW; elif W available, use W; elif N available, use N; else prev_roi_pixel.
  4. For NE: if (r > 0 and c < W-1 and in_roi(r-1, c+1)), use NE; elif N available, use N; elif W available, use W; else prev_roi_pixel.
- Modular residual wrap:
    diff = int(pixel) - int(pred)
    residual = ((diff + 128) % 256) - 128   # in range [-128, 127]
    pixel = (pred + residual) % 256         # exact inverse
- Adaptive canonical Huffman coding of size categories (0..8) and amplitude bits.
- Context-based option (context=True): quantizes local gradients (|W-NW| + |N-NW| + |N-NE|)
  into 4 activity classes with dedicated canonical Huffman tables stored in the header.
- Zero third-party compression libraries (relying purely on src/medcomp/entropy.py).
"""

from __future__ import annotations

import struct
from collections import Counter
from typing import Any, Sequence

import numpy as np

from .entropy import (
    BitReader,
    BitWriter,
    average_code_length,
    build_canonical_decode_trie,
    build_huffman_code_lengths,
    canonical_codes,
    compute_category,
    decode_amplitude,
    decode_huffman_symbol,
    deserialize_code_lengths,
    encode_amplitude,
    serialize_code_lengths,
    shannon_entropy_from_freqs,
)

LOSSLESS_MAGIC = b"MEDLS10"
NUM_GRADIENT_CONTEXTS = 4


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


def get_gradient_context(w: int, n: int, nw: int, ne: int) -> int:
    """Quantize local gradient activity into one of 4 context classes [0..3].

    Activity metric:
        G = |W - NW| + |N - NW| + |N - NE|

    Thresholds:
        - Context 0: G <= 4   (smooth / flat)
        - Context 1: 5 <= G <= 16  (gentle gradient)
        - Context 2: 17 <= G <= 40 (moderate texture)
        - Context 3: G > 40   (sharp edge / high activity)
    """
    g = abs(w - nw) + abs(n - nw) + abs(n - ne)
    if g <= 4:
        return 0
    elif g <= 16:
        return 1
    elif g <= 40:
        return 2
    return 3


def lossless_encode_with_stats(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    context: bool = False,
) -> tuple[bytes, dict[str, float]]:
    """Losslessly compress image (or masked ROI pixels) with MED prediction and Huffman coding.

    Args:
        img: 2D uint8 numpy array.
        mask: Optional boolean 2D numpy array of shape (H, W). If None, all pixels are coded.
        context: If True, uses 4 gradient-quantized context classes with separate Huffman tables.

    Returns:
        tuple[bytes, dict[str, float]]:
            - Compressed bitstream bytes.
            - Statistics dictionary containing entropy, avg_code_length, and pixel counts.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(f"Expected 2D uint8 array, got shape={getattr(img, 'shape', None)}, dtype={getattr(img, 'dtype', None)}")

    h, w_img = img.shape
    if mask is not None:
        if mask.shape != img.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match image shape {img.shape}")
        mask_bool = mask.astype(bool)
    else:
        mask_bool = None

    # Determine coordinates of pixels to code
    if mask_bool is not None:
        coords = [(r, c) for r in range(h) for c in range(w_img) if mask_bool[r, c]]
    else:
        coords = [(r, c) for r in range(h) for c in range(w_img)]

    n_pixels = len(coords)

    # Fast exit for empty ROI
    if n_pixels == 0:
        header = bytearray(LOSSLESS_MAGIC)
        header.extend(struct.pack(">HHBBIB", h, w_img, 1 if mask_bool is not None else 0, 1 if context else 0, 0, 0))
        stats = {
            "n_pixels": 0,
            "residual_entropy": 0.0,
            "avg_code_length": 0.0,
        }
        return bytes(header), stats

    # Pass 1: Compute predictions, modular residuals, size categories, and contexts
    residuals: list[int] = []
    categories: list[int] = []
    contexts: list[int] = []

    # Local buffer to track coded ROI pixel values for causal neighbor retrieval
    recon = np.zeros((h, w_img), dtype=np.int32)
    prev_val = 128

    for r, c in coords:
        w_avail = (c > 0) and (mask_bool is None or mask_bool[r, c - 1])
        n_avail = (r > 0) and (mask_bool is None or mask_bool[r - 1, c])
        nw_avail = (r > 0 and c > 0) and (mask_bool is None or mask_bool[r - 1, c - 1])
        ne_avail = (r > 0 and c < w_img - 1) and (mask_bool is None or mask_bool[r - 1, c + 1])

        # Causal fallback chain
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

        if ne_avail:
            ne_val = int(recon[r - 1, c + 1])
        elif n_avail:
            ne_val = int(recon[r - 1, c])
        elif w_avail:
            ne_val = int(recon[r, c - 1])
        else:
            ne_val = prev_val

        pred = med_predict(w_val, n_val, nw_val)
        pix = int(img[r, c])

        # Modular difference wrapped to [-128, 127]
        diff = pix - pred
        res = ((diff + 128) % 256) - 128
        cat = compute_category(res)

        residuals.append(res)
        categories.append(cat)

        if context:
            ctx = get_gradient_context(w_val, n_val, nw_val, ne_val)
            contexts.append(ctx)

        recon[r, c] = pix
        prev_val = pix

    # Build Huffman tables
    writer = BitWriter()
    header = bytearray(LOSSLESS_MAGIC)
    header.extend(struct.pack(">HHBBIB", h, w_img, 1 if mask_bool is not None else 0, 1 if context else 0, n_pixels, NUM_GRADIENT_CONTEXTS if context else 1))

    if not context:
        freqs = Counter(categories)
        lengths = build_huffman_code_lengths(freqs)
        codes = canonical_codes(lengths)
        header.extend(serialize_code_lengths(lengths))

        h_entropy = shannon_entropy_from_freqs(freqs)
        avg_len = average_code_length(freqs, lengths)

        for res, cat in zip(residuals, categories):
            c_bits, c_len = codes[cat]
            writer.write_bits(c_bits, c_len)
            if cat > 0:
                amp = encode_amplitude(res, cat)
                writer.write_bits(amp, cat)

    else:
        # Separate tables for each context
        ctx_freqs: list[Counter[int]] = [Counter() for _ in range(NUM_GRADIENT_CONTEXTS)]
        for cat, ctx in zip(categories, contexts):
            ctx_freqs[ctx][cat] += 1

        ctx_lengths: list[dict[int, int]] = []
        ctx_codes: list[dict[int, tuple[int, int]]] = []

        total_cat_bits = 0
        for ctx_idx in range(NUM_GRADIENT_CONTEXTS):
            cf = ctx_freqs[ctx_idx]
            if not cf:
                # Dummy code for empty context
                cf = Counter({0: 1})
            lens = build_huffman_code_lengths(cf)
            ctx_lengths.append(lens)
            ctx_codes.append(canonical_codes(lens))
            header.extend(serialize_code_lengths(lens))

        overall_freqs = Counter(categories)
        h_entropy = shannon_entropy_from_freqs(overall_freqs)

        for res, cat, ctx in zip(residuals, categories, contexts):
            c_bits, c_len = ctx_codes[ctx][cat]
            writer.write_bits(c_bits, c_len)
            total_cat_bits += c_len
            if cat > 0:
                amp = encode_amplitude(res, cat)
                writer.write_bits(amp, cat)

        avg_len = float(total_cat_bits / max(1, n_pixels))

    payload = writer.pad_and_flush()
    final_bytes = bytes(header) + payload

    stats = {
        "n_pixels": n_pixels,
        "residual_entropy": round(h_entropy, 4),
        "avg_code_length": round(avg_len, 4),
    }
    return final_bytes, stats


def lossless_encode(
    img: np.ndarray,
    mask: np.ndarray | None = None,
    context: bool = False,
) -> bytes:
    """Encode an image or masked ROI losslessly using MED prediction and Huffman coding."""
    data, _ = lossless_encode_with_stats(img, mask=mask, context=context)
    return data


def lossless_decode(
    data: bytes,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Decode a compressed lossless bitstream into a 2D uint8 grayscale image.

    Args:
        data: Compressed bitstream bytes.
        mask: Optional boolean 2D numpy array. If the bitstream was encoded with a mask,
              the identical mask must be supplied to reconstruct the ROI pixels.

    Returns:
        np.ndarray: Reconstructed 2D uint8 image (non-ROI pixels are 0).
    """
    if len(data) < 18:
        raise ValueError("Data too short for lossless codec header.")

    magic = data[:7]
    if magic != LOSSLESS_MAGIC:
        raise ValueError(f"Invalid magic bytes '{magic}', expected '{LOSSLESS_MAGIC}'")

    h, w_img, has_mask_u8, ctx_u8, n_pixels, n_tables = struct.unpack_from(">HHBBIB", data, 7)
    offset = 18

    has_mask = bool(has_mask_u8)
    use_context = bool(ctx_u8)

    if has_mask and mask is None:
        raise ValueError("Bitstream was encoded with a mask; mask must be provided for decoding.")

    recon = np.zeros((h, w_img), dtype=np.uint8)
    if n_pixels == 0:
        return recon

    if mask is not None:
        if mask.shape != (h, w_img):
            raise ValueError(f"Provided mask shape {mask.shape} does not match header shape ({h}, {w_img})")
        mask_bool = mask.astype(bool)
    else:
        mask_bool = None

    coords = [(r, c) for r in range(h) for c in range(w_img) if (mask_bool is None or mask_bool[r, c])]
    if len(coords) != n_pixels:
        raise ValueError(f"Number of mask True pixels ({len(coords)}) does not match header n_pixels ({n_pixels})")

    # Deserialize Huffman tables
    if not use_context:
        lengths, n_read = deserialize_code_lengths(data, offset)
        offset += n_read
        trie = build_canonical_decode_trie(lengths)
    else:
        ctx_tries = []
        for _ in range(n_tables):
            lengths, n_read = deserialize_code_lengths(data, offset)
            offset += n_read
            ctx_tries.append(build_canonical_decode_trie(lengths))

    reader = BitReader(data, byte_offset=offset)
    prev_val = 128

    for r, c in coords:
        w_avail = (c > 0) and (mask_bool is None or mask_bool[r, c - 1])
        n_avail = (r > 0) and (mask_bool is None or mask_bool[r - 1, c])
        nw_avail = (r > 0 and c > 0) and (mask_bool is None or mask_bool[r - 1, c - 1])
        ne_avail = (r > 0 and c < w_img - 1) and (mask_bool is None or mask_bool[r - 1, c + 1])

        # Identical causal fallback chain
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

        if ne_avail:
            ne_val = int(recon[r - 1, c + 1])
        elif n_avail:
            ne_val = int(recon[r - 1, c])
        elif w_avail:
            ne_val = int(recon[r, c - 1])
        else:
            ne_val = prev_val

        pred = med_predict(w_val, n_val, nw_val)

        if not use_context:
            cat = decode_huffman_symbol(reader, trie)
        else:
            ctx = get_gradient_context(w_val, n_val, nw_val, ne_val)
            cat = decode_huffman_symbol(reader, ctx_tries[ctx])

        amp = reader.read_bits(cat) if cat > 0 else 0
        res = decode_amplitude(amp, cat)

        pix = (pred + res) % 256
        recon[r, c] = pix
        prev_val = pix

    return recon
