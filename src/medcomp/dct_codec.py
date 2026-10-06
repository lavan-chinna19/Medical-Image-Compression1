"""Custom discrete cosine transform (DCT) image codec with scratch-built entropy coding.

Pipeline:
1. Pad image to multiple of 8x8 using edge-replication.
2. Level-shift (subtract 128).
3. 2D 8x8 block DCT with `scipy.fft.dctn(norm="ortho")` vectorized over all blocks.
4. Quantization using standard JPEG luminance matrix scaled by quality (1-100).
5. Zigzag scanning of quantized DCT coefficients.
6. Differential DC coding + run-length AC coding with amplitude categorization.
7. Adaptive Canonical Huffman entropy coding (separate DC and AC tables per image).
8. Bitstream serialization with compact table headers.
9. Dequantization, 2D IDCT, level-shift restoration, clipping, rounding, and cropping.
"""

from __future__ import annotations

import struct
from collections import Counter
from typing import Sequence

import numpy as np
import scipy.fft

from .entropy import (
    BitReader,
    BitWriter,
    average_code_length,
    build_canonical_decode_trie,
    build_huffman_code_lengths,
    canonical_codes,
    decode_huffman_symbol,
    deserialize_code_lengths,
    serialize_code_lengths,
    shannon_entropy_from_freqs,
)
from .io_utils import crop_to, pad_to_multiple

# Magic bytes identifying this custom DCT bitstream format (8 bytes)
DCT_MAGIC = b"MEDDCT10"

# Standard JPEG Luminance Quantization Table (50% quality benchmark)
Q_LUMINANCE = np.array([
    [16, 11, 10, 16,  24,  40,  51,  61],
    [12, 12, 14, 19,  26,  58,  60,  55],
    [14, 13, 16, 24,  40,  57,  69,  56],
    [14, 17, 22, 29,  51,  87,  80,  62],
    [18, 22, 37, 56,  68, 109, 103,  77],
    [24, 35, 55, 64,  81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float32)


def get_quantization_table(
    quality: int,
    base_q: np.ndarray = Q_LUMINANCE,
) -> np.ndarray:
    """Scale the base quantization table according to the standard IJG quality formula.

    Args:
        quality: Integer quality factor in [1, 100].
        base_q: Base 8x8 quantization matrix (default: JPEG luminance).

    Returns:
        np.ndarray: Scaled 8x8 integer quantization matrix clamped to [1, 255].
    """
    quality = max(1, min(100, int(quality)))
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality

    q = np.floor((base_q * scale + 50.0) / 100.0)
    q = np.clip(q, 1, 255)
    return q.astype(np.float32)


def _generate_zigzag_indices() -> tuple[np.ndarray, np.ndarray]:
    """Generate forward and inverse index arrays for 8x8 zigzag scanning."""
    order = []
    for s in range(15):
        if s % 2 == 0:
            for r in range(s, -1, -1):
                c = s - r
                if r < 8 and c < 8:
                    order.append((r, c))
        else:
            for c in range(s, -1, -1):
                r = s - c
                if r < 8 and c < 8:
                    order.append((r, c))

    forward = np.array([r * 8 + c for r, c in order], dtype=int)
    inverse = np.zeros(64, dtype=int)
    for i, idx in enumerate(forward):
        inverse[idx] = i

    return forward, inverse


ZIGZAG_INDICES, INV_ZIGZAG_INDICES = _generate_zigzag_indices()


def compute_category(val: int) -> int:
    """Determine the JPEG magnitude category (number of bits needed for amplitude)."""
    if val == 0:
        return 0
    return abs(val).bit_length()


def encode_amplitude(val: int, size: int) -> int:
    """Encode an integer difference into its JPEG standard ones' complement representation."""
    if size == 0:
        return 0
    if val > 0:
        return val
    return val + (1 << size) - 1


def decode_amplitude(code_val: int, size: int) -> int:
    """Decode a JPEG ones' complement amplitude value back to a signed integer."""
    if size == 0:
        return 0
    if code_val >= (1 << (size - 1)):
        return code_val
    return code_val - (1 << size) + 1


def dct_encode_with_stats(
    img: np.ndarray,
    quality: int = 50,
    q_table: np.ndarray | None = None,
) -> tuple[bytes, np.ndarray, dict[str, float]]:
    """Compress a 2D grayscale image using 8x8 block DCT and custom canonical Huffman coding.

    Returns:
        tuple[bytes, np.ndarray, dict[str, float]]:
            - bitstream: Complete serialized bytes including headers and Huffman tables.
            - recon: The encoder's exact internal reconstruction (dequantized + IDCT).
            - stats: Dictionary containing dc_entropy, dc_avg_code_length,
                     ac_entropy, and ac_avg_code_length.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError(
            f"Expected 2D uint8 array, got shape={getattr(img, 'shape', None)}, "
            f"dtype={getattr(img, 'dtype', None)}"
        )

    orig_h, orig_w = img.shape

    # 1. Pad image to multiple of 8
    padded = pad_to_multiple(img, 8)
    h_pad, w_pad = padded.shape
    n_bh, n_bw = h_pad // 8, w_pad // 8
    n_blocks = n_bh * n_bw

    # 2. Level shift (subtract 128)
    shifted = padded.astype(np.float32) - 128.0

    # 3. Vectorized 2D DCT across all 8x8 blocks
    blocks = shifted.reshape(n_bh, 8, n_bw, 8).transpose(0, 2, 1, 3).reshape(n_blocks, 8, 8)
    dct_blocks = scipy.fft.dctn(blocks, axes=(-2, -1), norm="ortho")

    # 4. Quantization
    q = q_table if q_table is not None else get_quantization_table(quality)
    quant_blocks = np.round(dct_blocks / q).astype(np.int32)

    # 5. Encoder's own reconstruction path (Dequantize + IDCT)
    dequant_blocks = quant_blocks.astype(np.float32) * q
    idct_blocks = scipy.fft.idctn(dequant_blocks, axes=(-2, -1), norm="ortho")
    recon_shifted = idct_blocks.reshape(n_bh, n_bw, 8, 8).transpose(0, 2, 1, 3).reshape(h_pad, w_pad)
    recon = np.clip(np.round(recon_shifted + 128.0), 0, 255).astype(np.uint8)
    recon = crop_to(recon, (orig_h, orig_w))

    # 6. Zigzag scan blocks into flat (N, 64) representation
    flat_quant = quant_blocks.reshape(n_blocks, 64)[:, ZIGZAG_INDICES]

    # 7. Collect DC differences and AC run-length symbols
    dc_diffs: list[int] = []
    prev_dc = 0
    for b in range(n_blocks):
        curr_dc = int(flat_quant[b, 0])
        diff = curr_dc - prev_dc
        dc_diffs.append(diff)
        prev_dc = curr_dc

    dc_categories = [compute_category(d) for d in dc_diffs]
    dc_amps = [encode_amplitude(d, sz) for d, sz in zip(dc_diffs, dc_categories)]

    # AC tokens per block: list of (sym, sz, amp_bits)
    blocks_ac_tokens: list[list[tuple[int, int, int]]] = []
    all_ac_symbols: list[int] = []

    for b in range(n_blocks):
        ac_coeffs = flat_quant[b, 1:]
        tokens: list[tuple[int, int, int]] = []
        r = 0
        for i in range(63):
            val = int(ac_coeffs[i])
            if val == 0:
                r += 1
            else:
                while r >= 16:
                    tokens.append((0xF0, 0, 0))  # ZRL (16 zeros)
                    all_ac_symbols.append(0xF0)
                    r -= 16
                sz = abs(val).bit_length()
                sym = (r << 4) | sz
                amp_bits = encode_amplitude(val, sz)
                tokens.append((sym, sz, amp_bits))
                all_ac_symbols.append(sym)
                r = 0
        if r > 0:
            tokens.append((0x00, 0, 0))  # EOB
            all_ac_symbols.append(0x00)

        blocks_ac_tokens.append(tokens)

    # 8. Adaptive Huffman table generation
    dc_freqs = Counter(dc_categories)
    ac_freqs = Counter(all_ac_symbols)

    dc_lengths = build_huffman_code_lengths(dc_freqs)
    ac_lengths = build_huffman_code_lengths(ac_freqs)

    dc_codes = canonical_codes(dc_lengths)
    ac_codes = canonical_codes(ac_lengths)

    # 9. Compute entropy stats
    dc_entropy = shannon_entropy_from_freqs(dc_freqs)
    dc_avg_len = average_code_length(dc_freqs, dc_lengths)
    ac_entropy = shannon_entropy_from_freqs(ac_freqs)
    ac_avg_len = average_code_length(ac_freqs, ac_lengths)

    stats = {
        "dc_entropy": round(dc_entropy, 4),
        "dc_avg_code_length": round(dc_avg_len, 4),
        "ac_entropy": round(ac_entropy, 4),
        "ac_avg_code_length": round(ac_avg_len, 4),
    }

    # 10. Encode bitstream
    writer = BitWriter()
    for b in range(n_blocks):
        # DC: symbol + amplitude bits
        cat = dc_categories[b]
        code, length = dc_codes[cat]
        writer.write_bits(code, length)
        if cat > 0:
            writer.write_bits(dc_amps[b], cat)

        # AC: tokens
        for sym, sz, amp_bits in blocks_ac_tokens[b]:
            code, length = ac_codes[sym]
            writer.write_bits(code, length)
            if sz > 0:
                writer.write_bits(amp_bits, sz)

    bitstream_payload = writer.pad_and_flush()

    # 11. Assemble final binary data with header
    # Header format:
    # - 8 bytes: DCT_MAGIC
    # - 2 bytes: orig_h (uint16)
    # - 2 bytes: orig_w (uint16)
    # - 1 byte: quality (uint8)
    # - Serialized DC code lengths table
    # - Serialized AC code lengths table
    # - Bitstream payload
    header = bytearray(DCT_MAGIC)
    header.extend(struct.pack(">HHB", orig_h, orig_w, quality))
    header.extend(serialize_code_lengths(dc_lengths))
    header.extend(serialize_code_lengths(ac_lengths))

    final_bytes = bytes(header) + bitstream_payload
    return final_bytes, recon, stats


def dct_encode(
    img: np.ndarray,
    quality: int = 50,
    q_table: np.ndarray | None = None,
) -> tuple[bytes, np.ndarray]:
    """Encode a 2D grayscale image using DCT and custom Huffman entropy coding.

    Args:
        img: 2D uint8 numpy array.
        quality: Compression quality factor in [1, 100] (default: 50).
        q_table: Optional custom 8x8 quantization matrix.

    Returns:
        tuple[bytes, np.ndarray]:
            - compressed_bytes: Real serialized bitstream bytes including header.
            - recon: Exact internal reconstructed image.
    """
    comp_bytes, recon, _ = dct_encode_with_stats(img, quality=quality, q_table=q_table)
    return comp_bytes, recon


def dct_decode(
    data: bytes,
    q_table: np.ndarray | None = None,
) -> np.ndarray:
    """Decode a compressed DCT bitstream into a 2D uint8 grayscale image.

    Parses header metadata, deserializes canonical Huffman tables, decodes all blocks,
    applies dequantization and 2D IDCT, clips, and crops back to original dimensions.

    Args:
        data: Compressed bitstream bytes.
        q_table: Optional custom 8x8 quantization matrix (if None, derived from quality).

    Returns:
        np.ndarray: Reconstructed 2D uint8 image.

    Raises:
        ValueError: If bitstream header is invalid or corrupt.
    """
    if len(data) < 13:
        raise ValueError("Bitstream data too short for DCT header.")

    # 1. Parse header
    magic = data[:8]
    if magic != DCT_MAGIC:
        raise ValueError(f"Invalid magic bytes '{magic}', expected '{DCT_MAGIC}'")

    orig_h, orig_w, quality = struct.unpack_from(">HHB", data, 8)
    offset = 13

    # 2. Deserialize Huffman tables
    dc_lengths, bytes_dc = deserialize_code_lengths(data, offset)
    offset += bytes_dc

    ac_lengths, bytes_ac = deserialize_code_lengths(data, offset)
    offset += bytes_ac

    # 3. Derive geometry
    h_pad = ((orig_h + 7) // 8) * 8
    w_pad = ((orig_w + 7) // 8) * 8
    n_bh, n_bw = h_pad // 8, w_pad // 8
    n_blocks = n_bh * n_bw

    # 4. Quantization matrix
    q = q_table if q_table is not None else get_quantization_table(quality)

    # 5. Build decoding tries
    dc_trie = build_canonical_decode_trie(dc_lengths)
    ac_trie = build_canonical_decode_trie(ac_lengths)

    # 6. Read bitstream
    reader = BitReader(data, byte_offset=offset)
    flat_quant = np.zeros((n_blocks, 64), dtype=np.int32)
    prev_dc = 0

    for b in range(n_blocks):
        # DC decoding
        cat = decode_huffman_symbol(reader, dc_trie)
        amp = reader.read_bits(cat) if cat > 0 else 0
        diff = decode_amplitude(amp, cat)
        curr_dc = prev_dc + diff
        prev_dc = curr_dc
        flat_quant[b, 0] = curr_dc

        # AC decoding
        k = 1
        while k < 64:
            sym = decode_huffman_symbol(reader, ac_trie)
            if sym == 0x00:  # EOB
                break
            if sym == 0xF0:  # ZRL
                k += 16
                continue
            run = sym >> 4
            sz = sym & 0x0F
            k += run
            amp = reader.read_bits(sz) if sz > 0 else 0
            val = decode_amplitude(amp, sz)
            flat_quant[b, k] = val
            k += 1

    # 7. Inverse zigzag scan and dequantize
    quant_blocks = flat_quant[:, INV_ZIGZAG_INDICES].reshape(n_blocks, 8, 8)
    dequant_blocks = quant_blocks.astype(np.float32) * q

    # 8. 2D IDCT
    idct_blocks = scipy.fft.idctn(dequant_blocks, axes=(-2, -1), norm="ortho")
    recon_shifted = idct_blocks.reshape(n_bh, n_bw, 8, 8).transpose(0, 2, 1, 3).reshape(h_pad, w_pad)

    # 9. Level-shift restore, clip to [0, 255], round, crop
    recon = np.clip(np.round(recon_shifted + 128.0), 0, 255).astype(np.uint8)
    return crop_to(recon, (orig_h, orig_w))
