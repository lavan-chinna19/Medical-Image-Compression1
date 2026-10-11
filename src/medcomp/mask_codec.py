"""Custom binary mask codec using row-wise run-length and Huffman entropy coding.

Features:
- Row-wise run-length coding (RLE):
  Each row is partitioned into alternating binary runs starting with an explicit 1-bit color flag.
  Run lengths L are parameterized as (category, extra_bits):
    cat = L.bit_length()  # [1..11]
    extra_bits = L - (1 << (cat - 1))  # stored in (cat - 1) bits
  Categories are Huffman-coded with an image-adaptive canonical codebook stored in the header.
- Bounding-box variant (bbox_mode=True):
  Computes [r_min, r_max, c_min, c_max] enclosing all True pixels. Stores the bounding box coordinates
  in the header, and codes only the sub-mask within the box. All pixels outside are reconstructed as False.
- Zero external compression libraries (purely uses src/medcomp/entropy.py).
"""

from __future__ import annotations

import struct
from collections import Counter
from typing import Tuple

import numpy as np

from .entropy import (
    BitReader,
    BitWriter,
    build_canonical_decode_trie,
    build_huffman_code_lengths,
    canonical_codes,
    decode_huffman_symbol,
    deserialize_code_lengths,
    serialize_code_lengths,
)

MASK_MAGIC = b"MEDMSK10"


def _extract_runs_for_row(row: np.ndarray) -> tuple[int, list[int]]:
    """Extract alternating run lengths for a single 1D boolean row.

    Returns:
        tuple[int, list[int]]:
            - start_color: 0 or 1.
            - run_lengths: Sequence of run lengths summing to len(row).
    """
    w = len(row)
    if w == 0:
        return 0, []

    start_color = 1 if row[0] else 0
    runs: list[int] = []
    cur_color = start_color
    cur_len = 0

    for val in row:
        c = 1 if val else 0
        if c == cur_color:
            cur_len += 1
        else:
            runs.append(cur_len)
            cur_color = c
            cur_len = 1
    if cur_len > 0:
        runs.append(cur_len)

    return start_color, runs


def mask_encode(mask: np.ndarray, bbox_mode: bool = False) -> bytes:
    """Encode a 2D boolean mask into compressed bytes.

    Args:
        mask: 2D boolean numpy array.
        bbox_mode: If True, computes bounding box and encodes only interior pixels.

    Returns:
        bytes: Compressed bitstream.
    """
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape={getattr(mask, 'shape', None)}")

    h, w = mask.shape
    mask_bool = mask.astype(bool)

    header = bytearray(MASK_MAGIC)
    header.extend(struct.pack(">HHB", h, w, 1 if bbox_mode else 0))

    if bbox_mode:
        true_indices = np.argwhere(mask_bool)
        if len(true_indices) == 0:
            # Empty mask
            header.append(1)  # is_empty = 1
            return bytes(header)

        header.append(0)  # is_empty = 0
        r_min, c_min = true_indices.min(axis=0)
        r_max, c_max = true_indices.max(axis=0)
        header.extend(struct.pack(">HHHH", int(r_min), int(r_max), int(c_min), int(c_max)))
        target_mask = mask_bool[r_min : r_max + 1, c_min : c_max + 1]
    else:
        target_mask = mask_bool

    t_h, t_w = target_mask.shape

    # Decompose into rows of runs
    row_starts: list[int] = []
    all_runs: list[list[int]] = []
    all_categories: list[int] = []

    for r in range(t_h):
        start_c, runs = _extract_runs_for_row(target_mask[r])
        row_starts.append(start_c)
        all_runs.append(runs)
        for length in runs:
            cat = length.bit_length()
            all_categories.append(cat)

    if not all_categories:
        # Edge case: empty grid
        freqs = Counter({1: 1})
    else:
        freqs = Counter(all_categories)

    lengths = build_huffman_code_lengths(freqs)
    codes = canonical_codes(lengths)
    header.extend(serialize_code_lengths(lengths))

    writer = BitWriter()
    for start_c, runs in zip(row_starts, all_runs):
        writer.write_bit(start_c)
        for length in runs:
            cat = length.bit_length()
            c_bits, c_len = codes[cat]
            writer.write_bits(c_bits, c_len)
            extra_bits_count = cat - 1
            if extra_bits_count > 0:
                extra_val = length - (1 << extra_bits_count)
                writer.write_bits(extra_val, extra_bits_count)

    payload = writer.pad_and_flush()
    return bytes(header) + payload


def mask_decode(data: bytes) -> np.ndarray:
    """Decode a compressed bitstream into a 2D boolean mask.

    Args:
        data: Compressed bitstream bytes.

    Returns:
        np.ndarray: Reconstructed 2D boolean numpy array.
    """
    if len(data) < 13:
        raise ValueError("Data too short for mask codec header.")

    magic = data[:8]
    if magic != MASK_MAGIC:
        raise ValueError(f"Invalid magic bytes '{magic}', expected '{MASK_MAGIC}'")

    orig_h, orig_w, mode_u8 = struct.unpack_from(">HHB", data, 8)
    offset = 13
    bbox_mode = bool(mode_u8)

    if bbox_mode:
        is_empty = bool(data[offset])
        offset += 1
        if is_empty:
            return np.zeros((orig_h, orig_w), dtype=bool)

        r_min, r_max, c_min, c_max = struct.unpack_from(">HHHH", data, offset)
        offset += 8
        t_h = r_max - r_min + 1
        t_w = c_max - c_min + 1
    else:
        t_h = orig_h
        t_w = orig_w

    code_lengths, n_read = deserialize_code_lengths(data, offset)
    offset += n_read
    trie = build_canonical_decode_trie(code_lengths)
    reader = BitReader(data, byte_offset=offset)

    target_mask = np.zeros((t_h, t_w), dtype=bool)

    for r in range(t_h):
        cur_color = bool(reader.read_bit())
        col_pos = 0
        while col_pos < t_w:
            cat = decode_huffman_symbol(reader, trie)
            if cat == 1:
                run_length = 1
            else:
                extra_bits_count = cat - 1
                extra_val = reader.read_bits(extra_bits_count)
                run_length = (1 << extra_bits_count) + extra_val

            end_pos = min(t_w, col_pos + run_length)
            if cur_color:
                target_mask[r, col_pos:end_pos] = True
            col_pos = end_pos
            cur_color = not cur_color

    if not bbox_mode:
        return target_mask

    full_mask = np.zeros((orig_h, orig_w), dtype=bool)
    full_mask[r_min : r_max + 1, c_min : c_max + 1] = target_mask
    return full_mask
