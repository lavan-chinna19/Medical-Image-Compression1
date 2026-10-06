"""Entropy coding module implemented completely from scratch.

Provides:
- BitWriter: Accumulates individual bits and flushes byte streams.
- BitReader: Reads arbitrary bit sequences from byte streams.
- Huffman tree construction using heapq.
- Canonical Huffman code assignment.
- Huffman bitstream encode and decode routines.
- Shannon entropy and average code length calculation (verifying H <= L < H + 1).
- Compact serialization of Huffman code-length tables for bitstream headers.
"""

from __future__ import annotations

import heapq
import struct
from collections import Counter
from typing import Any, Sequence

import numpy as np


class BitWriter:
    """Bit-level writer that accumulates bits into a bytearray."""

    def __init__(self) -> None:
        self._bytes = bytearray()
        self._accumulator = 0
        self._bit_count = 0

    def write_bit(self, bit: int) -> None:
        """Write a single bit (0 or 1)."""
        self._accumulator = (self._accumulator << 1) | (bit & 1)
        self._bit_count += 1
        if self._bit_count == 8:
            self._bytes.append(self._accumulator)
            self._accumulator = 0
            self._bit_count = 0

    def write_bits(self, value: int, n_bits: int) -> None:
        """Write the lower `n_bits` of `value` to the bitstream (MSB first)."""
        if n_bits <= 0:
            return
        # Mask value to ensure only lower n_bits are used
        value &= (1 << n_bits) - 1
        for i in range(n_bits - 1, -1, -1):
            self.write_bit((value >> i) & 1)

    def pad_and_flush(self, pad_bit: int = 0) -> bytes:
        """Pad remaining bits in the current byte with `pad_bit` and return final bytes."""
        if self._bit_count > 0:
            pad_needed = 8 - self._bit_count
            pad_val = ((1 << pad_needed) - 1) if pad_bit else 0
            self._accumulator = (self._accumulator << pad_needed) | pad_val
            self._bytes.append(self._accumulator)
            self._accumulator = 0
            self._bit_count = 0
        return bytes(self._bytes)

    @property
    def byte_count(self) -> int:
        """Number of full bytes emitted plus 1 if a partial byte is pending."""
        return len(self._bytes) + (1 if self._bit_count > 0 else 0)


class BitReader:
    """Bit-level reader that consumes arbitrary bits from a bytes buffer."""

    def __init__(self, data: bytes, byte_offset: int = 0) -> None:
        self._data = data
        self._pos = byte_offset
        self._accumulator = 0
        self._bit_count = 0

    def read_bit(self) -> int:
        """Read a single bit (0 or 1). Raises EOFError if exhausted."""
        if self._bit_count == 0:
            if self._pos >= len(self._data):
                raise EOFError("BitReader: Attempted to read past end of data stream.")
            self._accumulator = self._data[self._pos]
            self._pos += 1
            self._bit_count = 8
        self._bit_count -= 1
        return (self._accumulator >> self._bit_count) & 1

    def read_bits(self, n_bits: int) -> int:
        """Read `n_bits` sequentially and return as an unsigned integer."""
        val = 0
        for _ in range(n_bits):
            val = (val << 1) | self.read_bit()
        return val

    @property
    def byte_offset(self) -> int:
        """Current byte pointer position in the underlying data."""
        return self._pos


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


def build_huffman_code_lengths(freqs: dict[int, int]) -> dict[int, int]:
    """Build optimal Huffman code lengths from symbol frequencies using heapq.

    Handles the edge case of a single distinct symbol by assigning it code length 1.

    Args:
        freqs: Mapping of symbol -> frequency (positive counts).

    Returns:
        dict[int, int]: Mapping of symbol -> code length in bits.
    """
    # Filter non-positive counts
    valid_freqs = {s: f for s, f in freqs.items() if f > 0}
    if not valid_freqs:
        return {}

    # Single distinct symbol edge case
    if len(valid_freqs) == 1:
        single_sym = next(iter(valid_freqs.keys()))
        return {single_sym: 1}

    # Priority queue entries: (weight, unique_id, node)
    heap: list[tuple[int, int, Any]] = []
    uid = 0
    for sym, count in valid_freqs.items():
        heapq.heappush(heap, (count, uid, sym))
        uid += 1

    # Merge nodes until 1 root remains
    while len(heap) > 1:
        f1, _, node1 = heapq.heappop(heap)
        f2, _, node2 = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, uid, (node1, node2)))
        uid += 1

    root = heap[0][2]
    lengths: dict[int, int] = {}

    def _traverse(node: Any, depth: int) -> None:
        if not isinstance(node, tuple):
            lengths[node] = depth
            return
        _traverse(node[0], depth + 1)
        _traverse(node[1], depth + 1)

    _traverse(root, 0)
    return lengths


def canonical_codes(lengths: dict[int, int]) -> dict[int, tuple[int, int]]:
    """Assign canonical Huffman codes to symbols based on their code lengths.

    Canonical ordering:
    - Sort symbols primarily by code length (ascending)
    - Secondarily by symbol value (ascending)
    - Consecutive codes: code = (prev_code + 1) << (curr_len - prev_len)

    Args:
        lengths: Mapping of symbol -> code length.

    Returns:
        dict[int, tuple[int, int]]: Mapping of symbol -> (code_value, code_length).
    """
    if not lengths:
        return {}

    sorted_syms = sorted(lengths.keys(), key=lambda s: (lengths[s], s))
    codebook: dict[int, tuple[int, int]] = {}

    code = 0
    prev_len = 0
    for sym in sorted_syms:
        cur_len = lengths[sym]
        code = code << (cur_len - prev_len)
        codebook[sym] = (code, cur_len)
        code += 1
        prev_len = cur_len

    return codebook


def build_canonical_decode_trie(lengths: dict[int, int]) -> Any:
    """Build a fast binary decoding trie from canonical code lengths.

    Each internal trie node is a 2-element list `[left_child, right_child]`.
    Leaves store the integer symbol directly.
    """
    codebook = canonical_codes(lengths)
    root: list[Any] = [None, None]

    for sym, (code, length) in codebook.items():
        node = root
        for i in range(length - 1, -1, -1):
            bit = (code >> i) & 1
            if i == 0:
                node[bit] = sym
            else:
                if node[bit] is None:
                    node[bit] = [None, None]
                node = node[bit]
    return root


def decode_huffman_symbol(bitreader: BitReader, trie_root: Any) -> int:
    """Decode a single symbol by traversing the binary trie from bitreader."""
    node = trie_root
    while isinstance(node, list):
        bit = bitreader.read_bit()
        node = node[bit]
        if node is None:
            raise ValueError("Corrupt bitstream: invalid Huffman code encountered.")
    return int(node)


def huffman_encode(
    symbols: Sequence[int],
    lengths: dict[int, int],
    bitwriter: BitWriter,
) -> None:
    """Encode a sequence of symbols into the given BitWriter using canonical codes."""
    codebook = canonical_codes(lengths)
    for sym in symbols:
        if sym not in codebook:
            raise KeyError(f"Symbol {sym} not found in Huffman codebook.")
        code, length = codebook[sym]
        bitwriter.write_bits(code, length)


def huffman_decode(
    bitreader: BitReader,
    lengths: dict[int, int],
    n_symbols: int,
) -> list[int]:
    """Decode `n_symbols` from BitReader using the canonical Huffman tree derived from `lengths`."""
    if n_symbols <= 0 or not lengths:
        return []

    trie = build_canonical_decode_trie(lengths)
    decoded: list[int] = []
    for _ in range(n_symbols):
        sym = decode_huffman_symbol(bitreader, trie)
        decoded.append(sym)
    return decoded


def average_code_length(freqs: dict[int, int], lengths: dict[int, int]) -> float:
    """Compute the expected (weighted average) code length in bits/symbol.

    Formula: L = sum(freq_i * length_i) / sum(freq_i)
    """
    total = sum(freqs.values())
    if total <= 0:
        return 0.0
    weighted_bits = sum(freqs[s] * lengths.get(s, 0) for s in freqs if freqs[s] > 0)
    return float(weighted_bits / total)


def shannon_entropy_from_freqs(freqs: dict[int, int]) -> float:
    """Compute empirical Shannon entropy in bits/symbol from symbol frequencies.

    Formula: H = -sum(p_i * log2(p_i))
    """
    total = sum(freqs.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for count in freqs.values():
        if count > 0:
            p = count / total
            h -= p * np.log2(p)
    return float(h)


def serialize_code_lengths(lengths: dict[int, int]) -> bytes:
    """Serialize a code-length table into a compact binary representation.

    Format:
    - 2 bytes (uint16 big-endian): count N of symbols
    - N entries of 2 bytes:
        - 1 byte (uint8): symbol (0-255)
        - 1 byte (uint8): code length (1-255)

    Total header overhead: 2 + 2 * N bytes.
    """
    n_entries = len(lengths)
    if n_entries > 65535:
        raise ValueError(f"Too many symbols to serialize ({n_entries})")

    out = bytearray(struct.pack(">H", n_entries))
    # Sort for deterministic header representation
    for sym in sorted(lengths.keys()):
        length = lengths[sym]
        if not (0 <= sym <= 255):
            raise ValueError(f"Symbol {sym} out of 1-byte range [0, 255]")
        if not (1 <= length <= 255):
            raise ValueError(f"Code length {length} out of 1-byte range [1, 255]")
        out.extend(struct.pack(">BB", sym, length))

    return bytes(out)


def deserialize_code_lengths(data: bytes, offset: int = 0) -> tuple[dict[int, int], int]:
    """Deserialize a code-length table from binary data.

    Args:
        data: Byte buffer containing serialized table.
        offset: Starting byte offset.

    Returns:
        tuple[dict[int, int], int]:
            - lengths: Reconstructed symbol -> code length mapping.
            - bytes_read: Number of bytes consumed from data.
    """
    if len(data) < offset + 2:
        raise EOFError("Insufficient data for code lengths count.")

    (n_entries,) = struct.unpack_from(">H", data, offset)
    cur = offset + 2
    required = cur + 2 * n_entries
    if len(data) < required:
        raise EOFError(f"Insufficient data for {n_entries} code length entries.")

    lengths: dict[int, int] = {}
    for _ in range(n_entries):
        sym, length = struct.unpack_from(">BB", data, cur)
        lengths[sym] = length
        cur += 2

    return lengths, cur - offset
