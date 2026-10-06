"""Unit tests for custom BitWriter, BitReader, and scratch-built Huffman entropy coding."""

from collections import Counter

import numpy as np
import pytest

from medcomp.entropy import (
    BitReader,
    BitWriter,
    average_code_length,
    build_huffman_code_lengths,
    canonical_codes,
    deserialize_code_lengths,
    huffman_decode,
    huffman_encode,
    serialize_code_lengths,
    shannon_entropy_from_freqs,
)


def test_bitwriter_and_bitreader_basic():
    """Verify bit-level writing and reading across byte boundaries."""
    bw = BitWriter()
    # Write patterns: 3 bits, 5 bits, 9 bits, 1 bit
    bw.write_bits(0b101, 3)
    bw.write_bits(0b11001, 5)
    bw.write_bits(0b110100101, 9)
    bw.write_bit(1)

    raw_bytes = bw.pad_and_flush(pad_bit=0)
    assert len(raw_bytes) == 3  # (3 + 5 + 9 + 1) = 18 bits -> 3 bytes (24 bits padded)

    br = BitReader(raw_bytes)
    assert br.read_bits(3) == 0b101
    assert br.read_bits(5) == 0b11001
    assert br.read_bits(9) == 0b110100101
    assert br.read_bit() == 1


def test_huffman_roundtrip_random_symbols():
    """Huffman encoding and canonical decoding must achieve exact symbol reconstruction."""
    rng = np.random.default_rng(42)
    # 2000 symbols drawn from an alphabet of 20 symbols
    symbols = rng.integers(0, 20, size=2000).tolist()
    freqs = Counter(symbols)

    lengths = build_huffman_code_lengths(freqs)
    bw = BitWriter()
    huffman_encode(symbols, lengths, bw)
    encoded_data = bw.pad_and_flush()

    br = BitReader(encoded_data)
    decoded_symbols = huffman_decode(br, lengths, len(symbols))

    assert decoded_symbols == symbols


def test_huffman_single_symbol_stream():
    """Single distinct symbol must be assigned code length 1 and round-trip successfully."""
    symbols = [7] * 500
    freqs = Counter(symbols)

    lengths = build_huffman_code_lengths(freqs)
    assert lengths == {7: 1}

    bw = BitWriter()
    huffman_encode(symbols, lengths, bw)
    encoded_data = bw.pad_and_flush()

    # 500 bits padded to bytes -> ceil(500 / 8) = 63 bytes
    assert len(encoded_data) == 63

    br = BitReader(encoded_data)
    decoded = huffman_decode(br, lengths, len(symbols))
    assert decoded == symbols


def test_huffman_empty_stream():
    """Empty symbol list must handle gracefully without errors."""
    lengths = build_huffman_code_lengths({})
    assert lengths == {}

    bw = BitWriter()
    huffman_encode([], lengths, bw)
    data = bw.pad_and_flush()
    assert len(data) == 0

    br = BitReader(data)
    decoded = huffman_decode(br, lengths, 0)
    assert decoded == []


def test_huffman_entropy_inequality():
    """Verify fundamental data compression theorem: H <= L < H + 1 on a skewed distribution."""
    # Geometric-like distribution: 1/2, 1/4, 1/8, 1/16, ...
    freqs = {0: 1024, 1: 512, 2: 256, 3: 128, 4: 64, 5: 32, 6: 16, 7: 8}
    lengths = build_huffman_code_lengths(freqs)

    H = shannon_entropy_from_freqs(freqs)
    L = average_code_length(freqs, lengths)

    assert H <= L
    assert L < H + 1.0


def test_table_serialization_roundtrip():
    """Code-length table must serialize and deserialize deterministically."""
    original_lengths = {0: 1, 15: 4, 16: 4, 240: 8, 255: 9}
    serialized = serialize_code_lengths(original_lengths)

    # 2 bytes count + 5 * 2 bytes = 12 bytes
    assert len(serialized) == 12

    restored_lengths, bytes_read = deserialize_code_lengths(serialized)
    assert bytes_read == len(serialized)
    assert restored_lengths == original_lengths
