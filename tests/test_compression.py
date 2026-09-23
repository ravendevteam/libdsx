from __future__ import annotations

import zlib

import pytest

from libdsx import DSXError
from libdsx.compression import check_empty_stream


ORDER = (16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15)
FIXED = (8,) * 144 + (9,) * 112 + (7,) * 24 + (8,) * 8


class Bits:
    def __init__(self) -> None:
        self.values: list[int] = []

    def integer(self, value: int, width: int) -> None:
        self.values.extend((value >> index) & 1 for index in range(width))

    def align(self) -> None:
        while len(self.values) % 8:
            self.values.append(0)

    def symbol(self, symbol: int, lengths: tuple[int, ...]) -> None:
        width = lengths[symbol]
        assert width
        code = 0
        for bits in range(1, width + 1):
            code <<= 1
            if bits > 1:
                code += lengths.count(bits - 1) << 1
        code += sum(length == width for length in lengths[:symbol])
        self.values.extend((code >> index) & 1 for index in range(width - 1, -1, -1))

    def bytes(self) -> bytes:
        padded = self.values + [0] * (-len(self.values) % 8)
        return bytes(sum(padded[start + offset] << offset for offset in range(8)) for start in range(0, len(padded), 8))


def wrap(payload: bytes, plain: bytes = b"", window: int = 15) -> bytes:
    cmf = ((window - 8) << 4) | 8
    flg = -(cmf << 8) % 31
    return bytes((cmf, flg)) + payload + zlib.adler32(plain).to_bytes(4, "big")


def stored(bits: Bits, final: bool, value: bytes = b"") -> None:
    bits.integer(final, 1)
    bits.integer(0, 2)
    bits.align()
    bits.integer(len(value), 16)
    bits.integer(len(value) ^ 65535, 16)
    for byte in value:
        bits.integer(byte, 8)


def fixed(bits: Bits, final: bool, value: bytes = b"") -> None:
    bits.integer(final, 1)
    bits.integer(1, 2)
    for byte in value:
        bits.symbol(byte, FIXED)
    bits.symbol(256, FIXED)


def dynamic(bits: Bits, final: bool, style: str = "literal", distance_length: int = 0, value: bytes = b"") -> None:
    code_lengths = tuple({0: 2, 1: 2, 16: 2, 17: 3, 18: 3}.get(symbol, 0) for symbol in range(19))
    bits.integer(final, 1)
    bits.integer(2, 2)
    bits.integer(0, 5)
    bits.integer(0, 5)
    bits.integer(14, 4)
    for symbol in ORDER[:18]:
        bits.integer(code_lengths[symbol], 3)
    literal_lengths = tuple(1 if symbol == 256 or value and symbol == 65 else 0 for symbol in range(257))
    if value or style == "literal":
        for length in literal_lengths:
            bits.symbol(length, code_lengths)
    else:
        repetitions = ((18, 127, 7), (18, 107, 7)) if style == "repeat18" else ((0, 0, 0), (16, 3, 2), (17, 7, 3), (18, 127, 7), (18, 90, 7))
        for symbol, extra, width in repetitions:
            bits.symbol(symbol, code_lengths)
            bits.integer(extra, width)
        bits.symbol(1, code_lengths)
    bits.symbol(distance_length, code_lengths)
    for byte in value:
        bits.symbol(byte, literal_lengths)
    bits.symbol(256, literal_lengths)


def dynamic_cross_boundary_repeat(bits: Bits, final: bool) -> None:
    code_lengths = tuple(2 if symbol in (0, 2, 16, 18) else 0 for symbol in range(19))
    bits.integer(final, 1)
    bits.integer(2, 2)
    bits.integer(0, 5)
    bits.integer(3, 5)
    bits.integer(12, 4)
    for symbol in ORDER[:16]:
        bits.integer(code_lengths[symbol], 3)
    for symbol, extra, width in ((18, 54, 7), (2, 0, 0), (2, 0, 0), (2, 0, 0), (18, 127, 7), (18, 39, 7), (2, 0, 0), (16, 1, 2)):
        bits.symbol(symbol, code_lengths)
        bits.integer(extra, width)
    literal_lengths = tuple(2 if symbol in (65, 66, 67, 256) else 0 for symbol in range(257))
    bits.symbol(256, literal_lengths)


def crafted_empty(style: str, distance_length: int = 0) -> bytes:
    bits = Bits()
    if style == "stored":
        stored(bits, True)
    elif style == "fixed":
        fixed(bits, True)
    elif style == "cross-boundary":
        dynamic_cross_boundary_repeat(bits, True)
    elif style == "mixed":
        fixed(bits, False)
        dynamic(bits, False, "all-repeats", distance_length)
        stored(bits, False)
        fixed(bits, False)
        dynamic_cross_boundary_repeat(bits, True)
    else:
        dynamic(bits, True, style, distance_length)
    return wrap(bits.bytes())


@pytest.mark.parametrize("style", ("stored", "fixed", "literal", "repeat18", "all-repeats", "cross-boundary", "mixed"))
@pytest.mark.parametrize("distance_length", (0, 1))
@pytest.mark.parametrize("factory", (bytes, memoryview))
def test_accepts_independently_encoded_empty_streams(style: str, distance_length: int, factory) -> None:
    encoded = crafted_empty(style, distance_length)
    assert zlib.decompress(encoded) == b""
    assert check_empty_stream(factory(encoded)) is None


@pytest.mark.parametrize("window", range(8, 16))
def test_accepts_every_declared_zlib_window(window: int) -> None:
    bits = Bits()
    fixed(bits, True)
    encoded = wrap(bits.bytes(), window=window)
    assert zlib.decompress(encoded) == b""
    assert check_empty_stream(encoded) is None


@pytest.mark.parametrize("level", range(-1, 10))
def test_accepts_native_empty_streams_at_every_compression_level(level: int) -> None:
    assert check_empty_stream(zlib.compress(b"", level)) is None


@pytest.mark.parametrize("window", range(9, 16))
@pytest.mark.parametrize("strategy", (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED, zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE, zlib.Z_FIXED))
def test_accepts_native_empty_streams_with_windows_and_strategies(window: int, strategy: int) -> None:
    compressor = zlib.compressobj(wbits=window, strategy=strategy)
    encoded = compressor.compress(b"") + compressor.flush()
    assert zlib.decompress(encoded) == b""
    assert check_empty_stream(encoded) is None


def test_accepts_native_empty_flush_blocks() -> None:
    compressor = zlib.compressobj()
    encoded = compressor.compress(b"") + compressor.flush(zlib.Z_SYNC_FLUSH) + compressor.flush(zlib.Z_FULL_FLUSH) + compressor.flush()
    assert zlib.decompress(encoded) == b""
    assert check_empty_stream(encoded) is None


@pytest.mark.parametrize("kind", ("stored", "fixed", "dynamic"))
@pytest.mark.parametrize("prefix", (False, True))
def test_rejects_nonempty_block_of_every_type(kind: str, prefix: bool) -> None:
    bits = Bits()
    if prefix:
        stored(bits, False)
        fixed(bits, False)
        dynamic(bits, False, "all-repeats")
    if kind == "stored":
        stored(bits, True, b"A")
    elif kind == "fixed":
        fixed(bits, True, b"A")
    else:
        dynamic(bits, True, value=b"A")
    encoded = wrap(bits.bytes(), b"A")
    assert zlib.decompress(encoded) == b"A"
    with pytest.raises(DSXError):
        check_empty_stream(encoded)


@pytest.mark.parametrize("level", range(-1, 10))
def test_rejects_native_nonempty_streams(level: int) -> None:
    with pytest.raises(DSXError):
        check_empty_stream(zlib.compress(b"A", level))


def test_rejects_reserved_block_type() -> None:
    bits = Bits()
    bits.integer(1, 1)
    bits.integer(3, 2)
    with pytest.raises(DSXError):
        check_empty_stream(wrap(bits.bytes()))


@pytest.mark.parametrize("style", ("stored", "fixed", "literal", "repeat18", "all-repeats", "cross-boundary"))
def test_rejects_truncated_deflate_payloads(style: str) -> None:
    encoded = crafted_empty(style)
    truncated = encoded[:2] + encoded[2:-5] + encoded[-4:]
    with pytest.raises(DSXError):
        check_empty_stream(truncated)
