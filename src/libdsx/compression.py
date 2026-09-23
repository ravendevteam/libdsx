from __future__ import annotations

import zlib
from typing import BinaryIO, Iterator

from .errors import FormatError


_LENGTH_ORDER = (16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15)
_LENGTH_BASE = (3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258)
_LENGTH_EXTRA = (0,) * 8 + (1,) * 4 + (2,) * 4 + (3,) * 4 + (4,) * 4 + (5,) * 4 + (0,)


class _Bits:
    def __init__(self, data: bytes | memoryview):
        self.data = memoryview(data)
        self.position = 16
        self.limit = (len(data) - 4) * 8

    def read(self, width: int) -> int:
        if self.position + width > self.limit:
            raise FormatError("truncated empty DEFLATE stream")
        result = 0
        shift = 0
        while width:
            offset = self.position % 8
            count = min(width, 8 - offset)
            result |= ((self.data[self.position // 8] >> offset) & ((1 << count) - 1)) << shift
            self.position += count
            shift += count
            width -= count
        return result

    def align(self) -> None:
        self.read(-self.position % 8)

    def skip_bytes(self, length: int) -> None:
        if self.position + length * 8 > self.limit:
            raise FormatError("truncated DEFLATE stream")
        self.position += length * 8


class _StreamBits:
    def __init__(self, stream: BinaryIO, length: int, chunk_size: int):
        self.stream = stream
        self.position = 0
        self.limit = (length - 4) * 8
        self.remaining = length - 4
        self.chunk_size = chunk_size
        self.data = b""
        self.index = 0
        self.value = 0
        self.width = 0

    def _refill(self) -> None:
        if not self.remaining:
            raise FormatError("truncated DEFLATE stream")
        self.data = _read_chunk(self.stream, min(self.chunk_size, self.remaining))
        self.remaining -= len(self.data)
        self.index = 0

    def read(self, width: int) -> int:
        if self.position + width > self.limit:
            raise FormatError("truncated DEFLATE stream")
        while self.width < width:
            if self.index == len(self.data):
                self._refill()
            self.value |= self.data[self.index] << self.width
            self.index += 1
            self.width += 8
        result = self.value & ((1 << width) - 1)
        self.value >>= width
        self.width -= width
        self.position += width
        return result

    def align(self) -> None:
        self.read(-self.position % 8)

    def skip_bytes(self, length: int) -> None:
        if self.position + length * 8 > self.limit:
            raise FormatError("truncated DEFLATE stream")
        self.position += length * 8
        while length:
            if self.index == len(self.data):
                self._refill()
            count = min(length, len(self.data) - self.index)
            self.index += count
            length -= count


def _table(lengths: list[int]) -> list[dict[int, int]]:
    maximum = max(lengths, default=0)
    counts = [0] * (maximum + 1)
    for length in lengths:
        if length:
            counts[length] += 1
    tables = [{} for _ in counts]
    next_codes = [0] * len(counts)
    code = 0
    for width in range(1, len(counts)):
        code = (code + counts[width - 1]) << 1
        if code + counts[width] > 1 << width:
            raise FormatError("oversubscribed DEFLATE Huffman tree")
        next_codes[width] = code
    for symbol, width in enumerate(lengths):
        if width:
            tables[width][next_codes[width]] = symbol
            next_codes[width] += 1
    return tables


def _symbol(bits: _Bits | _StreamBits, table: list[dict[int, int]]) -> int:
    code = 0
    for width in range(1, len(table)):
        code = (code << 1) | bits.read(1)
        symbol = table[width].get(code)
        if symbol is not None:
            return symbol
    raise FormatError("invalid DEFLATE Huffman symbol")


def _dynamic_tables(bits: _Bits | _StreamBits) -> tuple[list[dict[int, int]], list[dict[int, int]]]:
    literal_count = bits.read(5) + 257
    distance_count = bits.read(5) + 1
    code_count = bits.read(4) + 4
    if literal_count > 286:
        raise FormatError("invalid DEFLATE literal code count")
    code_lengths = [0] * 19
    for symbol in _LENGTH_ORDER[:code_count]:
        code_lengths[symbol] = bits.read(3)
    code_table = _table(code_lengths)
    lengths = []
    total = literal_count + distance_count
    while len(lengths) < total:
        symbol = _symbol(bits, code_table)
        if symbol < 16:
            lengths.append(symbol)
            continue
        if symbol == 16:
            if not lengths:
                raise FormatError("DEFLATE repeat has no previous code length")
            length = lengths[-1]
            repeat = bits.read(2) + 3
        elif symbol == 17:
            length = 0
            repeat = bits.read(3) + 3
        else:
            length = 0
            repeat = bits.read(7) + 11
        if len(lengths) + repeat > total:
            raise FormatError("DEFLATE code length repeat exceeds its alphabet")
        lengths.extend([length] * repeat)
    if not lengths[256]:
        raise FormatError("DEFLATE block lacks an end-of-block code")
    return _table(lengths[:literal_count]), _table(lengths[literal_count:])


_FIXED_LITERAL = _table([8] * 144 + [9] * 112 + [7] * 24 + [8] * 8)
_FIXED_DISTANCE = _table([5] * 32)


def _check_header(method: int, flags: int) -> None:
    if method & 15 != 8 or method >> 4 > 7 or (method * 256 + flags) % 31:
        raise FormatError("invalid zlib header")
    if flags & 32:
        raise FormatError("preset zlib dictionary is forbidden")


def _scan_length(bits: _Bits | _StreamBits, declared_length: int) -> None:
    decoded = 0
    while True:
        final = bits.read(1)
        kind = bits.read(2)
        if kind == 0:
            bits.align()
            length = bits.read(16)
            complement = bits.read(16)
            if length ^ complement != 65535:
                raise FormatError("invalid stored DEFLATE block length")
            decoded += length
            if decoded > declared_length:
                raise FormatError("decoded length exceeds its declaration")
            bits.skip_bytes(length)
        elif kind in (1, 2):
            literals, distances = (_FIXED_LITERAL, _FIXED_DISTANCE) if kind == 1 else _dynamic_tables(bits)
            while True:
                symbol = _symbol(bits, literals)
                if symbol == 256:
                    break
                if symbol < 256:
                    decoded += 1
                elif symbol <= 285:
                    code = symbol - 257
                    decoded += _LENGTH_BASE[code] + bits.read(_LENGTH_EXTRA[code])
                    distance = _symbol(bits, distances)
                    if distance > 29:
                        raise FormatError("invalid DEFLATE distance symbol")
                    if distance > 3:
                        bits.read(distance // 2 - 1)
                else:
                    raise FormatError("invalid DEFLATE length symbol")
                if decoded > declared_length:
                    raise FormatError("decoded length exceeds its declaration")
        else:
            raise FormatError("reserved DEFLATE block type")
        if final:
            break
    if decoded != declared_length:
        raise FormatError("uncompressed length mismatch")
    if (bits.position + 7) // 8 * 8 != bits.limit:
        raise FormatError("zlib stream does not end at its section boundary")


def check_empty_stream(data: bytes | memoryview) -> None:
    if len(data) < 6:
        raise FormatError("truncated zlib stream")
    _check_header(data[0], data[1])
    _scan_length(_Bits(data), 0)


def _read_chunk(stream: BinaryIO, length: int) -> bytes:
    chunk = stream.read(length)
    if not isinstance(chunk, bytes):
        raise TypeError("expected a binary stream returning bytes")
    if not chunk or len(chunk) > length:
        raise FormatError("unexpected end of stream or invalid read length")
    return chunk


def _prove_length(stream: BinaryIO, start: int, stored_length: int, decoded_length: int, chunk_size: int) -> None:
    position = stream.tell()
    try:
        stream.seek(start)
        if stored_length < 6:
            raise FormatError("truncated zlib stream")
        bits = _StreamBits(stream, stored_length, chunk_size)
        _check_header(bits.read(8), bits.read(8))
        _scan_length(bits, decoded_length)
    finally:
        stream.seek(position)


def iter_inflate(
    stream: BinaryIO,
    stored_length: int,
    decoded_length: int,
    context: str,
    *,
    chunk_size: int = 65536,
) -> Iterator[bytes]:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(stored_length, bool) or not isinstance(stored_length, int) or stored_length < 1:
        raise FormatError(f"{context}: stored length must be a positive integer")
    if isinstance(decoded_length, bool) or not isinstance(decoded_length, int) or decoded_length < 0:
        raise FormatError(f"{context}: decoded length must be a nonnegative integer")
    start = stream.tell()
    remaining = stored_length
    produced = 0
    proven = False
    pending = b""
    try:
        if decoded_length == 0:
            _prove_length(stream, start, stored_length, decoded_length, chunk_size)
            proven = True
        decoder = zlib.decompressobj()
        while not decoder.eof:
            if produced == decoded_length and not proven:
                _prove_length(stream, start, stored_length, decoded_length, chunk_size)
                proven = True
            if not pending and remaining:
                pending = _read_chunk(stream, min(chunk_size, remaining))
                remaining -= len(pending)
            cap = min(chunk_size, decoded_length - produced) if produced < decoded_length else 1
            decoded = decoder.decompress(pending, cap)
            pending = decoder.unconsumed_tail
            produced += len(decoded)
            if produced > decoded_length:
                raise FormatError("decoded length exceeds its declaration")
            if decoded:
                yield decoded
            if not decoder.eof and not decoded and not pending and not remaining:
                raise FormatError("incomplete zlib stream")
        if produced != decoded_length:
            raise FormatError("uncompressed length mismatch")
        if remaining or decoder.unused_data or pending:
            raise FormatError("zlib stream does not end at its section boundary")
    except zlib.error as error:
        raise FormatError(f"{context}: invalid zlib stream: {error}") from error
    except FormatError as error:
        raise FormatError(f"{context}: {error}") from error
