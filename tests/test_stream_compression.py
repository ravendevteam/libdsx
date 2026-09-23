from __future__ import annotations

import io
import random
import zlib

import pytest

from libdsx import FormatError
from libdsx.compression import iter_inflate


class TrackedStream(io.BytesIO):
    def __init__(self, data: bytes, partial: int | None = None):
        super().__init__(data)
        self.requests = []
        self.partial = partial

    def read(self, size: int = -1) -> bytes:
        self.requests.append(size)
        return super().read(size if self.partial is None else min(size, self.partial))


def padded_stream(payload: bytes, empty_blocks: int) -> bytes:
    size = len(payload)
    assert size <= 65535
    first = b"\x00" + size.to_bytes(2, "little") + (size ^ 65535).to_bytes(2, "little") + payload
    empty = b"\x00\x00\x00\xff\xff"
    final = b"\x01\x00\x00\xff\xff"
    return b"\x78\x01" + first + empty * empty_blocks + final + zlib.adler32(payload).to_bytes(4, "big")


@pytest.mark.parametrize("chunk_size", (1, 7, 257, 65536))
@pytest.mark.parametrize("level", (0, 1, 6, 9))
@pytest.mark.parametrize("size", (0, 1, 99, 32768, 150001))
def test_native_stream_roundtrip_is_bounded(chunk_size: int, level: int, size: int) -> None:
    payload = (b"0123456789ABCDE\n" * (size // 16 + 1))[:size]
    encoded = zlib.compress(payload, level)
    stream = TrackedStream(b"PREFIX" + encoded + b"NEXT")
    stream.seek(6)
    chunks = list(iter_inflate(stream, len(encoded), len(payload), "content", chunk_size=chunk_size))
    assert b"".join(chunks) == payload
    assert all(0 < len(chunk) <= chunk_size for chunk in chunks)
    assert all(0 < request <= chunk_size for request in stream.requests)
    assert stream.tell() == 6 + len(encoded)
    assert stream.read(4) == b"NEXT"


@pytest.mark.parametrize("strategy", (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED, zlib.Z_HUFFMAN_ONLY, zlib.Z_RLE, zlib.Z_FIXED))
@pytest.mark.parametrize("level", (0, 1, 6, 9))
def test_structural_replay_understands_native_deflate_codes(strategy: int, level: int) -> None:
    randomizer = random.Random(90210)
    payload = bytes(randomizer.randrange(32, 127) for _ in range(3000)) + b"ABCD" * 3000
    compressor = zlib.compressobj(level, strategy=strategy)
    prefix = compressor.compress(payload) + compressor.flush(zlib.Z_SYNC_FLUSH)
    encoded = prefix + b"\x00\x00\x00\xff\xff" * 20 + compressor.flush()
    assert zlib.decompress(encoded) == payload
    stream = TrackedStream(encoded, partial=3)
    assert b"".join(iter_inflate(stream, len(encoded), len(payload), "content", chunk_size=31)) == payload


@pytest.mark.parametrize("payload", (b"", b"A", b"abc" * 500))
def test_accepts_large_trailing_empty_blocks(payload: bytes) -> None:
    encoded = padded_stream(payload, 30000)
    stream = TrackedStream(encoded)
    assert zlib.decompress(encoded) == payload
    assert b"".join(iter_inflate(stream, len(encoded), len(payload), "content")) == payload
    assert max(stream.requests) <= 65536


@pytest.mark.parametrize("level", (0, 1, 9))
def test_zero_declared_size_rejects_before_native_inflater(monkeypatch, level: int) -> None:
    encoded = zlib.compress(b"A" * 100000, level)

    def forbidden():
        raise AssertionError("native decoder must not run before zero-output proof")

    monkeypatch.setattr(zlib, "decompressobj", forbidden)
    with pytest.raises(FormatError, match="content: decoded length exceeds"):
        list(iter_inflate(io.BytesIO(encoded), len(encoded), 0, "content", chunk_size=17))


@pytest.mark.parametrize("actual_size", (99, 100, 100000))
def test_native_decoder_never_exceeds_declared_budget(monkeypatch, actual_size: int) -> None:
    declared = 99
    encoded = zlib.compress(b"A" * actual_size)
    native_factory = zlib.decompressobj
    produced = 0
    caps = []

    class ObservedInflater:
        def __init__(self):
            self.native = native_factory()

        def decompress(self, data, max_length):
            nonlocal produced
            assert 0 < max_length <= declared - produced
            caps.append(max_length)
            chunk = self.native.decompress(data, max_length)
            produced += len(chunk)
            return chunk

        def __getattr__(self, name):
            return getattr(self.native, name)

    monkeypatch.setattr(zlib, "decompressobj", ObservedInflater)
    chunks = iter_inflate(io.BytesIO(encoded), len(encoded), declared, "content", chunk_size=31)
    if actual_size == declared:
        assert b"".join(chunks) == b"A" * declared
    else:
        with pytest.raises(FormatError, match="decoded length exceeds"):
            list(chunks)
    assert produced == declared
    assert caps == [31, 31, 31, 6]


@pytest.mark.parametrize("declared", (0, 1, 9))
def test_proven_trailing_blocks_do_not_emit_probe_bytes(monkeypatch, declared: int) -> None:
    encoded = padded_stream(b"A" * declared, 100)
    native_factory = zlib.decompressobj
    produced = 0

    class ObservedInflater:
        def __init__(self):
            self.native = native_factory()

        def decompress(self, data, max_length):
            nonlocal produced
            assert 0 < max_length <= max(1, declared - produced)
            chunk = self.native.decompress(data, max_length)
            produced += len(chunk)
            assert produced <= declared
            return chunk

        def __getattr__(self, name):
            return getattr(self.native, name)

    monkeypatch.setattr(zlib, "decompressobj", ObservedInflater)
    assert b"".join(iter_inflate(io.BytesIO(encoded), len(encoded), declared, "content", chunk_size=17)) == b"A" * declared


@pytest.mark.parametrize("suffix", (b"X", zlib.compress(b""), zlib.compress(b"other")))
@pytest.mark.parametrize("payload", (b"", b"document"))
def test_rejects_trailing_bytes_and_concatenated_streams(suffix: bytes, payload: bytes) -> None:
    encoded = zlib.compress(payload) + suffix
    with pytest.raises(FormatError):
        list(iter_inflate(io.BytesIO(encoded), len(encoded), len(payload), "content", chunk_size=3))


@pytest.mark.parametrize("payload", (b"", b"A"))
@pytest.mark.parametrize("cut", range(1, 12))
def test_rejects_truncated_stream_and_checksum(payload: bytes, cut: int) -> None:
    encoded = padded_stream(payload, 10)[:-cut]
    with pytest.raises(FormatError):
        list(iter_inflate(io.BytesIO(encoded), len(encoded), len(payload), "content", chunk_size=3))


@pytest.mark.parametrize("payload", (b"", b"A"))
def test_rejects_wrong_checksum_after_structural_replay(payload: bytes) -> None:
    encoded = padded_stream(payload, 100)
    encoded = encoded[:-1] + bytes((encoded[-1] ^ 255,))
    with pytest.raises(FormatError, match="content: invalid zlib stream"):
        list(iter_inflate(io.BytesIO(encoded), len(encoded), len(payload), "content", chunk_size=13))


def test_rejects_short_binary_source() -> None:
    encoded = zlib.compress(b"a document")
    with pytest.raises(FormatError, match="unexpected end of stream"):
        list(iter_inflate(io.BytesIO(encoded[:-2]), len(encoded), 10, "content", chunk_size=3))


@pytest.mark.parametrize("size", (False, 0, -1, 1.5, "1"))
def test_chunk_size_must_be_positive_integer(size) -> None:
    encoded = zlib.compress(b"")
    with pytest.raises(ValueError, match="chunk_size"):
        list(iter_inflate(io.BytesIO(encoded), len(encoded), 0, "content", chunk_size=size))


@pytest.mark.parametrize("stored,decoded", ((-1, 0), (0, 0), (True, 0), (8, -1), (8, False), (8, 0.5)))
def test_invalid_lengths_rejected_without_reading(stored, decoded) -> None:
    stream = TrackedStream(zlib.compress(b""))
    with pytest.raises(FormatError, match="content: .* length"):
        list(iter_inflate(stream, stored, decoded, "content"))
    assert stream.requests == []
