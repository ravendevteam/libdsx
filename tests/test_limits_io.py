from __future__ import annotations

import io
import struct
import zlib
from dataclasses import replace
from datetime import date

import pytest

import libdsx
from libdsx import codec


def document() -> libdsx.Document:
    return libdsx.Document(libdsx.Metadata("A", ("A",), 1, date(1, 1, 1)))


def test_metadata_decoded_size_limit_is_inclusive() -> None:
    base = document()
    candidate = replace(base, metadata=replace(base.metadata, additional={"a": "x" * 65502}))
    encoded = libdsx.dumps(candidate)
    assert struct.unpack_from("<I", encoded, 16)[0] == 65536
    assert libdsx.loads(encoded) == candidate
    too_large = replace(base, metadata=replace(base.metadata, additional={"a": "x" * 65503}))
    with pytest.raises(libdsx.ValidationError, match="metadata.*65536"):
        libdsx.validate(too_large)
    with pytest.raises(libdsx.ValidationError, match="metadata.*65536"):
        libdsx.dumps(too_large)


def test_content_size_includes_tlv_headers(monkeypatch) -> None:
    monkeypatch.setattr(codec, "MAX_CONTENT_DECODED", 100)
    base = document()
    exact = replace(base, records=(libdsx.AsciiBlock("A" * 98),))
    libdsx.validate(exact)
    assert libdsx.loads(libdsx.dumps(exact)) == exact
    oversized = replace(base, records=exact.records + (libdsx.AsciiBlock(""),))
    with pytest.raises(libdsx.ValidationError, match="content.*100"):
        libdsx.validate(oversized)


def test_content_size_counts_merged_text_and_multibyte_lengths(monkeypatch) -> None:
    base = document()
    record = libdsx.Paragraph(tuple(libdsx.Text("A ") for _ in range(70)) + (libdsx.Text("A"),))
    candidate = replace(base, records=(record,))
    monkeypatch.setattr(codec, "MAX_CONTENT_DECODED", 147)
    encoded = libdsx.dumps(candidate)
    assert struct.unpack_from("<I", encoded, 24)[0] == 147
    monkeypatch.setattr(codec, "MAX_CONTENT_DECODED", 146)
    with pytest.raises(libdsx.ValidationError, match="content.*146"):
        libdsx.validate(candidate)


def test_format_section_limits_match_the_specification() -> None:
    assert libdsx.MAX_METADATA_STORED == 131072
    assert libdsx.MAX_METADATA_DECODED == 65536
    assert libdsx.MAX_CONTENT_STORED == 68157440
    assert libdsx.MAX_CONTENT_DECODED == 67108864


@pytest.mark.parametrize("year", (1, 99, 999, 9999))
def test_date_encoding_pads_year_to_four_digits(year: int) -> None:
    base = document()
    candidate = replace(base, metadata=replace(base.metadata, date=date(year, 1, 1)))
    encoded = libdsx.dumps(candidate)
    stored_length = struct.unpack_from("<I", encoded, 12)[0]
    plain = zlib.decompress(encoded[32:32 + stored_length])
    assert plain.endswith(f"01.01.{year:04}".encode("ascii"))
    assert libdsx.loads(encoded) == candidate


def test_partial_reads_and_writes() -> None:
    class PartialStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(min(size, 3))

        def write(self, data) -> int:
            return super().write(data[:3])

    stream = PartialStream()
    libdsx.dump(document(), stream)
    stream.seek(0)
    assert libdsx.load(stream) == document()
    assert not stream.closed


@pytest.mark.parametrize("written", (None, 0, -1, 100000))
def test_failed_stream_writes_raise(written) -> None:
    class FailedStream(io.BytesIO):
        def write(self, data):
            return written

    with pytest.raises(OSError):
        libdsx.dump(document(), FailedStream())


def test_stream_operations_start_at_current_position() -> None:
    stream = io.BytesIO(b"PREFIX" + libdsx.dumps(document()))
    stream.seek(6)
    assert libdsx.load(stream) == document()
    stream.seek(6)
    assert libdsx.read_metadata(stream).metadata == document().metadata


def test_invalid_document_does_not_truncate_existing_file(tmp_path) -> None:
    target = tmp_path / "existing.dsx"
    target.write_bytes(b"Existing content")
    invalid = replace(document(), records=(libdsx.Paragraph((libdsx.Citation(1),)),))
    with pytest.raises(libdsx.ValidationError):
        libdsx.dump(invalid, target)
    assert target.read_bytes() == b"Existing content"


@pytest.mark.parametrize("value", (-2, 10, True, 1.5, "1", None))
def test_invalid_compression_level(value) -> None:
    with pytest.raises(ValueError, match="compression_level"):
        libdsx.dumps(document(), compression_level=value)


@pytest.mark.parametrize("value", ("DSX", None, [1, 2, 3]))
def test_loads_requires_binary_data(value) -> None:
    with pytest.raises(TypeError):
        libdsx.loads(value)


def test_loads_rejects_noncontiguous_memoryview() -> None:
    with pytest.raises(TypeError, match="contiguous"):
        libdsx.loads(memoryview(libdsx.dumps(document()))[::2])


def test_load_rejects_text_stream() -> None:
    with pytest.raises(TypeError, match="binary"):
        libdsx.load(io.StringIO("x" * 100))


@pytest.mark.parametrize("flush_mode", (zlib.Z_SYNC_FLUSH, zlib.Z_FULL_FLUSH, zlib.Z_BLOCK))
@pytest.mark.parametrize("size", (0, 1, 32768))
def test_valid_stream_with_multiple_deflate_blocks(flush_mode: int, size: int) -> None:
    payload = b"A" * size
    compressor = zlib.compressobj()
    stream = compressor.compress(payload) + compressor.flush(flush_mode)
    stream += compressor.flush(zlib.Z_FINISH)
    assert codec._inflate(stream, len(payload), "test section") == payload


@pytest.mark.parametrize("level", (0, 1, 9))
def test_empty_length_overflow_is_rejected_before_native_decompression(monkeypatch, level: int) -> None:
    stream = zlib.compress(b"X" * 262144, level)

    def forbidden_inflater():
        raise AssertionError("A section declared empty must not decompress nonempty data")

    monkeypatch.setattr(codec.zlib, "decompressobj", forbidden_inflater)
    with pytest.raises(libdsx.FormatError):
        codec._inflate(stream, 0, "content")


def test_nonempty_decompression_uses_the_exact_declared_cap(monkeypatch) -> None:
    payload = b"A" * 1024
    stream = zlib.compress(payload)
    native_factory = zlib.decompressobj
    observed_caps = []

    class ObservedInflater:
        def __init__(self):
            self.decoder = native_factory()

        def decompress(self, data, max_length):
            observed_caps.append(max_length)
            return self.decoder.decompress(data, max_length)

        def __getattr__(self, name):
            return getattr(self.decoder, name)

    monkeypatch.setattr(codec.zlib, "decompressobj", ObservedInflater)
    assert codec._inflate(stream, len(payload), "content") == payload
    assert observed_caps == [len(payload)]
