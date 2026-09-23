from __future__ import annotations

import io
import struct
import zlib
from datetime import date

import pytest

import libdsx
from libdsx import codec, writing
from libdsx.errors import ResourceLimitError
from libdsx.limits import Limits


def document(records: tuple = ()) -> libdsx.Document:
    return libdsx.Document(libdsx.Metadata("TEST", ("Ada",), 1, date(2026, 9, 23)), records)


def content_bytes(encoded: bytes) -> bytes:
    boundary = 32 + struct.unpack_from("<I", encoded, 12)[0]
    return zlib.decompress(encoded[boundary:])


@pytest.mark.parametrize("level", range(-1, 10))
def test_stream_writer_emits_canonical_sections(level: int) -> None:
    candidate = document((
        libdsx.Heading(1, "TITLE"),
        libdsx.Paragraph((libdsx.Text("Hello"), libdsx.Text(" world"), libdsx.Citation(1), libdsx.Text("."))),
        libdsx.BulletItem((libdsx.Text("Item"),)),
        libdsx.AsciiBlock("A\n\n"),
        libdsx.Reference(1, "https://example.com"),
    ))
    encoded = writing.dumps(candidate, compression_level=level, spool_limit=32)
    expected = (
        b"\x01\x06\x01TITLE"
        b"\x02\x13\x01\x0bHello world\x02\x01\x01\x01\x01."
        b"\x03\x06\x01\x04Item"
        b"\x04\x03A\n\n"
        b"\x05\x14\x01https://example.com"
    )
    assert content_bytes(encoded) == expected
    restored = libdsx.loads(encoded)
    assert restored.records[1].inlines == (libdsx.Text("Hello world"), libdsx.Citation(1), libdsx.Text("."))
    assert libdsx.render(restored) == libdsx.render(candidate)


def test_writer_rolls_compressed_spool_to_disk_and_closes_it(monkeypatch) -> None:
    factory = writing.SpooledTemporaryFile
    spools = []

    def observed_spool(*args, **kwargs):
        spool = factory(*args, **kwargs)
        spools.append(spool)
        return spool

    monkeypatch.setattr(writing, "SpooledTemporaryFile", observed_spool)
    candidate = document((libdsx.AsciiBlock("A\n" * 1000),))
    encoded = writing.dumps(candidate, compression_level=0, spool_limit=64)
    assert libdsx.loads(encoded) == candidate
    assert len(spools) == 1
    assert spools[0]._rolled
    assert spools[0].closed


def test_writer_keeps_small_documents_in_memory(monkeypatch) -> None:
    factory = writing.SpooledTemporaryFile
    spools = []

    def observed_spool(*args, **kwargs):
        spool = factory(*args, **kwargs)
        spools.append(spool)
        return spool

    monkeypatch.setattr(writing, "SpooledTemporaryFile", observed_spool)
    assert libdsx.loads(writing.dumps(document())) == document()
    assert not spools[0]._rolled
    assert spools[0].closed


def test_writer_supports_partial_writes_without_seeking_target() -> None:
    class PartialSink:
        def __init__(self):
            self.data = bytearray()
            self.calls = 0

        def write(self, value):
            self.calls += 1
            accepted = min(7, len(value))
            self.data.extend(value[:accepted])
            return accepted

    sink = PartialSink()
    writing.dump(document(), sink)
    assert libdsx.loads(sink.data) == document()
    assert sink.calls > 2


def test_compression_failure_does_not_truncate_destination(tmp_path, monkeypatch) -> None:
    target = tmp_path / "existing.dsx"
    target.write_bytes(b"previous document")
    factory = writing.zlib.compressobj
    calls = 0

    def failing_compressor(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("compression failed")
        return factory(*args, **kwargs)

    monkeypatch.setattr(writing.zlib, "compressobj", failing_compressor)
    with pytest.raises(OSError, match="compression failed"):
        writing.dump(document(), target)
    assert target.read_bytes() == b"previous document"


def test_failed_output_closes_spool(monkeypatch) -> None:
    factory = writing.SpooledTemporaryFile
    spools = []

    def observed_spool(*args, **kwargs):
        spool = factory(*args, **kwargs)
        spools.append(spool)
        return spool

    class FailedSink:
        def write(self, value):
            raise OSError("output failed")

    monkeypatch.setattr(writing, "SpooledTemporaryFile", observed_spool)
    with pytest.raises(OSError, match="output failed"):
        writing.dump(document(), FailedSink(), spool_limit=1)
    assert spools[0].closed


@pytest.mark.parametrize("limit", (0, -1, True, 1.5, "100", None))
def test_spool_limit_must_be_positive_before_target_is_written(limit) -> None:
    target = io.BytesIO(b"original")
    with pytest.raises(ValueError, match="spool_limit"):
        writing.dump(document(), target, spool_limit=limit)
    assert target.getvalue() == b"original"


def test_large_records_and_merged_text_are_compressed_in_bounded_chunks(monkeypatch) -> None:
    candidate = document((
        libdsx.Paragraph(tuple(libdsx.Text("word " * 20000) for _ in range(4)) + (libdsx.Text("end"),)),
        libdsx.AsciiBlock("A\n" * 200000),
    ))
    sizes = []
    factory = writing.zlib.compressobj

    class ObservedCompressor:
        def __init__(self, *args, **kwargs):
            self.compressor = factory(*args, **kwargs)

        def compress(self, data):
            sizes.append(len(data))
            return self.compressor.compress(data)

        def flush(self):
            return self.compressor.flush()

    def eager_encoder_forbidden(*args, **kwargs):
        raise AssertionError("writer constructed a complete content or inline buffer")

    monkeypatch.setattr(writing.zlib, "compressobj", ObservedCompressor)
    monkeypatch.setattr(codec, "_encode_content", eager_encoder_forbidden, raising=False)
    monkeypatch.setattr(codec, "_encode_inlines", eager_encoder_forbidden, raising=False)
    encoded = writing.dumps(candidate, spool_limit=256)
    assert max(sizes) <= 65536
    assert sizes.count(65536) >= 12
    restored = libdsx.loads(encoded)
    assert restored.records[0].inlines == (libdsx.Text("word " * 80000 + "end"),)
    assert restored.records[1] == candidate.records[1]


def test_writer_batches_many_small_records_before_compressing(monkeypatch) -> None:
    factory = writing.zlib.compressobj
    sizes = []

    class ObservedCompressor:
        def __init__(self, *args, **kwargs):
            self.compressor = factory(*args, **kwargs)

        def compress(self, data):
            sizes.append(len(data))
            return self.compressor.compress(data)

        def flush(self):
            return self.compressor.flush()

    monkeypatch.setattr(writing.zlib, "compressobj", ObservedCompressor)
    candidate = document((libdsx.AsciiBlock(""),) * 10000)
    encoded = writing.dumps(candidate)
    assert len(sizes) == 2
    assert sizes[-1] == 20000
    assert libdsx.loads(encoded) == candidate


@pytest.mark.parametrize("name,offset", (("max_metadata_stored", 12), ("max_content_stored", 20)))
def test_compressed_limits_are_checked_before_destination_writes(name: str, offset: int) -> None:
    candidate = document((libdsx.AsciiBlock("A\n" * 1000),))
    encoded = writing.dumps(candidate)
    exact = struct.unpack_from("<I", encoded, offset)[0]
    assert writing.dumps(candidate, limits=Limits(**{name: exact})) == encoded
    target = io.BytesIO(b"original")
    with pytest.raises(ResourceLimitError, match=name):
        writing.dump(candidate, target, limits=Limits(**{name: exact - 1}))
    assert target.getvalue() == b"original"


def test_inline_byte_limit_applies_to_the_canonical_merged_text() -> None:
    candidate = document((libdsx.Paragraph((libdsx.Text("abc"), libdsx.Text("def"))),))
    with pytest.raises(ResourceLimitError, match="max_inline_bytes"):
        writing.dumps(candidate, limits=Limits(max_inline_bytes=3))
    encoded = writing.dumps(candidate, limits=Limits(max_inline_bytes=6))
    assert content_bytes(encoded) == b"\x02\x08\x01\x06abcdef"
