from __future__ import annotations

import io
import struct
import tracemalloc
from dataclasses import fields
from datetime import date
from pathlib import Path

import pytest

import libdsx as dsx
from libdsx import codec, streaming
from test_codec import container, tlv, u


SAMPLE = Path(__file__).parent / "data" / "DossierRev2.dsx"


def document(*records) -> dsx.Document:
    return dsx.Document(dsx.Metadata("TITLE", ("Author",), 1, date(2026, 1, 1)), records)


@pytest.mark.parametrize("chunk_size", (1, 2, 7, 127, 65536))
def test_streamed_records_match_sample_at_chunk_boundaries(monkeypatch, chunk_size):
    expected = dsx.load(SAMPLE)
    monkeypatch.setattr(streaming, "CHUNK_SIZE", chunk_size)
    with dsx.open_document(SAMPLE) as reader:
        assert reader.metadata == expected.metadata
        assert not reader.content_verified
        assert tuple(reader) == expected.records
        assert reader.content_verified
    assert reader.closed
    assert reader.content_verified


def test_stopping_early_is_not_validation_and_does_not_close_caller_stream():
    encoded = dsx.dumps(document(dsx.AsciiBlock("one"), dsx.AsciiBlock("two")))
    stream = io.BytesIO(encoded)
    with dsx.open_document(stream) as reader:
        assert next(reader) == dsx.AsciiBlock("one")
        assert not reader.content_verified
    assert reader.closed
    assert not reader.content_verified
    assert not stream.closed
    with pytest.raises(ValueError, match="closed"):
        next(reader)


def test_finish_validates_remaining_content_without_materializing(monkeypatch):
    data = dsx.dumps(document(dsx.AsciiBlock("one"), dsx.AsciiBlock("two")))
    with dsx.open_document(io.BytesIO(data)) as reader:
        assert next(reader) == dsx.AsciiBlock("one")
        monkeypatch.setattr(streaming, "AsciiBlock", lambda *args: pytest.fail("remaining records materialized"))
        reader.finish()
        assert reader.content_verified
        assert list(reader) == []
        reader.finish()


def test_validation_only_avoids_model_and_large_text_allocations(monkeypatch):
    source = container(content_bytes=tlv(4, (b"A" * 97 + b"\n") * 30000))
    for name in ("AsciiBlock", "Paragraph", "BulletItem", "Text", "Citation", "Heading", "Reference"):
        monkeypatch.setattr(streaming, name, lambda *args: pytest.fail("model object allocated"))
    stream = io.BytesIO(source)
    tracemalloc.start()
    try:
        dsx.validate_file(stream)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 700000


def test_validation_only_large_paragraph_uses_bounded_chunks(monkeypatch):
    text = b"word " * 400000 + b"end"
    encoded = container(content_bytes=tlv(2, tlv(1, text)))
    del text
    stream = io.BytesIO(encoded)
    monkeypatch.setattr(codec, "load", lambda *args, **kwargs: pytest.fail("eager load called"))
    tracemalloc.start()
    try:
        dsx.validate_file(stream)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 700000


def test_tiny_record_validation_does_not_retain_records():
    data = container(content_bytes=b"\x04\x00" * 100000)
    stream = io.BytesIO(data)
    tracemalloc.start()
    try:
        dsx.validate_file(stream)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 500000


def test_materializing_tiny_text_chunks_does_not_amplify_memory(monkeypatch):
    expected = document(dsx.AsciiBlock(("A" * 97 + "\n") * 3000))
    encoded = dsx.dumps(expected)
    monkeypatch.setattr(streaming, "CHUNK_SIZE", 2)
    tracemalloc.start()
    try:
        actual = dsx.loads(encoded)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert actual == expected
    assert peak < 1000000


def test_materializing_large_text_avoids_full_section_copy():
    expected = document(dsx.AsciiBlock(("A" * 97 + "\n") * 30000))
    encoded = dsx.dumps(expected)
    tracemalloc.start()
    try:
        actual = dsx.loads(encoded)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert actual == expected
    assert peak < len(expected.records[0].value) * 2 + 700000


@pytest.mark.parametrize("tail", (b"\x06\x00", tlv(2, tlv(2, u(1)))))
def test_late_invalid_records_fail_finish_and_never_mark_verified(tail):
    encoded = container(content_bytes=tlv(4, b"first") + tail)
    stream = io.BytesIO(encoded)
    reader = dsx.open_document(stream)
    assert next(reader) == dsx.AsciiBlock("first")
    with pytest.raises(dsx.DSXError):
        reader.finish()
    assert reader.closed
    assert not reader.content_verified
    assert not stream.closed


def test_late_checksum_failure_is_reported(monkeypatch):
    data = bytearray(dsx.dumps(document(*(dsx.AsciiBlock(str(i)) for i in range(500))), compression_level=0))
    data[-1] ^= 1
    monkeypatch.setattr(streaming, "CHUNK_SIZE", 64)
    with dsx.open_document(io.BytesIO(data)) as reader:
        assert next(reader) == dsx.AsciiBlock("0")
        with pytest.raises(dsx.FormatError):
            list(reader)
        assert not reader.content_verified
        assert reader.closed


def test_record_iterator_closes_owned_file_on_generator_close(monkeypatch):
    stream = io.BytesIO(dsx.dumps(document(dsx.AsciiBlock("one"), dsx.AsciiBlock("two"))))
    monkeypatch.setattr(streaming, "open", lambda *args: stream, raising=False)
    records = dsx.iter_records("example.dsx")
    assert next(records) == dsx.AsciiBlock("one")
    records.close()
    assert stream.closed


def test_reader_rejects_mixing_record_and_event_consumption():
    with dsx.open_document(SAMPLE) as reader:
        next(reader)
        with pytest.raises(ValueError, match="consumed"):
            next(reader._iter_events())


@pytest.mark.parametrize("field", [field.name for field in fields(dsx.Limits)])
@pytest.mark.parametrize("value", (-1, True, 1.0, "1"))
def test_invalid_resource_budgets(field, value):
    with pytest.raises(ValueError, match=field):
        dsx.Limits(**{field: value})


@pytest.mark.parametrize("function", (dsx.load, dsx.validate_file, dsx.read_metadata, dsx.open_document))
def test_header_budgets_fail_before_content_reads(function):
    encoded = dsx.dumps(document(dsx.AsciiBlock("body")))

    class HeaderOnly(io.BytesIO):
        def read(self, size=-1):
            assert self.tell() + size <= 32
            return super().read(size)

    with pytest.raises(dsx.ResourceLimitError, match="max_content_decoded"):
        function(HeaderOnly(encoded), limits=dsx.Limits(max_content_decoded=0))


@pytest.mark.parametrize("field,offset", (("max_metadata_stored", 12), ("max_metadata_decoded", 16), ("max_content_stored", 20), ("max_content_decoded", 24)))
def test_section_budgets_are_inclusive(field, offset):
    expected = document(dsx.AsciiBlock("body"))
    data = dsx.dumps(expected)
    length = struct.unpack_from("<I", data, offset)[0]
    assert dsx.loads(data, limits=dsx.Limits(**{field: length})) == expected
    with pytest.raises(dsx.ResourceLimitError, match=field):
        dsx.loads(data, limits=dsx.Limits(**{field: length - 1}))


@pytest.mark.parametrize("factory", (bytes, bytearray, memoryview))
def test_streamed_bytes_entrypoint_preserves_buffer_support(factory):
    expected = document(dsx.AsciiBlock("body"))
    assert dsx.loads(factory(dsx.dumps(expected))) == expected


@pytest.mark.parametrize("operation", ("load", "validate", "dump"))
def test_record_and_inline_budgets(operation):
    expected = document(dsx.Paragraph((dsx.Text("hello"), dsx.Citation(1))), dsx.Reference(1, "urn:example"))
    data = dsx.dumps(expected)
    for limits, field in ((dsx.Limits(max_records=1), "max_records"), (dsx.Limits(max_inlines=1), "max_inlines"), (dsx.Limits(max_pending_citations=0), "max_pending_citations"), (dsx.Limits(max_record_bytes=1), "max_record_bytes"), (dsx.Limits(max_inline_bytes=4), "max_inline_bytes")):
        with pytest.raises(dsx.ResourceLimitError, match=field):
            if operation == "load":
                dsx.loads(data, limits=limits)
            elif operation == "validate":
                dsx.validate(expected, limits=limits)
            else:
                dsx.dumps(expected, limits=limits)


def test_record_budget_stops_before_model_allocation(monkeypatch):
    data = container(content_bytes=b"\x04\x00" * 1000)
    monkeypatch.setattr(streaming, "AsciiBlock", lambda *args: pytest.fail("record allocated despite zero budget"))
    with pytest.raises(dsx.ResourceLimitError, match="max_records"):
        dsx.loads(data, limits=dsx.Limits(max_records=0))


def test_large_payload_budget_stops_before_decoding_text(monkeypatch):
    data = container(content_bytes=tlv(2, tlv(1, b"word " * 100000 + b"end")))
    monkeypatch.setattr(streaming._DecodedReader, "text", lambda *args: pytest.fail("text decoded before budget check"))
    with pytest.raises(dsx.ResourceLimitError, match="max_inline_bytes"):
        dsx.loads(data, limits=dsx.Limits(max_inline_bytes=100))


def test_repeated_citations_consume_one_pending_budget():
    expected = document(dsx.Paragraph((dsx.Citation(1), dsx.Citation(1))), dsx.Reference(1, "urn:example"))
    assert dsx.loads(dsx.dumps(expected), limits=dsx.Limits(max_pending_citations=1)) == expected


@pytest.mark.parametrize("function", (dsx.load, dsx.validate_file, dsx.open_document))
def test_invalid_limits_type(function):
    with pytest.raises(TypeError, match="Limits"):
        function(io.BytesIO(dsx.dumps(document())), limits={})


@pytest.mark.parametrize("payload", (b"\x04", tlv(2, b"\x01"), tlv(2, tlv(2, b"\x01\x00")), tlv(1, b"\x80"), tlv(5, b"\x80")))
def test_record_boundaries_fail_even_with_single_byte_chunks(monkeypatch, payload):
    monkeypatch.setattr(streaming, "CHUNK_SIZE", 1)
    with pytest.raises(dsx.FormatError):
        dsx.validate_file(io.BytesIO(container(content_bytes=payload)))
