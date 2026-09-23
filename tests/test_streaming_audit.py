from __future__ import annotations

import io
from datetime import date

import pytest

import libdsx as dsx
from libdsx import streaming


def sample_document() -> dsx.Document:
    return dsx.Document(
        dsx.Metadata("TITLE", ("Author",), 1, date(2026, 9, 23)),
        (dsx.AsciiBlock("first"), dsx.AsciiBlock("last")),
    )


@pytest.mark.parametrize("value", [None, "", bytearray(), memoryview(b"")])
def test_end_of_file_probe_requires_bytes(value: object) -> None:
    class InvalidEOF(io.BytesIO):
        def read(self, size: int = -1):
            data = super().read(size)
            return data if data else value

    stream = InvalidEOF(dsx.dumps(sample_document()))
    reader = dsx.open_document(stream)
    with pytest.raises(TypeError, match="binary"):
        reader.finish()
    assert reader.closed
    assert not reader.content_verified
    assert not stream.closed


def test_finish_before_iteration_checks_document_without_yielding_records() -> None:
    stream = io.BytesIO(dsx.dumps(sample_document()))
    with dsx.open_document(stream) as reader:
        reader.finish()
        assert reader.content_verified
        assert list(reader) == []
    assert not stream.closed


def test_materialization_rejects_active_event_consumption() -> None:
    with dsx.open_document(io.BytesIO(dsx.dumps(sample_document()))) as reader:
        events = reader._iter_events()
        assert next(events)[0] == "record"
        with pytest.raises(ValueError, match="events"):
            next(reader)
        reader.finish()
        assert reader.content_verified
        assert list(events) == []


@pytest.mark.parametrize("owned", [False, True])
def test_constructor_failure_closes_only_owned_stream(monkeypatch, owned: bool) -> None:
    stream = io.BytesIO(b"broken header")
    if owned:
        monkeypatch.setattr(streaming, "open", lambda *args: stream, raising=False)
    with pytest.raises(dsx.FormatError):
        dsx.open_document("broken.dsx" if owned else stream)
    assert stream.closed is owned


@pytest.mark.parametrize("owned", [False, True])
def test_stream_read_failure_closes_only_owned_stream(monkeypatch, owned: bool) -> None:
    class FailedContent(io.BytesIO):
        failed = False

        def read(self, size: int = -1) -> bytes:
            if self.failed:
                raise OSError("read failed")
            return super().read(size)

    stream = FailedContent(dsx.dumps(sample_document()))
    if owned:
        monkeypatch.setattr(streaming, "open", lambda *args: stream, raising=False)
    reader = dsx.open_document("document.dsx" if owned else stream)
    stream.failed = True
    with pytest.raises(OSError, match="read failed"):
        next(reader)
    assert reader.closed
    assert not reader.content_verified
    assert stream.closed is owned


def test_record_budget_allows_exact_number_of_materializations(monkeypatch) -> None:
    data = dsx.dumps(sample_document())
    original = streaming.AsciiBlock
    values = []

    def observed(value):
        values.append(value)
        return original(value)

    monkeypatch.setattr(streaming, "AsciiBlock", observed)
    with dsx.open_document(io.BytesIO(data), limits=dsx.Limits(max_records=1)) as reader:
        assert next(reader) == dsx.AsciiBlock("first")
        with pytest.raises(dsx.ResourceLimitError, match="max_records"):
            next(reader)
    assert values == ["first"]


@pytest.mark.parametrize("short_read", [1, 7, 65535])
def test_short_reads_with_all_record_types_and_large_payload(short_read: int) -> None:
    class PartialStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(min(size, short_read))

    document = dsx.Document(
        sample_document().metadata,
        (
            dsx.Heading(1, "HEADING"),
            dsx.Paragraph((dsx.Text("word " * 15000), dsx.Citation(127))),
            dsx.BulletItem((dsx.Text("a bullet"),)),
            dsx.AsciiBlock(("A" * 98 + "\n") * 1000),
            dsx.Reference(127, "https:example"),
        ),
    )
    data = dsx.dumps(document)
    stream = PartialStream(b"PREFIX" + data)
    stream.seek(6)
    with dsx.open_document(stream) as reader:
        assert tuple(reader) == document.records
        assert reader.content_verified
    assert not stream.closed
