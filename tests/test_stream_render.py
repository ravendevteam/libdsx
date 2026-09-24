from __future__ import annotations

import hashlib
import importlib
import io
from datetime import date
from pathlib import Path
from textwrap import fill

import pytest

import libdsx
from libdsx.errors import ResourceLimitError
from libdsx.limits import Limits


rendering = importlib.import_module("libdsx.render")
DATA = Path(__file__).resolve().parent / "data"
TITLE_BLOCK = "\n\n\n" + " " * 46 + "TITLE\n" + " " * 47 + "Ada\n\n" + " " * 40 + "Rev. 1, 01.02.2026\n\n\n\n"


def document(*records: libdsx.Record) -> libdsx.Document:
    return libdsx.Document(libdsx.Metadata("TITLE", ("Ada",), 1, date(2026, 1, 2)), records)


@pytest.mark.parametrize("source_kind", ("document", "path", "stream"))
def test_iter_render_matches_specification(source_kind: str) -> None:
    path = DATA / "DossierRev2.dsx"
    if source_kind == "document":
        source = libdsx.load(path)
    elif source_kind == "stream":
        source = io.BytesIO(path.read_bytes())
    else:
        source = path
    assert "".join(rendering.iter_render(source)) == (DATA / "DossierRev2.txt").read_text(encoding="ascii")
    if source_kind == "stream":
        assert not source.closed


@pytest.mark.parametrize("chunk_size", (1, 7, 97, 65536))
@pytest.mark.parametrize("source_kind", ("document", "stream"))
def test_wrapping_and_citations_survive_arbitrary_chunk_boundaries(chunk_size: int, source_kind: str, monkeypatch) -> None:
    from libdsx import streaming

    monkeypatch.setattr(rendering, "_CHUNK_SIZE", chunk_size)
    monkeypatch.setattr(streaming, "CHUNK_SIZE", chunk_size)
    first = "A" * 90 + "[1]. words across chunk boundaries " + "middle " * 40 + "end"
    second = "hyphenated-word " * 30 + "end"
    candidate = document(
        libdsx.Paragraph((libdsx.Text("A" * 90), libdsx.Citation(1), libdsx.Text(". words across chunk boundaries "), libdsx.Text("middle " * 40), libdsx.Text("end"))),
        libdsx.BulletItem((libdsx.Text(second),)),
        libdsx.Reference(1, "https://example.com"),
    )
    source = candidate if source_kind == "document" else io.BytesIO(libdsx.dumps(candidate))
    expected = TITLE_BLOCK + fill(first, width=98, break_long_words=False, break_on_hyphens=False) + "\n\n"
    expected += fill(second, width=98, initial_indent="* ", subsequent_indent="  ", break_long_words=False, break_on_hyphens=False)
    expected += "\n\n[1] https://example.com\n"
    assert "".join(rendering.iter_render(source)) == expected


def test_large_ascii_file_renders_without_materializing_records(monkeypatch) -> None:
    from libdsx import codec, streaming

    body = "A\n" * 500000
    encoded = libdsx.dumps(document(libdsx.AsciiBlock(body)))

    def forbidden(*args, **kwargs):
        raise AssertionError("stream rendering materialized a document or record")

    class HashSink:
        def __init__(self):
            self.digest = hashlib.sha256()
            self.maximum_chunk = 0
            self.size = 0

        def write(self, value: str) -> int:
            self.maximum_chunk = max(self.maximum_chunk, len(value))
            self.size += len(value)
            self.digest.update(value.encode("ascii"))
            return len(value)

    monkeypatch.setattr(codec, "load", forbidden)
    monkeypatch.setattr(streaming.DocumentReader, "_materialize", forbidden)
    sink = HashSink()
    rendering.render_to(io.BytesIO(encoded), sink)
    expected = (TITLE_BLOCK + body).encode("ascii")
    assert sink.size == len(expected)
    assert sink.digest.digest() == hashlib.sha256(expected).digest()
    assert sink.maximum_chunk <= 65536


def test_render_to_supports_partial_text_writes() -> None:
    class PartialSink:
        def __init__(self):
            self.data = []

        def write(self, value: str) -> int:
            accepted = min(3, len(value))
            self.data.append(value[:accepted])
            return accepted

    sink = PartialSink()
    rendering.render_to(document(libdsx.AsciiBlock("ABC\n\n")), sink)
    assert "".join(sink.data) == TITLE_BLOCK + "ABC\n\n"


def test_one_character_writes_do_not_repeatedly_copy_the_remaining_chunk() -> None:
    slices = []

    class ObservedText(str):
        def __getitem__(self, key):
            result = super().__getitem__(key)
            if isinstance(key, slice):
                slices.append(len(result))
            return result

    class OneCharacterSink:
        def __init__(self):
            self.data = []

        def write(self, value):
            self.data.append(value[0])
            return 1

    value = ObservedText("AB" * 32768)
    sink = OneCharacterSink()
    rendering._write_text(sink, value)
    assert "".join(sink.data) == value
    assert max(slices) <= 2
    assert sum(slices) < 2 * len(value)


@pytest.mark.parametrize("written", (None, 0, -1, 1000000))
def test_render_to_rejects_invalid_stream_write_counts(written) -> None:
    class FailedSink:
        def write(self, value):
            return written

    with pytest.raises(OSError, match="text stream"):
        rendering.render_to(document(), FailedSink())


def test_failed_output_closes_owned_document_reader(monkeypatch) -> None:
    from libdsx import streaming

    factory = streaming.open_document
    readers = []

    def observed_reader(*args, **kwargs):
        reader = factory(*args, **kwargs)
        readers.append(reader)
        return reader

    class FailedSink:
        def write(self, value):
            raise OSError("output failed")

    monkeypatch.setattr(streaming, "open_document", observed_reader)
    with pytest.raises(OSError, match="output failed"):
        rendering.render_to(DATA / "DossierRev2.dsx", FailedSink())
    assert readers[0].closed
    assert not readers[0].content_verified


def test_render_path_uses_lf_and_replaces_existing_output(tmp_path) -> None:
    target = tmp_path / "output.txt"
    target.write_bytes(b"old")
    rendering.render_to(document(libdsx.AsciiBlock("A\nB")), target)
    assert target.read_bytes() == (TITLE_BLOCK + "A\nB\n").encode("ascii")
    assert list(tmp_path.glob(".libdsx-*.tmp")) == []


def test_late_checksum_failure_preserves_existing_path_and_removes_temporary_file(tmp_path) -> None:
    encoded = bytearray(libdsx.dumps(document(libdsx.AsciiBlock("A\n" * 1000))))
    encoded[-1] ^= 1
    target = tmp_path / "output.txt"
    target.write_bytes(b"old")
    with pytest.raises(libdsx.FormatError):
        rendering.render_to(io.BytesIO(encoded), target)
    assert target.read_bytes() == b"old"
    assert list(tmp_path.glob(".libdsx-*.tmp")) == []


def test_late_checksum_failure_propagates_for_stream_output() -> None:
    encoded = bytearray(libdsx.dumps(document(libdsx.AsciiBlock("A\n" * 1000))))
    encoded[-1] ^= 1
    target = io.StringIO()
    with pytest.raises(libdsx.FormatError):
        rendering.render_to(io.BytesIO(encoded), target)
    assert target.getvalue().startswith(TITLE_BLOCK)
    assert not target.closed


def test_rendered_character_budget_is_inclusive_and_never_overruns_stream() -> None:
    candidate = document(libdsx.AsciiBlock("A\n" * 1000))
    expected = TITLE_BLOCK + "A\n" * 1000
    assert rendering.render(candidate, limits=Limits(max_rendered_chars=len(expected))) == expected
    target = io.StringIO()
    maximum = len(expected) - 1
    with pytest.raises(ResourceLimitError, match="max_rendered_chars"):
        rendering.render_to(candidate, target, limits=Limits(max_rendered_chars=maximum))
    assert len(target.getvalue()) <= maximum


def test_invalid_in_memory_document_produces_no_stream_output() -> None:
    target = io.StringIO()
    with pytest.raises(libdsx.ValidationError):
        rendering.render_to(document(libdsx.Paragraph((libdsx.Text("A" * 99),))), target)
    assert target.getvalue() == ""


def test_render_remains_document_only() -> None:
    with pytest.raises(libdsx.ValidationError, match="Document instance"):
        rendering.render(DATA / "DossierRev2.dsx")
