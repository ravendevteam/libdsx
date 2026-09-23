from __future__ import annotations

import io
import struct
import zlib
from datetime import date
from pathlib import Path

import pytest

import libdsx


DOSSIER = Path(__file__).resolve().parent / "fixtures" / "DossierRev1.dsx"


def sections(data: bytes) -> tuple[bytes, bytes]:
    stored_metadata_length = struct.unpack_from("<I", data, 12)[0]
    return zlib.decompress(data[32 : 32 + stored_metadata_length]), zlib.decompress(data[32 + stored_metadata_length :])


def sample_document() -> libdsx.Document:
    return libdsx.Document(
        libdsx.Metadata("TEST DOCUMENT", ("Ada", "Grace"), 2, date(2026, 9, 22), {"z-field": "last", "a.field": "first", "empty": ""}),
        (
            libdsx.Heading(1, "FIRST SECTION"),
            libdsx.Paragraph((libdsx.Text("See reference"), libdsx.Citation(1), libdsx.Text(" for details."))),
            libdsx.Heading(2, "DETAILS"),
            libdsx.BulletItem((libdsx.Text("A useful item."),)),
            libdsx.AsciiBlock("  +---+  \n  | A |\n  +---+\n"),
            libdsx.Reference(1, "https://example.com"),
            libdsx.Reference(4294967295, "urn:example:uncited"),
        ),
    )


def test_checked_in_specification_roundtrip(trace) -> None:
    trace(f"Read reference document {DOSSIER.name}")
    original_bytes = DOSSIER.read_bytes()
    document = libdsx.load(DOSSIER)
    assert document.metadata.title == "DOSSIER BINARY FORMAT 1.0"
    assert document.metadata.authors == ("Elias A. Murphy",)
    assert document.metadata.revision == 1
    assert document.metadata.date == date(2026, 9, 22)
    assert document.metadata.additional == {"subject": "Dossier 1.0 binary format specification"}
    assert document.metadata.dsx_version == "1.0"
    assert len(document.records) == 96
    trace("Validated 96 content records and preserved subject metadata")
    rewritten_bytes = libdsx.dumps(document)
    assert sections(rewritten_bytes) == sections(original_bytes)
    assert libdsx.loads(rewritten_bytes) == document
    libdsx.validate(document)
    libdsx.validate_file(DOSSIER)
    trace("Re-encoded document has identical uncompressed metadata and content")


def test_sample_metadata_inspection(trace) -> None:
    inspection = libdsx.read_metadata(DOSSIER)
    trace(f"Inspected metadata for {inspection.metadata.title}")
    assert inspection.metadata == libdsx.load(DOSSIER).metadata
    assert inspection.content_verified is False
    trace("Content is explicitly reported as unverified")


def test_all_record_types_roundtrip_through_path_and_stream(tmp_path, trace) -> None:
    document = sample_document()
    destination = tmp_path / "all-records.dsx"
    trace("Write headings, inline citation, paragraph, bullet, ASCII block, and references")
    assert libdsx.dump(document, destination) is None
    assert libdsx.load(destination) == document
    assert libdsx.validate_file(destination) is None
    stream = io.BytesIO()
    assert libdsx.dump(document, stream, compression_level=0) is None
    assert not stream.closed
    stream.seek(0)
    assert libdsx.load(stream) == document
    assert not stream.closed
    stream.seek(0)
    assert libdsx.validate_file(stream) is None
    trace("Path and binary stream roundtrips preserve the structured document")


def test_writer_sorts_additional_fields_canonically() -> None:
    first = libdsx.Document(libdsx.Metadata("ORDERING", ("Ada",), 1, date(2026, 1, 1), {"z": "3", "a_": "2", "a-": "1"}))
    second = libdsx.Document(libdsx.Metadata("ORDERING", ("Ada",), 1, date(2026, 1, 1), {"a-": "1", "a_": "2", "z": "3"}))
    first_encoded = libdsx.dumps(first)
    second_encoded = libdsx.dumps(second)
    assert sections(first_encoded) == sections(second_encoded)
    metadata_bytes, _ = sections(first_encoded)
    assert metadata_bytes.endswith(b"\x80\x04\x02a-1\x80\x04\x02a_2\x80\x03\x01z3")
    assert libdsx.loads(first_encoded) == first


def test_writer_merges_adjacent_text_without_crossing_citations() -> None:
    document = libdsx.Document(
        libdsx.Metadata("TEXT MERGING", ("Ada",), 1, date(2026, 1, 1)),
        (
            libdsx.Paragraph((libdsx.Text("Hello"), libdsx.Text(" world"), libdsx.Citation(1), libdsx.Text(" and"), libdsx.Text(" more."))),
            libdsx.BulletItem((libdsx.Text("One"), libdsx.Text(" item."))),
            libdsx.Reference(1, "https://example.com"),
        ),
    )
    encoded = libdsx.dumps(document)
    decoded = libdsx.loads(encoded)
    assert decoded.records[0] == libdsx.Paragraph((libdsx.Text("Hello world"), libdsx.Citation(1), libdsx.Text(" and more.")))
    assert decoded.records[1] == libdsx.BulletItem((libdsx.Text("One item."),))
    assert sections(libdsx.dumps(decoded)) == sections(encoded)


def test_rewriting_preserves_unfamiliar_metadata_when_changing_content(tmp_path) -> None:
    document = libdsx.load(DOSSIER)
    edited = libdsx.Document(document.metadata, (libdsx.Paragraph((libdsx.Text("Replacement body."),)),))
    destination = tmp_path / "edited.dsx"
    libdsx.dump(edited, destination)
    restored = libdsx.load(destination)
    assert restored.metadata.additional == document.metadata.additional
    assert restored.records == edited.records


@pytest.mark.parametrize("value", ("", "\n", "\n\n", "  ", "FIRST\n", "\nLAST", "FIRST\n\nLAST  \n"))
def test_ascii_block_payload_is_preserved_byte_for_byte(value: str) -> None:
    document = libdsx.Document(libdsx.Metadata("ASCII PRESERVATION", ("Ada",), 1, date(2026, 1, 1)), (libdsx.AsciiBlock(value),))
    encoded = libdsx.dumps(document)
    assert libdsx.loads(encoded) == document
    assert sections(encoded)[1] == b"\x04" + bytes((len(value),)) + value.encode("ascii")


def test_literal_reference_marker_is_not_a_citation() -> None:
    document = libdsx.Document(libdsx.Metadata("LITERAL MARKER", ("Ada",), 1, date(2026, 1, 1)), (libdsx.Paragraph((libdsx.Text("Literal [1]."),)),))
    assert libdsx.loads(libdsx.dumps(document)) == document


def test_plaintext_dsx_extension_is_not_implicitly_imported(tmp_path) -> None:
    destination = tmp_path / "plaintext.dsx"
    destination.write_text("TITLE\nA body.\n", encoding="ascii")
    with pytest.raises(libdsx.DSXError):
        libdsx.load(destination)
