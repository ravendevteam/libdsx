from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime

import pytest

from libdsx import ValidationError, validate
from libdsx.model import AsciiBlock, BulletItem, Citation, Document, Heading, Metadata, Paragraph, Reference, Text
from libdsx.validation import validate_metadata


@pytest.fixture
def metadata() -> Metadata:
    return Metadata("VALID TITLE", ("Alice Author", "Bob Author"), 1, date(2026, 9, 22))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dsx_version", "2.0"),
        ("dsx_version", 1),
        ("title", ""),
        ("title", "123 !"),
        ("title", "Title"),
        ("title", " TITLE"),
        ("title", "TITLE "),
        ("title", "A" * 99),
        ("title", "CAF\u00c9"),
        ("title", None),
        ("authors", ()),
        ("authors", []),
        ("authors", "Author"),
        ("authors", ("",)),
        ("authors", (" Author",)),
        ("authors", ("Author ",)),
        ("authors", ("A" * 99,)),
        ("authors", ("Author\tName",)),
        ("authors", (None,)),
        ("revision", 0),
        ("revision", -1),
        ("revision", 4294967296),
        ("revision", True),
        ("revision", 1.0),
        ("revision", "1"),
        ("date", "09.22.2026"),
        ("date", datetime(2026, 9, 22)),
        ("date", None),
        ("additional", []),
        ("additional", {"": "value"}),
        ("additional", {"Upper": "value"}),
        ("additional", {"1key": "value"}),
        ("additional", {"a" * 65: "value"}),
        ("additional", {"key/part": "value"}),
        ("additional", {"key\n": "value"}),
        ("additional", {1: "value"}),
        ("additional", {"key": 1}),
        ("additional", {"key": "line\nbreak"}),
    ],
)
def test_rejects_invalid_metadata(metadata: Metadata, field: str, value: object) -> None:
    with pytest.raises(ValidationError, match="metadata"):
        validate_metadata(replace(metadata, **{field: value}))


@pytest.mark.parametrize("key", ["dsx_version", "title", "authors", "revision", "date"])
def test_rejects_reserved_additional_keys(metadata: Metadata, key: str) -> None:
    with pytest.raises(ValidationError, match="reserved"):
        validate_metadata(replace(metadata, additional={key: ""}))


@pytest.mark.parametrize("calendar_date", [date(1, 1, 1), date(2000, 2, 29), date(9999, 12, 31)])
def test_accepts_metadata_boundaries(metadata: Metadata, calendar_date: date) -> None:
    validate_metadata(
        replace(
            metadata,
            title="A" * 98,
            authors=("a" * 98, "123", "A  B"),
            revision=4294967295,
            date=calendar_date,
            additional={"z": "", "a" + "0_.-" * 15 + "abc": " !~"},
        )
    )


def test_accepts_empty_document(metadata: Metadata) -> None:
    validate(Document(metadata))


@pytest.mark.parametrize("value", [None, {}, "document"])
def test_rejects_non_document(value: object) -> None:
    with pytest.raises(ValidationError, match="Document"):
        validate(value)


def test_rejects_non_metadata() -> None:
    with pytest.raises(ValidationError, match="Metadata"):
        validate_metadata(None)


@pytest.mark.parametrize("records", [[], None, "paragraph"])
def test_rejects_non_tuple_records(metadata: Metadata, records: object) -> None:
    with pytest.raises(ValidationError, match="records"):
        validate(Document(metadata, records))


def test_rejects_unknown_record(metadata: Metadata) -> None:
    with pytest.raises(ValidationError, match="unknown document record"):
        validate(Document(metadata, (object(),)))


@pytest.mark.parametrize("depth", [0, 9, -1, True, 1.0, "1", None])
def test_rejects_invalid_heading_depth(metadata: Metadata, depth: object) -> None:
    with pytest.raises(ValidationError, match="depth"):
        validate(Document(metadata, (Heading(depth, "TITLE"),)))


@pytest.mark.parametrize(
    "records",
    [
        (Heading(2, "TITLE"),),
        (Paragraph((Text("First"),)), Heading(2, "TITLE")),
        (Heading(1, "TITLE"), Heading(3, "TITLE")),
        (Heading(1, "TITLE"), Heading(2, "TITLE"), Heading(1, "TITLE"), Heading(3, "TITLE")),
    ],
)
def test_rejects_heading_depth_skips(metadata: Metadata, records: tuple) -> None:
    with pytest.raises(ValidationError, match="first heading"):
        validate(Document(metadata, records))


def test_accepts_all_heading_depths_and_decreases(metadata: Metadata) -> None:
    records = tuple(Heading(depth, "TITLE") for depth in range(1, 9))
    records += (Heading(4, "TITLE"), Heading(4, "TITLE"), Heading(1, "TITLE"))
    validate(Document(metadata, records))


@pytest.mark.parametrize("title", ["", "123", "lower", "Mixed", " TITLE", "TITLE ", "A\nB", None])
def test_rejects_invalid_heading_title(metadata: Metadata, title: object) -> None:
    with pytest.raises(ValidationError, match="title"):
        validate(Document(metadata, (Heading(1, title),)))


def test_heading_width_includes_generated_number(metadata: Metadata) -> None:
    validate(Document(metadata, (Heading(1, "A" * 95),)))
    with pytest.raises(ValidationError, match="numbered heading"):
        validate(Document(metadata, (Heading(1, "A" * 96),)))
    records = (Heading(1, "A" * 95),) * 9
    validate(Document(metadata, records))
    with pytest.raises(ValidationError, match="numbered heading"):
        validate(Document(metadata, records + (Heading(1, "A" * 95),)))


def test_deep_heading_width_includes_all_number_components(metadata: Metadata) -> None:
    records = tuple(Heading(depth, "TITLE") for depth in range(1, 8))
    validate(Document(metadata, records + (Heading(8, "A" * 82),)))
    with pytest.raises(ValidationError, match="numbered heading"):
        validate(Document(metadata, records + (Heading(8, "A" * 83),)))


@pytest.mark.parametrize("record_type", [Paragraph, BulletItem])
@pytest.mark.parametrize(
    "inlines",
    [
        (),
        [],
        None,
        (object(),),
        (Text(""),),
        (Text(None),),
        (Text(" leading"),),
        (Text("trailing "),),
        (Text("two  spaces"),),
        (Text("two "), Text(" spaces")),
        (Text(" "), Citation(1)),
        (Citation(1), Text(" ")),
        (Text("tab\tcharacter"),),
        (Text("line\nfeed"),),
        (Text("carriage\rreturn"),),
        (Text("delete\x7f"),),
        (Text("caf\u00e9"),),
        (Citation(0),),
        (Citation(-1),),
        (Citation(4294967296),),
        (Citation(True),),
        (Citation(1.0),),
        (Citation("1"),),
    ],
)
def test_rejects_invalid_inline_content(metadata: Metadata, record_type: type, inlines: object) -> None:
    with pytest.raises(ValidationError):
        validate(Document(metadata, (record_type(inlines), Reference(1, "https:example"))))


@pytest.mark.parametrize("record_type", [Paragraph, BulletItem])
def test_accepts_adjacent_text_and_cross_record_spacing(metadata: Metadata, record_type: type) -> None:
    record = record_type((Text("Hello"), Text(" world "), Citation(1), Text(" and"), Text(" more.")))
    validate(Document(metadata, (record, Reference(1, "https:example"))))


def test_literal_brackets_do_not_require_a_reference(metadata: Metadata) -> None:
    validate(Document(metadata, (Paragraph((Text("Literal [1]"),)),)))


def test_citation_records_require_a_reference(metadata: Metadata) -> None:
    with pytest.raises(ValidationError, match="unresolved citation numbers: 1, 7"):
        validate(Document(metadata, (Paragraph((Citation(7), Citation(1), Citation(7))),)))


@pytest.mark.parametrize(("record_type", "width"), [(Paragraph, 98), (BulletItem, 96)])
def test_prose_width_is_measured_per_unbreakable_token(metadata: Metadata, record_type: type, width: int) -> None:
    validate(Document(metadata, (record_type((Text("A" * width + " " + "B" * width),)),)))
    with pytest.raises(ValidationError, match="unbreakable tokens"):
        validate(Document(metadata, (record_type((Text("A" * (width + 1)),)),)))
    with pytest.raises(ValidationError, match="unbreakable tokens"):
        validate(Document(metadata, (record_type((Text("A" * 50 + "-" + "B" * 50),)),)))


def test_citations_contribute_to_unbreakable_token_width(metadata: Metadata) -> None:
    records = (Paragraph((Text("A" * 95), Citation(1))), Reference(1, "a:"))
    validate(Document(metadata, records))
    with pytest.raises(ValidationError, match="unbreakable tokens"):
        validate(Document(metadata, (Paragraph((Text("A" * 96), Citation(1))), records[1])))


@pytest.mark.parametrize("value", ["", "\n", "\n\n", "line\n", "\nline\n", "line  ", " " * 98, "A" * 98 + "\n" + "B" * 98])
def test_accepts_ascii_block_boundaries(metadata: Metadata, value: str) -> None:
    validate(Document(metadata, (AsciiBlock(value),)))


@pytest.mark.parametrize("value", [None, b"ASCII", "\r", "\t", "\x00", "\x7f", "\u00e9", "A" * 99, "first\n" + "B" * 99])
def test_rejects_invalid_ascii_block(metadata: Metadata, value: object) -> None:
    with pytest.raises(ValidationError, match="ASCII|string"):
        validate(Document(metadata, (AsciiBlock(value),)))


@pytest.mark.parametrize("number", [0, -1, 4294967296, True, 1.0, "1", None])
def test_rejects_invalid_reference_number(metadata: Metadata, number: object) -> None:
    with pytest.raises(ValidationError, match="number"):
        validate(Document(metadata, (Reference(number, "a:"),)))


@pytest.mark.parametrize("url", ["", "relative", "//host/path", "1scheme:path", "+scheme:path", "a_b:path", "a:has space", "a:\n", "a:\u00e9", None])
def test_rejects_invalid_reference_url(metadata: Metadata, url: object) -> None:
    with pytest.raises(ValidationError, match="url"):
        validate(Document(metadata, (Reference(1, url),)))


@pytest.mark.parametrize("url", ["a:", "HTTP://example.test/path", "mailto:person@example.test", "A+1.-:value", "urn:isbn:123"])
def test_accepts_reference_schemes_without_extra_url_assumptions(metadata: Metadata, url: str) -> None:
    validate(Document(metadata, (Reference(1, url),)))


def test_reference_url_can_wrap_away_from_number(metadata: Metadata) -> None:
    validate(Document(metadata, (Reference(4294967295, "a:" + "x" * 96),)))
    with pytest.raises(ValidationError, match="unbreakable URL"):
        validate(Document(metadata, (Reference(1, "a:" + "x" * 97),)))


@pytest.mark.parametrize("numbers", [(1, 1), (2, 1), (1, 3, 2)])
def test_references_must_increase_strictly(metadata: Metadata, numbers: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        validate(Document(metadata, tuple(Reference(number, "a:") for number in numbers)))


@pytest.mark.parametrize("record", [Heading(1, "HEADING"), Paragraph((Text("Text"),)), BulletItem((Text("Text"),)), AsciiBlock("")])
def test_references_must_be_the_final_group(metadata: Metadata, record: object) -> None:
    with pytest.raises(ValidationError, match="final group"):
        validate(Document(metadata, (Reference(1, "a:"), record)))


def test_accepts_uncited_references_and_nonconsecutive_numbers(metadata: Metadata) -> None:
    validate(
        Document(
            metadata,
            (
                Paragraph((Citation(4294967295), Text(" and "), Citation(1))),
                Reference(1, "a:"),
                Reference(8, "a:"),
                Reference(4294967295, "a:"),
            ),
        )
    )
