from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import libdsx


DATA = Path(__file__).resolve().parent / "data"
TITLE_BLOCK = "\n\n\n" + " " * 46 + "TITLE\n" + " " * 47 + "Ada\n\n" + " " * 40 + "Rev. 1, 01.02.2026\n\n\n\n"


def document(*records: libdsx.Record) -> libdsx.Document:
    return libdsx.Document(libdsx.Metadata("TITLE", ("Ada",), 1, date(2026, 1, 2)), records)


def paragraph(value: str) -> libdsx.Paragraph:
    return libdsx.Paragraph((libdsx.Text(value),))


def test_specification_matches_supplied_plaintext(trace) -> None:
    source = libdsx.load(DATA / "DossierRev2.dsx")
    expected = (DATA / "DossierRev2.txt").read_text(encoding="ascii")
    actual = libdsx.render(source)
    trace("Render specification revision 2 to canonical text")
    assert actual == expected
    assert "\r" not in actual
    assert actual.endswith("\n")
    assert max(map(len, actual.splitlines())) <= 98
    trace(f"Matched all {len(actual)} characters of the supplied plaintext")


def test_empty_document_contains_complete_title_block() -> None:
    assert libdsx.render(document()) == TITLE_BLOCK


def test_title_centering_author_order_revision_and_date() -> None:
    source = libdsx.Document(libdsx.Metadata("ABCD", ("Ada", "Grace Hopper"), 4294967295, date(1, 1, 2), {"subject": "not displayed"}))
    expected = (
        "\n\n\n"
        + " " * 47 + "ABCD\n"
        + " " * 47 + "Ada\n"
        + " " * 43 + "Grace Hopper\n\n"
        + " " * 35 + "Rev. 4294967295, 01.02.0001\n\n\n\n"
    )
    assert libdsx.render(source) == expected


def test_heading_counters_reset_and_spacing_follows_next_record() -> None:
    source = document(
        libdsx.Heading(1, "ALPHA"),
        libdsx.Heading(2, "DETAIL"),
        libdsx.Heading(3, "SUB"),
        libdsx.Heading(2, "OTHER"),
        paragraph("A paragraph."),
        libdsx.Heading(1, "BETA"),
        libdsx.Heading(2, "CHILD"),
    )
    expected = "1. ALPHA\n========\n\n1.1 DETAIL\n\n1.1.1 SUB\n\n1.2 OTHER\n\nA paragraph.\n\n\n2. BETA\n=======\n\n2.1 CHILD\n"
    assert libdsx.render(source) == TITLE_BLOCK + expected


def test_paragraph_wraps_greedily_at_98_columns() -> None:
    first = "A" * 48
    second = "B" * 49
    source = document(paragraph(f"{first} {second} C"), paragraph("Next paragraph."))
    assert libdsx.render(source) == TITLE_BLOCK + first + " " + second + "\nC\n\nNext paragraph.\n"


def test_hyphenated_words_remain_whole() -> None:
    token = "A" * 47 + "-" + "B" * 48
    source = document(paragraph(f"prefix {token} x"))
    assert libdsx.render(source) == TITLE_BLOCK + "prefix\n" + token + " x\n"


def test_bullets_include_prefix_in_width_and_indent_continuations() -> None:
    first = "A" * 47
    second = "B" * 48
    source = document(
        libdsx.BulletItem((libdsx.Text(f"{first} {second} C"),)),
        libdsx.BulletItem((libdsx.Text("Next item."),)),
        paragraph("Following paragraph."),
    )
    expected = "* " + first + " " + second + "\n  C\n\n* Next item.\n\nFollowing paragraph.\n"
    assert libdsx.render(source) == TITLE_BLOCK + expected


@pytest.mark.parametrize(("number", "url_length", "wrapped"), ((1, 94, False), (1, 95, True), (4294967295, 85, False), (4294967295, 86, True), (1, 98, True)))
def test_reference_wraps_only_between_marker_and_url(number: int, url_length: int, wrapped: bool) -> None:
    url = "x:" + "a" * (url_length - 2)
    expected = f"[{number}]" + ("\n" if wrapped else " ") + url + "\n"
    assert libdsx.render(document(libdsx.Reference(number, url))) == TITLE_BLOCK + expected


def test_citations_and_literal_markers_render_without_invented_spaces() -> None:
    source = document(
        libdsx.Paragraph((libdsx.Text("Literal [2], cited"), libdsx.Citation(2), libdsx.Text("."))),
        libdsx.Reference(2, "https://example.com/two"),
        libdsx.Reference(3, "https://example.com/three"),
    )
    expected = "Literal [2], cited[2].\n\n[2] https://example.com/two\n[3] https://example.com/three\n"
    assert libdsx.render(source) == TITLE_BLOCK + expected


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        ("", "Before.\n\n\nAfter.\n"),
        ("ABC", "Before.\n\nABC\n\nAfter.\n"),
        ("ABC\n", "Before.\n\nABC\n\nAfter.\n"),
        ("ABC\n\n", "Before.\n\nABC\n\n\nAfter.\n"),
        ("\n", "Before.\n\n\n\nAfter.\n"),
        ("\nABC", "Before.\n\n\nABC\n\nAfter.\n"),
        ("  ABC  \n \n", "Before.\n\n  ABC  \n \n\nAfter.\n"),
    ),
)
def test_ascii_blocks_use_lf_terminators_and_preserve_additional_blank_lines(value: str, expected: str) -> None:
    source = document(paragraph("Before."), libdsx.AsciiBlock(value), paragraph("After."))
    assert libdsx.render(source) == TITLE_BLOCK + expected


@pytest.mark.parametrize(("value", "expected"), (("", ""), ("ABC", "ABC\n"), ("ABC\n", "ABC\n"), ("ABC\n\n", "ABC\n\n"), ("\n", "\n"), ("\n\n", "\n\n")))
def test_final_ascii_block_does_not_gain_an_extra_line(value: str, expected: str) -> None:
    assert libdsx.render(document(libdsx.AsciiBlock(value))) == TITLE_BLOCK + expected


def test_empty_ascii_blocks_keep_inter_record_gaps() -> None:
    assert libdsx.render(document(libdsx.AsciiBlock(""), paragraph("After."))) == TITLE_BLOCK + "\nAfter.\n"
    assert libdsx.render(document(paragraph("Before."), libdsx.AsciiBlock(""))) == TITLE_BLOCK + "Before.\n\n"
    assert libdsx.render(document(libdsx.AsciiBlock(""), libdsx.AsciiBlock(""))) == TITLE_BLOCK + "\n"
    assert libdsx.render(document(libdsx.AsciiBlock(""), libdsx.Heading(1, "NEXT"))) == TITLE_BLOCK + "\n\n1. NEXT\n=======\n"


def test_renderer_rejects_invalid_document() -> None:
    with pytest.raises(libdsx.ValidationError):
        libdsx.render(document(paragraph("A" * 99)))
