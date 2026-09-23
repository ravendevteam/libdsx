from __future__ import annotations

import tracemalloc
from datetime import date

import pytest

from libdsx.errors import ResourceLimitError, ValidationError
from libdsx.limits import Limits
from libdsx.model import AsciiBlock, Citation, Document, Metadata, Paragraph, Reference, Text
from libdsx.validation import ValidationState, validate


@pytest.fixture
def metadata() -> Metadata:
    return Metadata("TITLE", ("Author",), 1, date(2026, 9, 23))


def feed_text(state: ValidationState, text: str, chunk_size: int) -> None:
    state.begin_inline(1)
    for offset in range(0, len(text), chunk_size):
        state.text(text[offset:offset + chunk_size])
    state.end_inline()


@pytest.mark.parametrize("chunk_size", [1, 2, 7, 95, 96, 97, 98, 99, 65536])
@pytest.mark.parametrize(("kind", "width"), [(2, 98), (3, 96)])
def test_prose_validation_is_independent_of_chunks(chunk_size: int, kind: int, width: int) -> None:
    state = ValidationState()
    state.begin_record(kind)
    feed_text(state, "A" * width + " " + "B" * (width - 3), chunk_size)
    state.begin_inline(2)
    state.citation(1)
    state.end_inline()
    feed_text(state, " and text", chunk_size)
    state.end_record()
    state.begin_record(5)
    state.reference(1, "https:example")
    state.end_record()
    state.finish()


@pytest.mark.parametrize("chunk_size", [1, 2, 97, 98, 99, 65536])
@pytest.mark.parametrize("value", [" leading", "trailing ", "two  spaces", "A" * 99, "okay " + "A" * 99 + " end", "valid \tbad"])
def test_invalid_prose_remains_invalid_across_chunk_boundaries(chunk_size: int, value: str) -> None:
    state = ValidationState()
    state.begin_record(2)
    with pytest.raises(ValidationError):
        feed_text(state, value, chunk_size)
        state.end_record()


@pytest.mark.parametrize("chunk_size", [1, 2, 97, 98, 99, 65536])
@pytest.mark.parametrize("suffix", ["", "\n", "\n\n"])
def test_ascii_validation_preserves_line_state(chunk_size: int, suffix: str) -> None:
    state = ValidationState()
    state.begin_record(4)
    value = "\n" + "A" * 98 + "\n" + "B" * 98 + suffix
    for offset in range(0, len(value), chunk_size):
        state.ascii(value[offset:offset + chunk_size])
    state.end_record()
    state.finish()


@pytest.mark.parametrize("chunk_size", [1, 2, 97, 98, 99, 65536])
@pytest.mark.parametrize("value", ["A" * 99, "okay\n" + "A" * 99, "valid\n\r"])
def test_invalid_ascii_remains_invalid_across_chunk_boundaries(chunk_size: int, value: str) -> None:
    state = ValidationState()
    state.begin_record(4)
    with pytest.raises(ValidationError):
        for offset in range(0, len(value), chunk_size):
            state.ascii(value[offset:offset + chunk_size])


def test_empty_text_payload_is_rejected_after_empty_chunks() -> None:
    state = ValidationState()
    state.begin_record(2)
    state.begin_inline(1)
    state.text("")
    state.text("")
    with pytest.raises(ValidationError, match="must not be empty"):
        state.end_inline()


def test_pending_citation_budget_counts_distinct_numbers() -> None:
    state = ValidationState(Limits(max_pending_citations=1))
    for number in (17, 17, 17):
        state.begin_record(2)
        state.begin_inline(2)
        state.citation(number)
        state.end_inline()
        state.end_record()
    assert state.pending_citations == {17}
    state.begin_record(2)
    state.begin_inline(2)
    with pytest.raises(ResourceLimitError, match="max_pending_citations"):
        state.citation(18)
    assert state.pending_citations == {17}


def test_resolved_citations_are_discarded() -> None:
    state = ValidationState()
    for number in (1, 7):
        state.begin_record(2)
        state.begin_inline(2)
        state.citation(number)
        state.end_inline()
        state.end_record()
    state.begin_record(5)
    state.reference(1, "a:")
    state.end_record()
    assert state.pending_citations == {7}
    state.begin_record(5)
    state.reference(7, "a:")
    state.end_record()
    assert not state.pending_citations
    state.finish()


def test_many_uncited_references_keep_no_reference_set() -> None:
    state = ValidationState(Limits(max_pending_citations=0))
    for number in range(1, 10001):
        state.begin_record(5)
        state.reference(number, "a:")
        state.end_record()
    assert state.pending_citations == set()
    state.finish()


def test_unresolved_citation_diagnostic_is_bounded() -> None:
    state = ValidationState()
    for number in range(1, 10001):
        state.begin_record(2)
        state.begin_inline(2)
        state.citation(number)
        state.end_inline()
        state.end_record()
    with pytest.raises(ValidationError) as error:
        state.finish()
    message = str(error.value)
    assert message.startswith("document: unresolved citation numbers: 1, 2, 3")
    assert message.endswith("... (10000 total)")
    assert len(message) < 250


def test_record_budget_rejects_before_accepting_record() -> None:
    state = ValidationState(Limits(max_records=1))
    state.begin_record(4)
    state.end_record()
    with pytest.raises(ResourceLimitError, match="max_records"):
        state.begin_record(4)
    assert state.record_count == 1


def test_inline_budget_is_global_across_records() -> None:
    state = ValidationState(Limits(max_inlines=1))
    state.begin_record(2)
    feed_text(state, "text", 2)
    state.end_record()
    state.begin_record(2)
    with pytest.raises(ResourceLimitError, match="max_inlines"):
        state.begin_inline(1)
    assert state.inline_count == 1


@pytest.mark.parametrize(
    ("records", "limits"),
    [
        ((AsciiBlock(""),), Limits(max_records=0)),
        ((Paragraph((Text("one "), Text("two"))),), Limits(max_inlines=1)),
        ((Paragraph((Citation(1),)), Reference(1, "a:")), Limits(max_pending_citations=0)),
    ],
)
def test_model_validation_enforces_count_budgets(metadata: Metadata, records: tuple, limits: Limits) -> None:
    with pytest.raises(ResourceLimitError):
        validate(Document(metadata, records), limits=limits)


@pytest.mark.parametrize("record_type", ["prose", "ascii"])
def test_model_validation_memory_stays_bounded(metadata: Metadata, record_type: str) -> None:
    if record_type == "prose":
        record = Paragraph((Text("word " * 250000), Text("end")))
    else:
        record = AsciiBlock(("A" * 80 + "\n") * 16000)
    document = Document(metadata, (record,))
    tracemalloc.start()
    try:
        validate(document)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 262144


@pytest.mark.parametrize("value", [False, 1, {}, "limits"])
def test_state_rejects_invalid_limits(value: object) -> None:
    with pytest.raises(TypeError, match="Limits"):
        ValidationState(value)
