from __future__ import annotations

import re
from datetime import date
from heapq import nsmallest

from .errors import ValidationError
from .limits import Limits, check_limit, check_limits
from .model import AsciiBlock, BulletItem, Citation, Document, Heading, Metadata, Paragraph, Reference, Text


_MAX_UINT = 4294967295
_REQUIRED_KEYS = {"dsx_version", "title", "authors", "revision", "date"}
_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,63}\Z")
_SCHEME_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")
_ASCII_INVALID = re.compile(r"[^\x20-\x7e\n]")
_ASCII_WIDE = re.compile(r"\n[^\n]{99}")
_PROSE_WIDE = {2: re.compile(r" [^ ]{99}"), 3: re.compile(r" [^ ]{97}")}
_CHUNK_SIZE = 65536


def _fail(context: str, message: str) -> None:
    raise ValidationError(f"{context}: {message}")


def _printable(value: object, context: str, *, empty: bool = True) -> str:
    if not isinstance(value, str):
        _fail(context, "must be a string")
    if not empty and not value:
        _fail(context, "must not be empty")
    if value and (not value.isascii() or not value.isprintable()):
        _fail(context, "must contain only printable ASCII characters")
    return value


def _positive_uint(value: object, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(context, "must be an integer")
    if not 1 <= value <= _MAX_UINT:
        _fail(context, f"must be between 1 and {_MAX_UINT}")


def _title(value: object, context: str) -> str:
    value = _printable(value, context, empty=False)
    if value.startswith(" ") or value.endswith(" "):
        _fail(context, "must not have leading or trailing spaces")
    if any("a" <= character <= "z" for character in value):
        _fail(context, "must not contain lowercase letters")
    if not any("A" <= character <= "Z" for character in value):
        _fail(context, "must contain at least one uppercase letter")
    return value


def _heading_text(heading: Heading, counters: list[int]) -> str:
    counters[heading.depth - 1] += 1
    counters[heading.depth:] = [0] * (8 - heading.depth)
    number = ".".join(str(counter) for counter in counters[: heading.depth])
    if heading.depth == 1:
        number += "."
    return f"{number} {heading.title}"


def validate_metadata(metadata: Metadata) -> None:
    if not isinstance(metadata, Metadata):
        _fail("metadata", "must be a Metadata instance")
    if not isinstance(metadata.dsx_version, str) or metadata.dsx_version != "1.1":
        _fail("metadata.dsx_version", "must be '1.1'")
    title = _title(metadata.title, "metadata.title")
    if len(title) > 98:
        _fail("metadata.title", "must fit within 98 columns")
    if not isinstance(metadata.authors, tuple) or not metadata.authors:
        _fail("metadata.authors", "must be a nonempty tuple of names")
    for index, author in enumerate(metadata.authors):
        context = f"metadata.authors[{index}]"
        author = _printable(author, context, empty=False)
        if len(author) > 98:
            _fail(context, "must fit within 98 columns")
        if author.startswith(" ") or author.endswith(" "):
            _fail(context, "must not have leading or trailing spaces")
    _positive_uint(metadata.revision, "metadata.revision")
    if type(metadata.date) is not date:
        _fail("metadata.date", "must be a datetime.date, excluding datetime.datetime")
    if not isinstance(metadata.additional, dict):
        _fail("metadata.additional", "must be a dictionary")
    for key, value in metadata.additional.items():
        if not isinstance(key, str) or _KEY_PATTERN.fullmatch(key) is None:
            _fail("metadata.additional", "keys must match [a-z][a-z0-9_.-]{0,63}")
        if key in _REQUIRED_KEYS:
            _fail(f"metadata.additional[{key!r}]", "key is reserved")
        _printable(value, f"metadata.additional[{key!r}]")


class ValidationState:
    __slots__ = (
        "limits", "record_count", "inline_count", "pending_citations", "_counters",
        "_heading_depth", "_previous_reference", "_kind", "_inline_index",
        "_inline_kind", "_inline_nonempty", "_display_nonempty", "_last_space",
        "_token_length", "_ascii_line_length",
    )

    def __init__(self, limits: Limits | None = None):
        check_limits(limits)
        self.limits = limits
        self.record_count = 0
        self.inline_count = 0
        self.pending_citations: set[int] = set()
        self._counters = [0] * 8
        self._heading_depth = 0
        self._previous_reference = 0
        self._kind = 0
        self._inline_index = -1
        self._inline_kind = 0
        self._inline_nonempty = False
        self._display_nonempty = False
        self._last_space = False
        self._token_length = 0
        self._ascii_line_length = 0

    @property
    def context(self) -> str:
        return f"document.records[{self.record_count - 1}]"

    @property
    def inline_context(self) -> str:
        return f"{self.context}.inlines[{self._inline_index}]"

    def begin_record(self, kind: int) -> None:
        check_limit(self.limits, "max_records", self.record_count + 1)
        self.record_count += 1
        if self._previous_reference and kind != 5:
            _fail(self.context, "references must form the final group of records")
        if kind not in (1, 2, 3, 4, 5):
            _fail(self.context, "unknown document record type")
        self._kind = kind
        self._inline_index = -1
        self._display_nonempty = False
        self._last_space = False
        self._token_length = 0
        self._ascii_line_length = 0

    def heading(self, depth: int, title: str) -> None:
        if isinstance(depth, bool) or not isinstance(depth, int):
            _fail(f"{self.context}.depth", "must be an integer")
        if not 1 <= depth <= 8:
            _fail(f"{self.context}.depth", "must be between 1 and 8")
        if depth > self._heading_depth + 1:
            _fail(f"{self.context}.depth", "first heading must have depth 1 and later headings may deepen by only one level")
        _title(title, f"{self.context}.title")
        self._counters[depth - 1] += 1
        for index in range(depth, 8):
            self._counters[index] = 0
        number_length = sum(len(str(self._counters[index])) for index in range(depth))
        number_length += depth - 1 if depth > 1 else 1
        if number_length + 1 + len(title) > 98:
            _fail(self.context, "numbered heading must fit within 98 columns")
        self._heading_depth = depth

    def begin_inline(self, kind: int) -> None:
        check_limit(self.limits, "max_inlines", self.inline_count + 1)
        self.inline_count += 1
        self._inline_index += 1
        if kind not in (1, 2):
            _fail(self.inline_context, "must be a Text or Citation instance")
        self._inline_kind = kind
        self._inline_nonempty = False

    def _prose(self, value: str) -> None:
        if not value:
            return
        if (value[0] == " " and (not self._display_nonempty or self._last_space)) or "  " in value:
            _fail(self.context, "display text must have single internal spaces and no surrounding spaces")
        width = 96 if self._kind == 3 else 98
        first_space = value.find(" ")
        if first_space == -1:
            self._token_length += len(value)
            too_wide = self._token_length > width
        else:
            too_wide = self._token_length + first_space > width or _PROSE_WIDE[self._kind].search(value) is not None
            self._token_length = len(value) - value.rfind(" ") - 1
        if too_wide:
            _fail(self.context, f"unbreakable tokens must fit within {width} columns")
        self._display_nonempty = True
        self._last_space = value[-1] == " "

    def text(self, chunk: str) -> None:
        if not isinstance(chunk, str):
            _fail(f"{self.inline_context}.value", "must be a string")
        if chunk and (not chunk.isascii() or not chunk.isprintable()):
            _fail(f"{self.inline_context}.value", "must contain only printable ASCII characters")
        self._prose(chunk)
        self._inline_nonempty |= bool(chunk)

    def citation(self, number: int) -> None:
        if isinstance(number, bool) or not isinstance(number, int):
            _fail(f"{self.inline_context}.number", "must be an integer")
        if not 1 <= number <= _MAX_UINT:
            _fail(f"{self.inline_context}.number", f"must be between 1 and {_MAX_UINT}")
        if number not in self.pending_citations:
            check_limit(self.limits, "max_pending_citations", len(self.pending_citations) + 1)
            self.pending_citations.add(number)
        self._prose(f"[{number}]")
        self._inline_nonempty = True

    def end_inline(self) -> None:
        if self._inline_kind == 1 and not self._inline_nonempty:
            _fail(f"{self.inline_context}.value", "must not be empty")

    def ascii(self, chunk: str) -> None:
        if not isinstance(chunk, str):
            _fail(f"{self.context}.value", "must be a string")
        if _ASCII_INVALID.search(chunk) is not None:
            _fail(f"{self.context}.value", "must contain only printable ASCII characters and LF")
        first_newline = chunk.find("\n")
        if first_newline == -1:
            self._ascii_line_length += len(chunk)
            too_wide = self._ascii_line_length > 98
        else:
            too_wide = self._ascii_line_length + first_newline > 98 or _ASCII_WIDE.search(chunk) is not None
            self._ascii_line_length = len(chunk) - chunk.rfind("\n") - 1
        if too_wide:
            _fail(self.context, "each ASCII block line must fit within 98 columns")

    def reference(self, number: int, url: str) -> None:
        _positive_uint(number, f"{self.context}.number")
        if number <= self._previous_reference:
            _fail(f"{self.context}.number", "reference numbers must be strictly increasing")
        url = _printable(url, f"{self.context}.url", empty=False)
        if " " in url or _SCHEME_PATTERN.match(url) is None:
            _fail(f"{self.context}.url", "must have an absolute URL scheme and contain no spaces")
        if len(url) > 98:
            _fail(f"{self.context}.url", "unbreakable URL must fit within 98 columns")
        self._previous_reference = number
        self.pending_citations.discard(number)
        if not self.pending_citations:
            self.pending_citations.clear()

    def end_record(self) -> None:
        if self._kind in (2, 3):
            if not self._display_nonempty:
                _fail(self.context, "display text must not be empty")
            if self._last_space:
                _fail(self.context, "display text must have single internal spaces and no surrounding spaces")

    def finish(self) -> None:
        if self.pending_citations:
            count = len(self.pending_citations)
            numbers = sorted(self.pending_citations) if count <= 32 else nsmallest(32, self.pending_citations)
            missing = ", ".join(str(number) for number in numbers)
            if count > 32:
                missing += f", ... ({count} total)"
            _fail("document", f"unresolved citation numbers: {missing}")


def validate(document: Document, *, limits: Limits | None = None) -> None:
    if not isinstance(document, Document):
        _fail("document", "must be a Document instance")
    validate_metadata(document.metadata)
    if not isinstance(document.records, tuple):
        _fail("document.records", "must be a tuple")
    state = ValidationState(limits)
    for record in document.records:
        if isinstance(record, Heading):
            state.begin_record(1)
            state.heading(record.depth, record.title)
        elif isinstance(record, (Paragraph, BulletItem)):
            state.begin_record(2 if isinstance(record, Paragraph) else 3)
            if not isinstance(record.inlines, tuple):
                _fail(f"{state.context}.inlines", "must be a tuple")
            for inline in record.inlines:
                if isinstance(inline, Text):
                    state.begin_inline(1)
                    if not isinstance(inline.value, str):
                        state.text(inline.value)
                    for offset in range(0, len(inline.value), _CHUNK_SIZE):
                        state.text(inline.value[offset:offset + _CHUNK_SIZE])
                elif isinstance(inline, Citation):
                    state.begin_inline(2)
                    state.citation(inline.number)
                else:
                    state.begin_inline(0)
                state.end_inline()
        elif isinstance(record, AsciiBlock):
            state.begin_record(4)
            if not isinstance(record.value, str):
                state.ascii(record.value)
            for offset in range(0, len(record.value), _CHUNK_SIZE):
                state.ascii(record.value[offset:offset + _CHUNK_SIZE])
        elif isinstance(record, Reference):
            state.begin_record(5)
            state.reference(record.number, record.url)
        else:
            state.begin_record(0)
        state.end_record()
    state.finish()
