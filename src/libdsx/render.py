from __future__ import annotations

import os
from contextlib import closing
from io import StringIO
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Iterator, TextIO

from .errors import ValidationError
from .limits import Limits, check_limit, check_limits
from .model import AsciiBlock, BulletItem, Document, Heading, Paragraph, Text
from .validation import _heading_text


_CHUNK_SIZE = 65536
Source = Document | str | os.PathLike[str] | BinaryIO


def _center(text: str) -> str:
    return " " * ((98 - len(text)) // 2) + text


class _Wrap:
    __slots__ = ("width", "prefix", "continuation", "line", "word")

    def __init__(self, bullet: bool = False):
        self.width = 96 if bullet else 98
        self.prefix = "* " if bullet else ""
        self.continuation = "  " if bullet else ""
        self.line = ""
        self.word = ""

    def _accept_word(self) -> str | None:
        result = None
        if self.line and len(self.line) + 1 + len(self.word) > self.width:
            result = self.prefix + self.line + "\n"
            self.prefix = self.continuation
            self.line = self.word
        else:
            self.line += (" " if self.line else "") + self.word
        self.word = ""
        return result

    def feed(self, text: str) -> Iterator[str]:
        start = 0
        while True:
            end = text.find(" ", start)
            if end < 0:
                self.word += text[start:]
                return
            self.word += text[start:end]
            line = self._accept_word()
            if line is not None:
                yield line
            start = end + 1

    def finish(self) -> Iterator[str]:
        if self.word:
            line = self._accept_word()
            if line is not None:
                yield line
        if self.line:
            yield self.prefix + self.line + "\n"


def _text_events(value: str) -> Iterator[tuple]:
    for offset in range(0, len(value), _CHUNK_SIZE):
        yield ("text", value[offset:offset + _CHUNK_SIZE])


def _document_events(document: Document) -> Iterator[tuple]:
    for record in document.records:
        if isinstance(record, Heading):
            yield ("record", 1)
            yield ("heading", record.depth, record.title)
        elif isinstance(record, (Paragraph, BulletItem)):
            yield ("record", 2 if isinstance(record, Paragraph) else 3)
            for inline in record.inlines:
                if isinstance(inline, Text):
                    yield from _text_events(inline.value)
                else:
                    yield ("citation", inline.number)
        elif isinstance(record, AsciiBlock):
            yield ("record", 4)
            yield from _text_events(record.value)
        else:
            yield ("record", 5)
            yield ("reference", record.number, record.url)
        yield ("end_record",)


def _render_events(metadata, events: Iterator[tuple]) -> Iterator[str]:
    document_date = metadata.date
    date_text = f"{document_date.month:02}.{document_date.day:02}.{document_date.year:04}"
    yield "\n\n\n" + _center(metadata.title) + "\n"
    for author in metadata.authors:
        yield _center(author) + "\n"
    yield "\n" + _center(f"Rev. {metadata.revision}, {date_text}") + "\n\n\n\n"
    counters = [0] * 8
    previous = None
    kind = 0
    last_ascii_character = ""
    wrapper = None
    for event in events:
        tag = event[0]
        if tag == "record":
            kind = event[1]
            if previous is not None and kind != 1 and (kind != 5 or previous != 5):
                yield "\n"
            if kind in (2, 3, 5):
                wrapper = _Wrap(kind == 3)
            elif kind == 4:
                last_ascii_character = ""
        elif tag == "heading":
            depth, title = event[1:]
            if previous is not None:
                yield "\n\n" if depth == 1 else "\n"
            title = _heading_text(Heading(depth, title), counters)
            yield title + "\n"
            if depth == 1:
                yield "=" * len(title) + "\n"
        elif tag == "text":
            text = event[1]
            if kind == 4:
                if text:
                    last_ascii_character = text[-1]
                    yield text
            else:
                yield from wrapper.feed(text)
        elif tag == "citation":
            yield from wrapper.feed(f"[{event[1]}]")
        elif tag == "reference":
            yield from wrapper.feed(f"[{event[1]}] {event[2]}")
        elif tag == "end_record":
            if kind in (2, 3, 5):
                yield from wrapper.finish()
                wrapper = None
            elif kind == 4 and last_ascii_character and last_ascii_character != "\n":
                yield "\n"
            previous = kind


def iter_render(source: Source, *, limits: Limits | None = None) -> Iterator[str]:
    from .codec import validate
    from .streaming import open_document

    check_limits(limits)
    total = 0

    def checked(chunks: Iterator[str]) -> Iterator[str]:
        nonlocal total
        for chunk in chunks:
            total += len(chunk)
            check_limit(limits, "max_rendered_chars", total)
            yield chunk

    if isinstance(source, Document):
        validate(source, limits=limits)
        yield from checked(_render_events(source.metadata, _document_events(source)))
    else:
        with open_document(source, limits=limits) as reader:
            yield from checked(_render_events(reader.metadata, reader._iter_events()))


def render(document: Document, *, limits: Limits | None = None) -> str:
    if not isinstance(document, Document):
        raise ValidationError("document: must be a Document instance")
    with StringIO() as output:
        for chunk in iter_render(document, limits=limits):
            output.write(chunk)
        return output.getvalue()


def _write_text(stream: TextIO, value: str) -> None:
    offset = 0
    request = value
    while offset < len(value):
        written = stream.write(request)
        if written is None or written <= 0 or written > len(request):
            raise OSError("text stream did not accept the complete document")
        offset += written
        if offset < len(value):
            size = written if written < len(request) else len(request) * 2
            request = value[offset:offset + size]


def render_to(source: Source, target: str | os.PathLike[str] | TextIO, *, limits: Limits | None = None) -> None:
    if not isinstance(target, (str, os.PathLike)):
        with closing(iter_render(source, limits=limits)) as chunks:
            for chunk in chunks:
                _write_text(target, chunk)
        return
    path = os.path.abspath(target)
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", encoding="ascii", newline="\n", dir=os.path.dirname(path), prefix=".libdsx-", suffix=".tmp", delete=False) as stream:
            temporary_path = stream.name
            with closing(iter_render(source, limits=limits)) as chunks:
                for chunk in chunks:
                    _write_text(stream, chunk)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            os.unlink(temporary_path)
