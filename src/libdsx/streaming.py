from __future__ import annotations

import os
from typing import BinaryIO, Iterator

from .compression import iter_inflate
from .errors import FormatError, ValidationError
from .limits import Limits, check_limit, check_limits
from .model import AsciiBlock, BulletItem, Citation, Header, Heading, Paragraph, Record, Reference, Text
from .validation import ValidationState


CHUNK_SIZE = 65536
Source = str | os.PathLike[str] | BinaryIO


class _BufferStream:
    __slots__ = ("data", "position")

    def __init__(self, data: memoryview):
        self.data = data
        self.position = 0

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        position = offset if whence == 0 else self.position + offset if whence == 1 else len(self.data) + offset
        if position < 0:
            raise ValueError("negative seek position")
        self.position = position
        return position

    def read(self, size: int = -1) -> bytes:
        end = len(self.data) if size < 0 else min(len(self.data), self.position + size)
        value = self.data[self.position:end].tobytes()
        self.position = end
        return value


class _TextBuilder:
    __slots__ = ("chunks", "pending", "pending_length")

    def __init__(self):
        self.chunks: list[str] = []
        self.pending: list[str] = []
        self.pending_length = 0

    def _flush(self) -> None:
        if self.pending:
            self.chunks.append("".join(self.pending))
            self.pending.clear()
            self.pending_length = 0

    def write(self, value: str) -> None:
        if len(value) >= 65536:
            self._flush()
            self.chunks.append(value)
        else:
            self.pending.append(value)
            self.pending_length += len(value)
            if self.pending_length >= 65536 or len(self.pending) >= 1024:
                self._flush()

    def getvalue(self) -> str:
        self._flush()
        return "".join(self.chunks)

    def close(self) -> None:
        self.pending.clear()
        self.chunks.clear()


class _DecodedReader:
    __slots__ = ("chunks", "buffer", "offset", "position", "length")

    def __init__(self, chunks: Iterator[bytes], length: int):
        self.chunks = chunks
        self.buffer = memoryview(b"")
        self.offset = 0
        self.position = 0
        self.length = length

    def _fill(self) -> None:
        while self.offset == len(self.buffer):
            try:
                self.buffer = memoryview(next(self.chunks))
            except StopIteration:
                raise FormatError(f"content: unexpected end at byte {self.position}") from None
            self.offset = 0

    def byte(self, end: int) -> int:
        if self.position >= end:
            raise FormatError(f"content: truncated value at byte {self.position}")
        self._fill()
        value = self.buffer[self.offset]
        self.offset += 1
        self.position += 1
        return value

    def uint(self, end: int) -> int:
        start = self.position
        value = 0
        for index in range(5):
            byte = self.byte(end)
            if index == 4 and byte & 0xF0:
                raise FormatError(f"content: integer exceeds uint32 at byte {start}")
            value |= (byte & 127) << (index * 7)
            if not byte & 128:
                if index and byte == 0:
                    raise FormatError(f"content: noncanonical integer at byte {start}")
                return value
        raise FormatError(f"content: invalid integer at byte {start}")

    def tlv(self, end: int) -> tuple[int, int, int]:
        identifier = self.byte(end)
        length = self.uint(end)
        boundary = self.position + length
        if boundary > end:
            raise FormatError(f"content: truncated value at byte {self.position}")
        return identifier, length, boundary

    def text(self, end: int) -> Iterator[str]:
        while self.position < end:
            self._fill()
            size = min(end - self.position, len(self.buffer) - self.offset)
            part = self.buffer[self.offset:self.offset + size]
            self.offset += size
            self.position += size
            try:
                yield str(part, "ascii")
            except UnicodeDecodeError as error:
                raise FormatError(f"content: text is not ASCII at byte {self.position - size}") from error

    def short_text(self, end: int) -> str:
        if end - self.position > 98:
            raise ValidationError("content: heading or reference text must fit within 98 columns")
        return "".join(self.text(end))

    def finish(self) -> None:
        if self.position != self.length or self.offset != len(self.buffer):
            raise FormatError("content: unexpected residual bytes")
        for chunk in self.chunks:
            if chunk:
                raise FormatError("content: decoded length exceeds its declaration")


def _events(stream: BinaryIO, header: Header, limits: Limits | None) -> Iterator[tuple]:
    reader = _DecodedReader(iter_inflate(stream, header.content_compressed_length, header.content_length, "content", chunk_size=CHUNK_SIZE), header.content_length)
    state = ValidationState(limits)
    while reader.position < reader.length:
        kind, length, end = reader.tlv(reader.length)
        if kind not in (1, 2, 3, 4, 5):
            raise FormatError(f"content: unknown content identifier {kind}")
        check_limit(limits, "max_record_bytes", length)
        state.begin_record(kind)
        yield ("record", kind, length)
        if kind == 1:
            depth = reader.uint(end)
            title = reader.short_text(end)
            state.heading(depth, title)
            yield ("heading", depth, title)
        elif kind in (2, 3):
            while reader.position < end:
                inline_kind, inline_length, inline_end = reader.tlv(end)
                if inline_kind not in (1, 2):
                    raise FormatError(f"content: unknown inline identifier {inline_kind}")
                check_limit(limits, "max_inline_bytes", inline_length)
                state.begin_inline(inline_kind)
                yield ("inline", inline_kind, inline_length)
                if inline_kind == 1:
                    for text in reader.text(inline_end):
                        state.text(text)
                        yield ("text", text)
                else:
                    number = reader.uint(inline_end)
                    state.citation(number)
                    yield ("citation", number)
                if reader.position != inline_end:
                    raise FormatError("content: unexpected residual inline bytes")
                state.end_inline()
                yield ("end_inline",)
        elif kind == 4:
            for text in reader.text(end):
                state.ascii(text)
                yield ("text", text)
        else:
            number = reader.uint(end)
            url = reader.short_text(end)
            state.reference(number, url)
            yield ("reference", number, url)
        state.end_record()
        yield ("end_record",)
    reader.finish()
    trailing = stream.read(1)
    if not isinstance(trailing, bytes):
        raise TypeError("expected a binary stream returning bytes")
    if trailing:
        raise FormatError("file: unexpected trailing bytes")
    state.finish()


class DocumentReader:
    def __init__(self, source: Source, *, limits: Limits | None = None) -> None:
        from .codec import _check_header_limits, _decode_metadata, _inflate, _read_exact, _stream_header

        check_limits(limits)
        self._owned = isinstance(source, (str, os.PathLike))
        self._stream = open(source, "rb") if self._owned else source
        self._closed = False
        self._verified = False
        self._mode = None
        self._records = None
        try:
            self.header = _stream_header(self._stream)
            _check_header_limits(self.header, limits)
            data = _read_exact(self._stream, self.header.metadata_compressed_length)
            self.metadata = _decode_metadata(_inflate(data, self.header.metadata_length, "metadata"))
            self._iterator = _events(self._stream, self.header, limits)
        except BaseException:
            self.close()
            raise

    @property
    def content_verified(self) -> bool:
        return self._verified

    @property
    def closed(self) -> bool:
        return self._closed

    def __enter__(self) -> DocumentReader:
        if self._closed:
            raise ValueError("document reader is closed")
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            iterator = getattr(self, "_iterator", None)
            if iterator is not None:
                iterator.close()
            if self._records is not None and not self._records.gi_running:
                self._records.close()
            if self._owned:
                self._stream.close()

    def _next_event(self) -> tuple:
        if self._verified:
            raise StopIteration
        if self._closed:
            raise ValueError("document reader is closed")
        try:
            return next(self._iterator)
        except StopIteration:
            self._verified = True
            raise
        except BaseException:
            self.close()
            raise

    def _iter_events(self) -> Iterator[tuple]:
        if self._mode is not None:
            raise ValueError("document reader is already being consumed")
        self._mode = "events"
        while True:
            try:
                event = self._next_event()
            except StopIteration:
                return
            yield event

    def __iter__(self) -> DocumentReader:
        return self

    def __next__(self) -> Record:
        if self._closed and not self._verified:
            raise ValueError("document reader is closed")
        if self._mode == "events":
            raise ValueError("document reader is already being consumed as events")
        if self._records is None:
            self._mode = "records"
            self._records = self._materialize()
        return next(self._records)

    def _materialize(self) -> Iterator[Record]:
        text = None
        inlines = None
        record = None
        kind = 0
        inline_kind = 0
        try:
            while True:
                try:
                    event = self._next_event()
                except StopIteration:
                    return
                tag = event[0]
                if tag == "record":
                    kind = event[1]
                    if kind in (2, 3):
                        inlines = []
                    elif kind == 4:
                        text = _TextBuilder() if event[2] else None
                elif tag == "inline":
                    inline_kind = event[1]
                    if inline_kind == 1:
                        text = _TextBuilder()
                elif tag == "text":
                    text.write(event[1])
                elif tag == "citation":
                    inlines.append(Citation(event[1]))
                elif tag == "end_inline":
                    if inline_kind == 1:
                        inlines.append(Text(text.getvalue()))
                        text.close()
                        text = None
                elif tag == "heading":
                    record = Heading(event[1], event[2])
                elif tag == "reference":
                    record = Reference(event[1], event[2])
                elif tag == "end_record":
                    if kind in (2, 3):
                        record = (Paragraph if kind == 2 else BulletItem)(tuple(inlines))
                        inlines = None
                    elif kind == 4:
                        record = AsciiBlock(text.getvalue() if text is not None else "")
                        if text is not None:
                            text.close()
                        text = None
                    yield record
                    record = None
        finally:
            if text is not None:
                text.close()

    def finish(self) -> None:
        while True:
            try:
                self._next_event()
            except StopIteration:
                return


def open_document(source: Source, *, limits: Limits | None = None) -> DocumentReader:
    return DocumentReader(source, limits=limits)


def iter_records(source: Source, *, limits: Limits | None = None) -> Iterator[Record]:
    with open_document(source, limits=limits) as reader:
        yield from reader
