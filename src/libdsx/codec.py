from __future__ import annotations

import os
import re
import struct
import zlib
from contextlib import contextmanager
from datetime import date
from typing import BinaryIO, Iterator

from .compression import check_empty_stream
from .errors import FormatError, ValidationError
from .limits import Limits, check_limit, check_limits
from .model import AsciiBlock, BulletItem, Document, Header, Heading, Inline, Metadata, MetadataInspection, Paragraph, Text
from .validation import validate as _validate_document
from .validation import validate_metadata


SIGNATURE = b"\x89DSX\r\n\x1a\n"
MAX_METADATA_STORED = 131072
MAX_METADATA_DECODED = 65536
MAX_CONTENT_STORED = 68157440
MAX_CONTENT_DECODED = 67108864
_HEADER = struct.Struct("<8sBBHIIIII")
_REVISION = re.compile(r"[1-9][0-9]{0,9}\Z")
_DATE = re.compile(r"[0-9]{2}\.[0-9]{2}\.[0-9]{4}\Z")
Source = str | os.PathLike[str] | BinaryIO


class _Cursor:
    __slots__ = ("data", "position", "_context", "_offset", "_identifier")

    def __init__(self, data: bytes | memoryview, context, offset: int = 0, identifier: int = 0):
        self.data = memoryview(data)
        self.position = 0
        self._context = context
        self._offset = offset
        self._identifier = identifier

    @property
    def context(self) -> str:
        if isinstance(self._context, str):
            return self._context
        return f"{self._context.context}, type {self._identifier} at byte {self._offset}"

    def __bool__(self) -> bool:
        return self.position < len(self.data)

    def take(self, length: int) -> memoryview:
        end = self.position + length
        if end > len(self.data):
            raise FormatError(f"{self.context}: truncated value at byte {self.position}")
        result = self.data[self.position:end]
        self.position = end
        return result

    def byte(self) -> int:
        if self.position >= len(self.data):
            raise FormatError(f"{self.context}: truncated value at byte {self.position}")
        result = self.data[self.position]
        self.position += 1
        return result

    def uint(self) -> int:
        start = self.position
        value = 0
        for index in range(5):
            byte = self.byte()
            if index == 4 and byte & 0xF0:
                raise FormatError(f"{self.context}: integer exceeds uint32 at byte {start}")
            value |= (byte & 0x7F) << (7 * index)
            if not byte & 0x80:
                if index and byte == 0:
                    raise FormatError(f"{self.context}: noncanonical integer at byte {start}")
                return value
        raise FormatError(f"{self.context}: invalid integer at byte {start}")

    def text(self, length: int | None = None) -> str:
        if length is None:
            length = len(self.data) - self.position
        try:
            return str(self.take(length), "ascii")
        except UnicodeDecodeError as error:
            raise FormatError(f"{self.context}: text is not ASCII") from error

    def string(self) -> str:
        return self.text(self.uint())

    def tlv(self) -> tuple[int, _Cursor]:
        offset = self.position
        identifier = self.byte()
        length = self.uint()
        return identifier, _Cursor(self.take(length), self, offset, identifier)

    def finish(self) -> None:
        if self:
            raise FormatError(f"{self.context}: unexpected residual bytes")


def _parse_header(data: bytes | memoryview, total_size: int) -> Header:
    if len(data) < 8 or data[:8] != SIGNATURE:
        raise FormatError("header: invalid DSX signature")
    if len(data) != _HEADER.size:
        raise FormatError("header: expected 32 bytes")
    fields = _HEADER.unpack(data)
    header = Header(*fields[1:])
    if zlib.crc32(data[:28]) != header.checksum:
        raise FormatError("header: CRC-32 mismatch")
    if header.layout_version != 1:
        raise FormatError("header: unsupported container layout version")
    if header.compression != 1:
        raise FormatError("header: unsupported compression identifier")
    if header.reserved != 0:
        raise FormatError("header: reserved field must be zero")
    limits = (
        ("stored metadata", header.metadata_compressed_length, 1, MAX_METADATA_STORED),
        ("uncompressed metadata", header.metadata_length, 1, MAX_METADATA_DECODED),
        ("stored content", header.content_compressed_length, 1, MAX_CONTENT_STORED),
        ("uncompressed content", header.content_length, 0, MAX_CONTENT_DECODED),
    )
    for name, length, minimum, maximum in limits:
        if not minimum <= length <= maximum:
            raise FormatError(f"header: {name} length must be between {minimum} and {maximum}")
    if total_size != 32 + header.metadata_compressed_length + header.content_compressed_length:
        raise FormatError("file size does not match declared section lengths")
    return header


def _inflate(data: bytes | memoryview, length: int, context: str) -> bytes:
    if length == 0:
        try:
            check_empty_stream(data)
        except FormatError as error:
            raise FormatError(f"{context}: {error}") from error
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(data, max(1, length))
    except zlib.error as error:
        raise FormatError(f"{context}: invalid zlib stream: {error}") from error
    if len(decoded) != length:
        raise FormatError(f"{context}: uncompressed length mismatch")
    if not decoder.eof:
        raise FormatError(f"{context}: incomplete stream or decoded length exceeds its declaration")
    if decoder.unused_data or decoder.unconsumed_tail:
        raise FormatError(f"{context}: zlib stream does not end at its section boundary")
    return decoded


def _decode_metadata(data: bytes) -> Metadata:
    section = _Cursor(data, "metadata")
    values: list[object] = []
    for expected in range(1, 6):
        identifier, value = section.tlv()
        if identifier != expected:
            raise FormatError(f"metadata: expected required field {expected}, found {identifier}")
        if identifier == 3:
            count = value.uint()
            if count > len(value.data) - value.position:
                raise FormatError("metadata.authors: count exceeds available string data")
            values.append(tuple(value.string() for _ in range(count)))
        else:
            values.append(value.text())
        value.finish()
    version, title, authors, revision_text, date_text = values
    if _REVISION.fullmatch(revision_text) is None:
        raise FormatError("metadata.revision: expected a canonical positive uint32 decimal")
    revision = int(revision_text)
    if revision > 4294967295:
        raise FormatError("metadata.revision: exceeds uint32")
    if _DATE.fullmatch(date_text) is None:
        raise FormatError("metadata.date: expected MM.DD.YYYY")
    try:
        document_date = date(int(date_text[6:]), int(date_text[:2]), int(date_text[3:5]))
    except ValueError as error:
        raise FormatError("metadata.date: invalid Gregorian date") from error
    additional = {}
    previous_key = ""
    while section:
        identifier, value = section.tlv()
        if identifier != 128:
            raise FormatError(f"metadata: unknown or repeated field identifier {identifier}")
        key = value.string()
        if key <= previous_key:
            raise FormatError("metadata: additional keys must be unique and in ascending ASCII order")
        additional[key] = value.text()
        previous_key = key
    metadata = Metadata(title, authors, revision, document_date, additional, version)
    validate_metadata(metadata)
    return metadata


def _bytes_view(data: bytes | bytearray | memoryview) -> memoryview:
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("expected bytes, bytearray, or memoryview")
    view = memoryview(data)
    if not view.c_contiguous:
        raise TypeError("expected a contiguous byte buffer")
    return view.cast("B")


def loads(data: bytes | bytearray | memoryview, *, limits: Limits | None = None) -> Document:
    from .streaming import _BufferStream

    return load(_BufferStream(_bytes_view(data)), limits=limits)


def _check_header_limits(header: Header, limits: Limits | None) -> None:
    check_limits(limits)
    for name, value in (
        ("max_metadata_stored", header.metadata_compressed_length),
        ("max_metadata_decoded", header.metadata_length),
        ("max_content_stored", header.content_compressed_length),
        ("max_content_decoded", header.content_length),
    ):
        check_limit(limits, name, value)


def read_metadata_bytes(data: bytes | bytearray | memoryview, *, limits: Limits | None = None) -> MetadataInspection:
    view = _bytes_view(data)
    header = _parse_header(view[:32], len(view))
    _check_header_limits(header, limits)
    boundary = 32 + header.metadata_compressed_length
    metadata = _decode_metadata(_inflate(view[32:boundary], header.metadata_length, "metadata"))
    return MetadataInspection(metadata, header)


@contextmanager
def _reader(source: Source) -> Iterator[BinaryIO]:
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as stream:
            yield stream
    else:
        yield source


def _read_exact(stream: BinaryIO, length: int) -> bytes:
    if length == 0:
        return b""
    chunk = stream.read(length)
    if not isinstance(chunk, bytes):
        raise TypeError("expected a binary stream returning bytes")
    if not chunk or len(chunk) > length:
        raise FormatError("file: unexpected end of stream or invalid read length")
    if len(chunk) == length:
        return chunk
    buffer = bytearray(length)
    buffer[:len(chunk)] = chunk
    position = len(chunk)
    while position < length:
        chunk = stream.read(length - position)
        if not isinstance(chunk, bytes):
            raise TypeError("expected a binary stream returning bytes")
        if not chunk or len(chunk) > length - position:
            raise FormatError("file: unexpected end of stream or invalid read length")
        buffer[position:position + len(chunk)] = chunk
        position += len(chunk)
    return bytes(buffer)


def _stream_header(stream: BinaryIO) -> Header:
    start = stream.tell()
    end = stream.seek(0, os.SEEK_END)
    stream.seek(start)
    if end - start < 32:
        raise FormatError("header: expected 32 bytes")
    return _parse_header(_read_exact(stream, 32), end - start)


def load(source: Source, *, limits: Limits | None = None) -> Document:
    from .streaming import open_document

    with open_document(source, limits=limits) as reader:
        return Document(reader.metadata, tuple(reader))


def read_metadata(source: Source, *, limits: Limits | None = None) -> MetadataInspection:
    with _reader(source) as stream:
        header = _stream_header(stream)
        _check_header_limits(header, limits)
        data = _read_exact(stream, header.metadata_compressed_length)
        metadata = _decode_metadata(_inflate(data, header.metadata_length, "metadata"))
        return MetadataInspection(metadata, header)


def _uint(value: int) -> bytes:
    result = bytearray()
    while value >= 128:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _usize(value: int) -> int:
    return max(1, (value.bit_length() + 6) // 7)


def _tlv_size(length: int) -> int:
    return 1 + _usize(length) + length


def _inline_size(inlines: tuple[Inline, ...]) -> int:
    total = 0
    pending_text = 0
    for inline in inlines:
        if isinstance(inline, Text):
            pending_text += len(inline.value)
        else:
            if pending_text:
                total += _tlv_size(pending_text)
                pending_text = 0
            total += _tlv_size(_usize(inline.number))
    if pending_text:
        total += _tlv_size(pending_text)
    return total


def _check_sizes(document: Document, limits: Limits | None = None) -> None:
    metadata = document.metadata
    authors_length = _usize(len(metadata.authors)) + sum(_usize(len(author)) + len(author) for author in metadata.authors)
    metadata_length = sum(_tlv_size(length) for length in (3, len(metadata.title), authors_length, len(str(metadata.revision)), 10))
    for key, value in metadata.additional.items():
        metadata_length += _tlv_size(_usize(len(key)) + len(key) + len(value))
    if metadata_length > MAX_METADATA_DECODED:
        raise ValidationError(f"metadata: uncompressed section exceeds {MAX_METADATA_DECODED} bytes")
    check_limit(limits, "max_metadata_decoded", metadata_length)
    content_length = 0
    for record in document.records:
        if isinstance(record, Heading):
            length = _usize(record.depth) + len(record.title)
        elif isinstance(record, (Paragraph, BulletItem)):
            length = _inline_size(record.inlines)
            if limits is not None and limits.max_inline_bytes is not None:
                pending = 0
                for inline in record.inlines:
                    if isinstance(inline, Text):
                        pending += len(inline.value)
                        check_limit(limits, "max_inline_bytes", pending)
                    else:
                        pending = 0
                        check_limit(limits, "max_inline_bytes", _usize(inline.number))
        elif isinstance(record, AsciiBlock):
            length = len(record.value)
        else:
            length = _usize(record.number) + len(record.url)
        check_limit(limits, "max_record_bytes", length)
        content_length += _tlv_size(length)
        if content_length > MAX_CONTENT_DECODED:
            raise ValidationError(f"content: uncompressed section exceeds {MAX_CONTENT_DECODED} bytes")
        check_limit(limits, "max_content_decoded", content_length)


def validate(document: Document, *, limits: Limits | None = None) -> None:
    check_limits(limits)
    _validate_document(document, limits=limits)
    _check_sizes(document, limits)


def validate_file(source: Source, *, limits: Limits | None = None) -> None:
    from .streaming import open_document

    with open_document(source, limits=limits) as reader:
        reader.finish()


def _tlv(identifier: int, value: bytes) -> bytes:
    return bytes((identifier,)) + _uint(len(value)) + value


def _string(value: str) -> bytes:
    encoded = value.encode("ascii")
    return _uint(len(encoded)) + encoded


def _encode_metadata(metadata: Metadata) -> bytes:
    document_date = metadata.date
    values = (
        metadata.dsx_version.encode("ascii"),
        metadata.title.encode("ascii"),
        _uint(len(metadata.authors)) + b"".join(_string(author) for author in metadata.authors),
        str(metadata.revision).encode("ascii"),
        f"{document_date.month:02}.{document_date.day:02}.{document_date.year:04}".encode("ascii"),
    )
    required = b"".join(_tlv(identifier, value) for identifier, value in enumerate(values, 1))
    additional = b"".join(_tlv(128, _string(key) + metadata.additional[key].encode("ascii")) for key in sorted(metadata.additional))
    return required + additional


def dumps(document: Document, *, compression_level: int = -1, limits: Limits | None = None, spool_limit: int = 1048576) -> bytes:
    from .writing import dumps as encode

    return encode(document, compression_level=compression_level, limits=limits, spool_limit=spool_limit)


def _write_all(stream: BinaryIO, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = stream.write(remaining)
        if written is None or written <= 0 or written > len(remaining):
            raise OSError("binary stream did not accept the complete document")
        remaining = remaining[written:]


def dump(document: Document, target: Source, *, compression_level: int = -1, limits: Limits | None = None, spool_limit: int = 1048576) -> None:
    from .writing import dump as encode

    encode(document, target, compression_level=compression_level, limits=limits, spool_limit=spool_limit)
