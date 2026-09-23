from __future__ import annotations

import io
import os
import struct
import zlib
from contextlib import contextmanager
from tempfile import SpooledTemporaryFile
from typing import BinaryIO, Iterator

from .errors import ValidationError
from .limits import Limits, check_limit
from .model import AsciiBlock, BulletItem, Document, Heading, Inline, Paragraph, Record, Text


_CHUNK_SIZE = 65536


def _text_chunks(value: str) -> Iterator[bytes]:
    for offset in range(0, len(value), _CHUNK_SIZE):
        yield value[offset:offset + _CHUNK_SIZE].encode("ascii")


def _inline_chunks(inlines: tuple[Inline, ...], codec, limits: Limits | None) -> Iterator[bytes]:
    index = 0
    while index < len(inlines):
        inline = inlines[index]
        if isinstance(inline, Text):
            end = index + 1
            length = len(inline.value)
            while end < len(inlines) and isinstance(inlines[end], Text):
                length += len(inlines[end].value)
                end += 1
            check_limit(limits, "max_inline_bytes", length)
            yield b"\x01" + codec._uint(length)
            while index < end:
                yield from _text_chunks(inlines[index].value)
                index += 1
        else:
            value = codec._uint(inline.number)
            check_limit(limits, "max_inline_bytes", len(value))
            yield b"\x02" + codec._uint(len(value)) + value
            index += 1


def _content_chunks(records: tuple[Record, ...], codec, limits: Limits | None) -> Iterator[bytes]:
    for record in records:
        if isinstance(record, Heading):
            number = codec._uint(record.depth)
            length = len(number) + len(record.title)
            check_limit(limits, "max_record_bytes", length)
            yield b"\x01" + codec._uint(length) + number
            yield from _text_chunks(record.title)
        elif isinstance(record, (Paragraph, BulletItem)):
            length = codec._inline_size(record.inlines)
            check_limit(limits, "max_record_bytes", length)
            identifier = b"\x02" if isinstance(record, Paragraph) else b"\x03"
            yield identifier + codec._uint(length)
            yield from _inline_chunks(record.inlines, codec, limits)
        elif isinstance(record, AsciiBlock):
            length = len(record.value)
            check_limit(limits, "max_record_bytes", length)
            if length:
                yield b"\x04" + codec._uint(length)
                yield from _text_chunks(record.value)
            else:
                yield b"\x04\x00"
        else:
            number = codec._uint(record.number)
            length = len(number) + len(record.url)
            check_limit(limits, "max_record_bytes", length)
            yield b"\x05" + codec._uint(length) + number
            yield from _text_chunks(record.url)


def _compress_section(
    chunks: Iterator[bytes],
    spool: BinaryIO,
    compression_level: int,
    section: str,
    decoded_maximum: int,
    stored_maximum: int,
    limits: Limits | None,
    codec,
) -> tuple[int, int]:
    compressor = zlib.compressobj(compression_level)
    pending = bytearray()
    decoded_length = 0
    stored_length = 0
    decoded_limit_name = f"max_{section}_decoded"
    stored_limit_name = f"max_{section}_stored"

    def store(data: bytes) -> None:
        nonlocal stored_length
        stored_length += len(data)
        if stored_length > stored_maximum:
            raise ValidationError(f"{section}: compressed section exceeds {stored_maximum} bytes")
        check_limit(limits, stored_limit_name, stored_length)
        codec._write_all(spool, data)

    for chunk in chunks:
        decoded_length += len(chunk)
        if decoded_length > decoded_maximum:
            raise ValidationError(f"{section}: uncompressed section exceeds {decoded_maximum} bytes")
        check_limit(limits, decoded_limit_name, decoded_length)
        if len(pending) + len(chunk) <= _CHUNK_SIZE:
            pending.extend(chunk)
            if len(pending) == _CHUNK_SIZE:
                store(compressor.compress(pending))
                pending.clear()
            continue
        view = memoryview(chunk)
        offset = 0
        while offset < len(view):
            end = min(len(view), offset + _CHUNK_SIZE - len(pending))
            pending.extend(view[offset:end])
            offset = end
            if len(pending) == _CHUNK_SIZE:
                store(compressor.compress(pending))
                pending.clear()
    if pending:
        store(compressor.compress(pending))
    store(compressor.flush())
    return stored_length, decoded_length


@contextmanager
def _encoded_document(
    document: Document,
    compression_level: int,
    limits: Limits | None,
    spool_limit: int,
) -> Iterator[tuple[bytes, BinaryIO]]:
    from . import codec

    codec.validate(document, limits=limits)
    if isinstance(compression_level, bool) or not isinstance(compression_level, int) or not -1 <= compression_level <= 9:
        raise ValueError("compression_level must be an integer between -1 and 9")
    if isinstance(spool_limit, bool) or not isinstance(spool_limit, int) or spool_limit <= 0:
        raise ValueError("spool_limit must be a positive integer")
    with SpooledTemporaryFile(max_size=spool_limit, mode="w+b") as spool:
        stored_metadata, metadata_length = _compress_section(
            iter((codec._encode_metadata(document.metadata),)), spool, compression_level,
            "metadata", codec.MAX_METADATA_DECODED, codec.MAX_METADATA_STORED, limits, codec,
        )
        stored_content, content_length = _compress_section(
            _content_chunks(document.records, codec, limits), spool, compression_level,
            "content", codec.MAX_CONTENT_DECODED, codec.MAX_CONTENT_STORED, limits, codec,
        )
        prefix = codec._HEADER.pack(
            codec.SIGNATURE, 1, 1, 0, stored_metadata, metadata_length,
            stored_content, content_length, 0,
        )[:28]
        header = prefix + struct.pack("<I", zlib.crc32(prefix))
        spool.seek(0)
        yield header, spool


def dump(
    document: Document,
    target: str | os.PathLike[str] | BinaryIO,
    *,
    compression_level: int = -1,
    limits: Limits | None = None,
    spool_limit: int = 1048576,
) -> None:
    from . import codec

    def write(stream: BinaryIO, header: bytes, spool: BinaryIO) -> None:
        codec._write_all(stream, header)
        while chunk := spool.read(_CHUNK_SIZE):
            codec._write_all(stream, chunk)

    with _encoded_document(document, compression_level, limits, spool_limit) as (header, spool):
        if isinstance(target, (str, os.PathLike)):
            with open(target, "wb") as stream:
                write(stream, header, spool)
        else:
            write(target, header, spool)


def dumps(
    document: Document,
    *,
    compression_level: int = -1,
    limits: Limits | None = None,
    spool_limit: int = 1048576,
) -> bytes:
    target = io.BytesIO()
    dump(document, target, compression_level=compression_level, limits=limits, spool_limit=spool_limit)
    return target.getvalue()
