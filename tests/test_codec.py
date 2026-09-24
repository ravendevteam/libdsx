from __future__ import annotations

import io
import struct
import zlib
from datetime import date

import pytest

import libdsx


SIGNATURE = b"\x89DSX\r\n\x1a\n"


def u(value: int) -> bytes:
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def tlv(kind: int, value: bytes) -> bytes:
    return bytes((kind,)) + u(len(value)) + value


def metadata() -> bytes:
    return b"".join(
        (
            tlv(1, b"1.1"),
            tlv(2, b"TEST DOCUMENT"),
            tlv(3, b"\x01\x03Ada"),
            tlv(4, b"1"),
            tlv(5, b"09.22.2026"),
        )
    )


def container(
    metadata_bytes: bytes | None = None,
    content_bytes: bytes = b"",
    *,
    metadata_stream: bytes | None = None,
    content_stream: bytes | None = None,
    level: int = 6,
) -> bytes:
    metadata_bytes = metadata() if metadata_bytes is None else metadata_bytes
    metadata_stream = zlib.compress(metadata_bytes, level) if metadata_stream is None else metadata_stream
    content_stream = zlib.compress(content_bytes, level) if content_stream is None else content_stream
    header = struct.pack(
        "<8sBBHIIII",
        SIGNATURE,
        1,
        1,
        0,
        len(metadata_stream),
        len(metadata_bytes),
        len(content_stream),
        len(content_bytes),
    )
    return header + struct.pack("<I", zlib.crc32(header)) + metadata_stream + content_stream


def change_header(data: bytes, offset: int, value: int, fmt: str = "<I") -> bytes:
    result = bytearray(data)
    struct.pack_into(fmt, result, offset, value)
    struct.pack_into("<I", result, 28, zlib.crc32(result[:28]))
    return bytes(result)


def sections(data: bytes) -> tuple[bytes, bytes]:
    stored_metadata_length = struct.unpack_from("<I", data, 12)[0]
    return (
        zlib.decompress(data[32 : 32 + stored_metadata_length]),
        zlib.decompress(data[32 + stored_metadata_length :]),
    )


def basic_document(records: tuple = ()) -> libdsx.Document:
    return libdsx.Document(libdsx.Metadata("TEST DOCUMENT", ("Ada",), 1, date(2026, 9, 22)), records)


def test_reads_independently_encoded_empty_document() -> None:
    assert libdsx.loads(container()) == basic_document()


def test_dsx_1_1_roundtrip_preserves_container_layout() -> None:
    document = basic_document((libdsx.AsciiBlock("A\n\n"),))
    encoded = libdsx.dumps(document)
    assert sections(encoded)[0].startswith(tlv(1, b"1.1"))
    assert libdsx.read_metadata_bytes(encoded).header.layout_version == 1
    assert libdsx.loads(encoded) == document
    assert libdsx.loads(encoded).metadata.dsx_version == "1.1"


def test_writer_header_and_crc() -> None:
    encoded = libdsx.dumps(basic_document())
    signature, layout, compression, reserved, stored_m, plain_m, stored_c, plain_c, checksum = struct.unpack(
        "<8sBBHIIIII", encoded[:32]
    )
    assert signature == SIGNATURE
    assert (layout, compression, reserved) == (1, 1, 0)
    assert len(encoded) == 32 + stored_m + stored_c
    assert (plain_m, plain_c) == (len(metadata()), 0)
    assert checksum == zlib.crc32(encoded[:28])
    assert zlib.crc32(b"123456789") == 0xCBF43926
    assert sections(encoded) == (metadata(), b"")


@pytest.mark.parametrize("level", range(10))
def test_reads_every_zlib_compression_level(level: int) -> None:
    assert libdsx.loads(container(level=level)) == basic_document()


@pytest.mark.parametrize("level", (-1, *range(10)))
def test_writer_compression_levels_have_identical_decoded_sections(level: int) -> None:
    document = basic_document((libdsx.Paragraph((libdsx.Text("A paragraph."),)),))
    encoded = libdsx.dumps(document, compression_level=level)
    assert libdsx.loads(encoded) == document
    assert sections(encoded) == (metadata(), b"\x02\x0e\x01\x0cA paragraph.")


@pytest.mark.parametrize("factory", (bytes, bytearray, memoryview))
def test_loads_supported_byte_containers(factory) -> None:
    assert libdsx.loads(factory(container())) == basic_document()


def test_empty_content_has_complete_zlib_stream() -> None:
    encoded = libdsx.dumps(basic_document())
    stored_m = struct.unpack_from("<I", encoded, 12)[0]
    decoder = zlib.decompressobj()
    assert decoder.decompress(encoded[32 + stored_m :]) == b""
    assert decoder.eof
    assert decoder.unused_data == b""


@pytest.mark.parametrize("length", (0, 1, 7, 8, 12, 28, 31))
def test_rejects_truncated_header(length: int) -> None:
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container()[:length])


@pytest.mark.parametrize("offset", range(8))
def test_rejects_each_incorrect_signature_byte(offset: int) -> None:
    data = bytearray(container())
    data[offset] ^= 1
    struct.pack_into("<I", data, 28, zlib.crc32(data[:28]))
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(bytes(data))


@pytest.mark.parametrize("offset", (8, 12, 16, 20, 24, 28, 31))
def test_rejects_incorrect_header_crc(offset: int) -> None:
    data = bytearray(container())
    data[offset] ^= 1
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(bytes(data))


@pytest.mark.parametrize(
    ("offset", "value", "fmt"),
    (
        (8, 0, "<B"),
        (8, 2, "<B"),
        (9, 0, "<B"),
        (9, 2, "<B"),
        (10, 1, "<H"),
        (10, 256, "<H"),
        (12, 0, "<I"),
        (16, 0, "<I"),
        (20, 0, "<I"),
        (12, 131073, "<I"),
        (16, 65537, "<I"),
        (20, 68157441, "<I"),
        (24, 67108865, "<I"),
    ),
)
def test_rejects_unsupported_header_values_and_limits(offset: int, value: int, fmt: str) -> None:
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(change_header(container(), offset, value, fmt))


@pytest.mark.parametrize("suffix", (b"\x00", b"trailing data", container()))
def test_rejects_bytes_after_declared_end(suffix: bytes) -> None:
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container() + suffix)


@pytest.mark.parametrize("missing", (1, 4, 8, 12))
def test_rejects_file_shorter_than_declared(missing: int) -> None:
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container()[:-missing])


@pytest.mark.parametrize("section", ("metadata", "content"))
@pytest.mark.parametrize("delta", (-1, 1))
def test_rejects_decoded_length_mismatch(section: str, delta: int) -> None:
    data = container(content_bytes=tlv(4, b"TEXT"))
    offset = 16 if section == "metadata" else 24
    length = struct.unpack_from("<I", data, offset)[0]
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(change_header(data, offset, length + delta))


@pytest.mark.parametrize("section", ("metadata", "content"))
@pytest.mark.parametrize("fault", ("truncated", "adler", "junk", "second_stream", "raw", "gzip", "dictionary"))
def test_rejects_invalid_section_streams(section: str, fault: str) -> None:
    plain = metadata() if section == "metadata" else tlv(4, b"TEXT")
    encoded = zlib.compress(plain)
    if fault == "truncated":
        encoded = encoded[:-1]
    elif fault == "adler":
        encoded = encoded[:-1] + bytes((encoded[-1] ^ 1,))
    elif fault == "junk":
        encoded += b"\x00"
    elif fault == "second_stream":
        encoded += zlib.compress(b"")
    elif fault == "raw":
        compressor = zlib.compressobj(wbits=-15)
        encoded = compressor.compress(plain) + compressor.flush()
    elif fault == "gzip":
        compressor = zlib.compressobj(wbits=31)
        encoded = compressor.compress(plain) + compressor.flush()
    else:
        compressor = zlib.compressobj(zdict=b"TEST DOCUMENT TEXT")
        encoded = compressor.compress(plain) + compressor.flush()
    kwargs = {f"{section}_stream": encoded}
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(content_bytes=tlv(4, b"TEXT"), **kwargs))


@pytest.mark.parametrize("section", ("metadata", "content"))
def test_rejects_highly_compressed_output_beyond_declared_bound(section: str) -> None:
    kwargs = {f"{section}_stream": zlib.compress(b"A" * 262144)}
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(**kwargs))


@pytest.mark.parametrize(
    "encoded",
    (
        b"\x80",
        b"\x80\x00",
        b"\x81\x00",
        b"\xff\xff\xff\xff\x10",
        b"\x80\x80\x80\x80\x80\x00",
        b"\xff\xff\xff\xff\x8f",
    ),
)
@pytest.mark.parametrize("context", ("tlv_length", "depth", "citation", "author_count", "author_length", "reference"))
def test_rejects_invalid_uleb128_in_every_context(encoded: bytes, context: str) -> None:
    meta = metadata()
    if context == "tlv_length":
        content = b"\x04" + encoded
    elif context == "depth":
        content = tlv(1, encoded)
    elif context == "citation":
        content = tlv(2, tlv(2, encoded))
    elif context == "author_count":
        meta = metadata().replace(tlv(3, b"\x01\x03Ada"), tlv(3, encoded))
        content = b""
    elif context == "author_length":
        meta = metadata().replace(tlv(3, b"\x01\x03Ada"), tlv(3, b"\x01" + encoded))
        content = b""
    else:
        content = tlv(5, encoded)
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(meta, content))


@pytest.mark.parametrize("content", (b"\x04", b"\x04\x02A", b"\x04\x00\x01", tlv(2, b"\x01"), tlv(2, b"\x01\x02A")))
def test_rejects_truncated_records_and_residual_bytes(content: bytes) -> None:
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(content_bytes=content))


@pytest.mark.parametrize("kind", (0, 6, 127, 128, 255))
def test_rejects_unknown_content_identifiers(kind: int) -> None:
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(content_bytes=tlv(kind, b"")))


@pytest.mark.parametrize("kind", (0, 3, 127, 128, 255))
def test_rejects_unknown_inline_identifiers(kind: int) -> None:
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(content_bytes=tlv(2, tlv(kind, b"A"))))


@pytest.mark.parametrize("kind", (0, 6, 127, 129, 255))
def test_rejects_unknown_metadata_identifiers(kind: int) -> None:
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(metadata() + tlv(kind, b"")))


@pytest.mark.parametrize("tail", (b"\x00", b"\x01", b"\x80"))
def test_rejects_bytes_after_inline_citation_number(tail: bytes) -> None:
    content = tlv(2, tlv(2, b"\x01" + tail)) + tlv(5, b"\x01https://example.com")
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(content_bytes=content))


@pytest.mark.parametrize("number", (1, 127, 128, 16383, 16384, 2097151, 2097152, 268435455, 268435456, 4294967295))
def test_canonical_integer_boundary_citations(number: int) -> None:
    document = basic_document((libdsx.Paragraph((libdsx.Citation(number),)), libdsx.Reference(number, "https://example.com")))
    encoded = libdsx.dumps(document)
    expected = tlv(2, tlv(2, u(number))) + tlv(5, u(number) + b"https://example.com")
    assert sections(encoded)[1] == expected
    assert libdsx.loads(encoded) == document


def test_reader_accepts_adjacent_text_writer_merges_them() -> None:
    raw = container(content_bytes=tlv(2, tlv(1, b"Hello ") + tlv(1, b"world")))
    decoded = libdsx.loads(raw)
    assert sections(libdsx.dumps(decoded))[1] == tlv(2, tlv(1, b"Hello world"))


def test_metadata_only_inspection_exposes_header_and_unverified_content() -> None:
    encoded = container(content_bytes=tlv(4, b"TEXT"))
    result = libdsx.read_metadata_bytes(encoded)
    assert result.metadata == basic_document().metadata
    assert result.content_verified is False
    assert result.header.layout_version == 1
    assert result.header.compression == 1
    assert result.header.reserved == 0
    assert result.header.metadata_compressed_length == struct.unpack_from("<I", encoded, 12)[0]
    assert result.header.metadata_length == len(metadata())
    assert result.header.content_compressed_length == struct.unpack_from("<I", encoded, 20)[0]
    assert result.header.content_length == 6
    assert result.header.checksum == zlib.crc32(encoded[:28])


def test_metadata_only_inspection_leaves_content_unread_and_unverified() -> None:
    encoded = container(content_stream=b"broken content stream")
    boundary = 32 + struct.unpack_from("<I", encoded, 12)[0]

    class MetadataOnlyStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            if size < 0 or self.tell() + size > boundary:
                raise AssertionError("Metadata inspection attempted to read content")
            return super().read(size)

    stream = MetadataOnlyStream(encoded)
    result = libdsx.read_metadata(stream)
    assert result.metadata == basic_document().metadata
    assert result.content_verified is False
    assert not stream.closed
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(encoded)


@pytest.mark.parametrize("transform", (lambda data: data[:-1], lambda data: data + b"\x00", lambda data: change_header(data, 10, 1, "<H")))
def test_metadata_only_still_validates_header_and_total_file_size(transform) -> None:
    encoded = transform(container())
    with pytest.raises(libdsx.DSXError):
        libdsx.read_metadata_bytes(encoded)


def test_metadata_only_rejects_invalid_metadata() -> None:
    encoded = container(metadata().replace(b"TEST DOCUMENT", b"Test Document"))
    with pytest.raises(libdsx.DSXError):
        libdsx.read_metadata_bytes(encoded)


@pytest.mark.parametrize(
    "field",
    (tlv(1, b"1.1"), tlv(2, b"TEST DOCUMENT"), tlv(3, b"\x01\x03Ada"), tlv(4, b"1"), tlv(5, b"09.22.2026")),
)
@pytest.mark.parametrize("fault", ("missing", "duplicate", "wrong_order"))
def test_rejects_missing_repeated_and_out_of_order_required_metadata(field: bytes, fault: str) -> None:
    if fault == "missing":
        raw = metadata().replace(field, b"")
    elif fault == "duplicate":
        raw = metadata() + field
    else:
        raw = metadata().replace(field, b"") + field if field[0] != 5 else field + metadata().replace(field, b"")
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(raw))


@pytest.mark.parametrize("value", (b"", b"1", b"1.00", b"1.0", b"1.10", b"2.0"))
def test_rejects_unsupported_document_schema(value: bytes) -> None:
    raw = metadata().replace(tlv(1, b"1.1"), tlv(1, value))
    with pytest.raises(libdsx.ValidationError, match="metadata.dsx_version"):
        libdsx.loads(container(raw))


@pytest.mark.parametrize("value", (b"", b"0", b"01", b"+1", b"-1", b"1.0", b" 1", b"1 ", b"4294967296", b"9" * 6000))
def test_rejects_noncanonical_and_out_of_range_revision_text(value: bytes) -> None:
    raw = metadata().replace(tlv(4, b"1"), tlv(4, value))
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(raw))


@pytest.mark.parametrize("value", (b"9.22.2026", b"09.2.2026", b"09.22.026", b"2026-09-22", b"00.01.2026", b"01.00.2026", b"01.01.0000", b"02.29.1900", b"04.31.2026"))
def test_rejects_invalid_stored_date_text(value: bytes) -> None:
    raw = metadata().replace(tlv(5, b"09.22.2026"), tlv(5, value))
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(raw))


@pytest.mark.parametrize("value", (b"\x00", b"\x01\x00", b"\x02\x03Ada", b"\x01\x03Ada\x00", b"\x01\x04Ada", b"\x01\x03A\x7fa"))
def test_rejects_malformed_author_list(value: bytes) -> None:
    raw = metadata().replace(tlv(3, b"\x01\x03Ada"), tlv(3, value))
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(raw))


@pytest.mark.parametrize(
    "suffix",
    (
        tlv(128, b"\x01zfirst") + tlv(128, b"\x01asecond"),
        tlv(128, b"\x01afirst") + tlv(128, b"\x01asecond"),
        tlv(128, b"\x00value"),
        tlv(128, b"\x05titlevalue"),
        tlv(128, b"\x07authorsvalue"),
        tlv(128, b"\x08revisionvalue"),
        tlv(128, b"\x04datevalue"),
        tlv(128, b"\x0bdsx_versionvalue"),
        tlv(128, b"\x80\x00value"),
        tlv(128, b"\x03ab"),
        tlv(128, b"\x01a\x00"),
    ),
)
def test_rejects_invalid_additional_metadata_encoding(suffix: bytes) -> None:
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(metadata() + suffix))


def test_rejects_additional_metadata_before_required_fields() -> None:
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(container(tlv(128, b"\x01avalue") + metadata()))


@pytest.mark.parametrize("section", ("metadata", "content"))
@pytest.mark.parametrize("control", (b"\x00", b"\t", b"\r", b"\x1f", b"\x7f", b"\x80", b"\xff"))
def test_rejects_forbidden_stored_text_bytes(section: str, control: bytes) -> None:
    if section == "metadata":
        data = container(metadata().replace(tlv(2, b"TEST DOCUMENT"), tlv(2, b"TEST" + control)))
    else:
        data = container(content_bytes=tlv(4, b"TEST" + control))
    with pytest.raises(libdsx.DSXError):
        libdsx.loads(data)


@pytest.mark.parametrize("offset", (12, 16, 20, 24))
def test_invalid_header_is_rejected_before_section_reads(offset: int) -> None:
    encoded = change_header(container(), offset, 4294967295)

    class HeaderOnlyStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            if size < 0 or self.tell() + size > 32:
                raise AssertionError("Reader accessed section bytes before validating the header")
            return super().read(size)

    with pytest.raises(libdsx.DSXError):
        libdsx.load(HeaderOnlyStream(encoded))
