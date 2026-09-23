from __future__ import annotations

from .codec import MAX_CONTENT_DECODED, MAX_CONTENT_STORED, MAX_METADATA_DECODED, MAX_METADATA_STORED, SIGNATURE, dump, dumps, load, loads, read_metadata, read_metadata_bytes, validate, validate_file
from .errors import DSXError, FormatError, ResourceLimitError, ValidationError
from .limits import Limits
from .model import AsciiBlock, BulletItem, Citation, Document, Header, Heading, Inline, Metadata, MetadataInspection, Paragraph, Record, Reference, Text
from .render import iter_render, render, render_to
from .streaming import DocumentReader, iter_records, open_document


__all__ = [
    "AsciiBlock",
    "BulletItem",
    "Citation",
    "DSXError",
    "Document",
    "DocumentReader",
    "FormatError",
    "Header",
    "Heading",
    "Inline",
    "Limits",
    "MAX_CONTENT_DECODED",
    "MAX_CONTENT_STORED",
    "MAX_METADATA_DECODED",
    "MAX_METADATA_STORED",
    "Metadata",
    "MetadataInspection",
    "Paragraph",
    "Record",
    "Reference",
    "ResourceLimitError",
    "SIGNATURE",
    "Text",
    "ValidationError",
    "dump",
    "dumps",
    "load",
    "loads",
    "iter_records",
    "iter_render",
    "open_document",
    "read_metadata",
    "read_metadata_bytes",
    "render",
    "render_to",
    "validate",
    "validate_file",
]
