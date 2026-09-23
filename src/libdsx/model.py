from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, TypeAlias


@dataclass(frozen=True, slots=True)
class Text:
    value: str


@dataclass(frozen=True, slots=True)
class Citation:
    number: int


Inline: TypeAlias = Text | Citation


@dataclass(frozen=True, slots=True)
class Heading:
    depth: int
    title: str


@dataclass(frozen=True, slots=True)
class Paragraph:
    inlines: tuple[Inline, ...]


@dataclass(frozen=True, slots=True)
class BulletItem:
    inlines: tuple[Inline, ...]


@dataclass(frozen=True, slots=True)
class AsciiBlock:
    value: str


@dataclass(frozen=True, slots=True)
class Reference:
    number: int
    url: str


Record: TypeAlias = Heading | Paragraph | BulletItem | AsciiBlock | Reference


@dataclass(frozen=True, slots=True)
class Metadata:
    title: str
    authors: tuple[str, ...]
    revision: int
    date: date
    additional: dict[str, str] = field(default_factory=dict)
    dsx_version: str = "1.0"


@dataclass(frozen=True, slots=True)
class Document:
    metadata: Metadata
    records: tuple[Record, ...] = ()


@dataclass(frozen=True, slots=True)
class Header:
    layout_version: int
    compression: int
    reserved: int
    metadata_compressed_length: int
    metadata_length: int
    content_compressed_length: int
    content_length: int
    checksum: int


@dataclass(frozen=True, slots=True)
class MetadataInspection:
    metadata: Metadata
    header: Header
    content_verified: Literal[False] = field(default=False, init=False)
