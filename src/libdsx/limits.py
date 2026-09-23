from __future__ import annotations

from dataclasses import dataclass, fields

from .errors import ResourceLimitError


@dataclass(frozen=True, slots=True)
class Limits:
    max_metadata_stored: int | None = None
    max_metadata_decoded: int | None = None
    max_content_stored: int | None = None
    max_content_decoded: int | None = None
    max_records: int | None = None
    max_inlines: int | None = None
    max_pending_citations: int | None = None
    max_record_bytes: int | None = None
    max_inline_bytes: int | None = None
    max_rendered_chars: int | None = None

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{field.name} must be a nonnegative integer or None")


def check_limits(limits: Limits | None) -> None:
    if limits is not None and not isinstance(limits, Limits):
        raise TypeError("limits must be a Limits instance or None")


def check_limit(limits: Limits | None, name: str, value: int) -> None:
    if limits is not None:
        maximum = getattr(limits, name)
        if maximum is not None and value > maximum:
            raise ResourceLimitError(f"{name}: {value} exceeds the application limit of {maximum}")
