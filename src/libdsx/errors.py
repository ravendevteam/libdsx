from __future__ import annotations


class DSXError(ValueError):
    pass


class FormatError(DSXError):
    pass


class ValidationError(DSXError):
    pass


class ResourceLimitError(DSXError):
    pass
