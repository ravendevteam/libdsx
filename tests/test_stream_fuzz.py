from __future__ import annotations

import io
import random
from datetime import date
from textwrap import fill

import pytest

import libdsx as dsx
from libdsx import streaming
from test_codec import container, sections, tlv


def generated_document(seed: int) -> dsx.Document:
    rng = random.Random(seed)
    records = []
    citations = set()
    depth = 0
    for _ in range(rng.randrange(5, 35)):
        kind = rng.randrange(4)
        if kind == 0:
            depth = rng.randrange(1, min(depth + 1, 8) + 1)
            records.append(dsx.Heading(depth, "SECTION " + str(rng.randrange(100))))
        elif kind == 1:
            lines = [" " * rng.randrange(4) + "x" * rng.randrange(90) + " " * rng.randrange(4) for _ in range(rng.randrange(5))]
            value = "\n".join(lines)
            if rng.randrange(2):
                value += "\n"
            records.append(dsx.AsciiBlock(value))
        else:
            value = " ".join("word" * rng.randrange(1, 15) for _ in range(rng.randrange(1, 80)))
            cuts = sorted({0, len(value), *(rng.randrange(len(value) + 1) for _ in range(8))})
            inlines = tuple(dsx.Text(value[left:right]) for left, right in zip(cuts, cuts[1:]))
            if rng.randrange(2):
                number = rng.randrange(1, 20)
                citations.add(number)
                inlines += (dsx.Citation(number),)
            records.append((dsx.Paragraph if kind == 2 else dsx.BulletItem)(inlines))
    records.extend(dsx.Reference(number, f"https://example.com/{number}") for number in sorted(citations))
    return dsx.Document(dsx.Metadata(f"TEST {seed}", ("Ada", "Grace"), seed + 1, date(2026, 1, 1)), tuple(records))


def reference_render(document: dsx.Document) -> str:
    def center(value):
        return " " * ((98 - len(value)) // 2) + value

    metadata = document.metadata
    header = ["", "", "", center(metadata.title), *map(center, metadata.authors), "", center(f"Rev. {metadata.revision}, 01.01.2026"), "", "", ""]
    output = "\n".join(header) + "\n"
    counters = [0] * 8
    previous = None
    for record in document.records:
        if previous is not None:
            if isinstance(record, dsx.Heading) and record.depth == 1:
                output += "\n\n"
            elif not isinstance(record, dsx.Reference) or not isinstance(previous, dsx.Reference):
                output += "\n"
        if isinstance(record, dsx.Heading):
            counters[record.depth - 1] += 1
            counters[record.depth:] = [0] * (8 - record.depth)
            number = ".".join(map(str, counters[:record.depth])) + ("." if record.depth == 1 else "")
            title = number + " " + record.title
            output += title + "\n"
            if record.depth == 1:
                output += "=" * len(title) + "\n"
        elif isinstance(record, dsx.AsciiBlock):
            output += record.value
            if record.value and not record.value.endswith("\n"):
                output += "\n"
        else:
            if isinstance(record, dsx.Reference):
                text = f"[{record.number}] {record.url}"
            else:
                text = "".join(item.value if isinstance(item, dsx.Text) else f"[{item.number}]" for item in record.inlines)
            bullet = isinstance(record, dsx.BulletItem)
            output += fill(text, width=98, initial_indent="* " if bullet else "", subsequent_indent="  " if bullet else "", break_long_words=False, break_on_hyphens=False) + "\n"
        previous = record
    return output


@pytest.mark.parametrize("seed", range(50))
def test_generated_documents_have_identical_streaming_and_eager_semantics(seed, monkeypatch):
    expected = generated_document(seed)
    encoded = dsx.dumps(expected, compression_level=seed % 10)
    monkeypatch.setattr(streaming, "CHUNK_SIZE", (1, 7, 98, 257, 65536)[seed % 5])
    restored = dsx.loads(encoded)
    assert sections(dsx.dumps(restored)) == sections(encoded)
    assert tuple(dsx.iter_records(io.BytesIO(encoded))) == restored.records
    dsx.validate_file(io.BytesIO(encoded))
    text = reference_render(expected)
    assert dsx.render(expected) == text
    assert dsx.render(restored) == text
    assert "".join(dsx.iter_render(io.BytesIO(encoded))) == text


@pytest.mark.parametrize("seed", range(10))
def test_mutated_content_is_consistently_rejected_or_accepted(seed, monkeypatch):
    rng = random.Random(seed)
    monkeypatch.setattr(streaming, "CHUNK_SIZE", (1, 7, 257)[seed % 3])
    base = tlv(2, tlv(1, b"Hello world")) + tlv(4, b"ASCII\n")
    for index in range(100):
        if index % 4 == 0:
            content = rng.randbytes(rng.randrange(80))
        else:
            content = bytearray(base)
            for _ in range(rng.randrange(1, 5)):
                content[rng.randrange(len(content))] = rng.randrange(256)
            content = bytes(content)
        encoded = container(content_bytes=content)
        try:
            value = dsx.loads(encoded)
        except dsx.DSXError:
            for consume in (lambda: dsx.validate_file(io.BytesIO(encoded)), lambda: tuple(dsx.iter_records(io.BytesIO(encoded))), lambda: "".join(dsx.iter_render(io.BytesIO(encoded)))):
                with pytest.raises(dsx.DSXError):
                    consume()
        else:
            dsx.validate_file(io.BytesIO(encoded))
            assert tuple(dsx.iter_records(io.BytesIO(encoded))) == value.records
            assert "".join(dsx.iter_render(io.BytesIO(encoded))) == dsx.render(value)
