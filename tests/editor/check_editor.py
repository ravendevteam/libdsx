from __future__ import annotations

import importlib.util
import os
import sys
from datetime import date
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt")

from PyQt5.QtCore import QMimeData
from PyQt5.QtWidgets import QFileDialog, QMessageBox

import libdsx as dsx


ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "tests" / "fixtures" / "DossierRev1.dsx"
SPEC = importlib.util.spec_from_file_location("dossier_editor_checks", ROOT / "dossier_editor.py")
editor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = editor
SPEC.loader.exec_module(editor)


@pytest.fixture
def window(qtbot, monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Discard)
    widget = editor.DossierWindow()
    qtbot.addWidget(widget)
    yield widget
    for index in range(widget.tabs.count()):
        widget.tabs.widget(index).dirty = False


@pytest.fixture
def errors(monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: messages.append(args[2]))
    return messages


def test_new_document_has_ghost_fields_and_effective_defaults(window):
    assert window.tabs.count() == 1
    tab = window.tabs.widget(0)
    assert tab.title_edit.text() == ""
    assert tab.title_edit.placeholderText() == "Untitled Document"
    assert tab.authors_edit.toPlainText() == ""
    assert tab.authors_edit.placeholderText() == "Author Name"
    assert tab.revision_edit.text() == ""
    assert tab.revision_edit.placeholderText() == "1"
    assert tab.date_edit.text() == ""
    assert tab.date_edit.placeholderText() == date.today().strftime("%m.%d.%Y")
    assert tab.document() == dsx.Document(dsx.Metadata("UNTITLED DOCUMENT", ("Author Name",), 1, date.today()))
    assert tab.path is None
    assert not tab.dirty


def test_metadata_edits_and_tabs_are_independent(window):
    first = window.tabs.widget(0)
    first.title_edit.setText("FIRST DOCUMENT")
    first.authors_edit.setPlainText("Ada\nGrace")
    first.revision_edit.setText("3")
    first.date_edit.setText("02.29.2024")
    second = window.new_document()
    assert window.tabs.count() == 2
    assert window.tabs.currentWidget() is second
    assert first.document().metadata == dsx.Metadata("FIRST DOCUMENT", ("Ada", "Grace"), 3, date(2024, 2, 29))
    assert first.dirty
    assert not second.dirty
    assert second.document().metadata.title == "UNTITLED DOCUMENT"


def test_sample_load_save_preserves_entire_document(window, tmp_path, trace):
    trace("Load the specification into editor controls and save through libdsx")
    expected = dsx.load(SAMPLE)
    tab = window.open_path(SAMPLE)
    assert tab.document() == expected
    assert not tab.dirty
    assert tab.additional == expected.metadata.additional
    destination = tmp_path / "saved specification.dsx"
    assert window.save_tab(tab, destination)
    assert dsx.load(destination) == expected
    assert tab.path == destination.resolve()
    assert not tab.dirty
    trace("All records, citations, ASCII content, and additional metadata survived")


def test_open_same_path_focuses_existing_tab(window):
    first = window.open_path(SAMPLE)
    window.new_document()
    count = window.tabs.count()
    assert window.open_path(SAMPLE) is first
    assert window.tabs.count() == count
    assert window.tabs.currentWidget() is first


def test_invalid_open_preserves_tabs(window, tmp_path, errors):
    invalid = tmp_path / "bad.dsx"
    invalid.write_bytes(b"invalid")
    current = window.tabs.currentWidget()
    assert window.open_path(invalid) is None
    assert window.tabs.count() == 1
    assert window.tabs.currentWidget() is current
    assert errors


@pytest.mark.parametrize("value", ("", "\n", "ABC\n", "ABC\n\n", "  +---+  \n  | A |\n"))
def test_ascii_layout_roundtrip(window, tmp_path, value):
    document = dsx.Document(dsx.Metadata("ASCII DOCUMENT", ("Author",), 1, date.today()), (dsx.AsciiBlock(value),))
    path = tmp_path / "ascii.dsx"
    dsx.dump(document, path)
    tab = window.open_path(path)
    assert tab.document() == document
    assert window.save_tab(tab)
    assert dsx.load(path) == document


def test_new_document_saves_defaults_and_body(window, tmp_path):
    tab = window.tabs.widget(0)
    tab.record_editors[0].editor.set_inlines((dsx.Text("A new document."),))
    path = tmp_path / "new.dsx"
    assert tab.dirty
    assert window.save_tab(tab, path)
    assert dsx.load(path) == tab.document()
    assert dsx.load(path).records == (dsx.Paragraph((dsx.Text("A new document."),)),)
    assert tab.path == path.resolve()
    assert not tab.dirty


@pytest.mark.parametrize("field,value", (("date_edit", "02.30.2026"), ("revision_edit", "0")))
def test_invalid_save_preserves_existing_file(window, tmp_path, errors, field, value):
    path = tmp_path / "existing.dsx"
    path.write_bytes(SAMPLE.read_bytes())
    original = path.read_bytes()
    tab = window.open_path(path)
    getattr(tab, field).setText(value)
    assert not window.save_tab(tab)
    assert path.read_bytes() == original
    assert tab.path == path.resolve()
    assert tab.dirty
    assert errors


def test_failed_atomic_write_keeps_document_unsaved(window, tmp_path, errors, monkeypatch):
    tab = window.tabs.widget(0)
    tab.title_edit.setText("CHANGED")
    destination = tmp_path / "protected.dsx"
    destination.write_bytes(b"original")

    def fail_write(path, data):
        raise OSError("Simulated write failure")

    monkeypatch.setattr(editor, "save_bytes", fail_write)
    assert not window.save_tab(tab, destination)
    assert destination.read_bytes() == b"original"
    assert tab.path is None
    assert tab.dirty
    assert errors


@pytest.mark.parametrize("choice,closed", ((QMessageBox.Cancel, False), (QMessageBox.Discard, True)))
def test_close_dirty_document_honors_choice(window, monkeypatch, choice, closed):
    tab = window.tabs.widget(0)
    tab.title_edit.setText("CHANGED")
    window.new_document()
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: choice)
    assert window.close_tab(0) is closed
    assert window.tabs.count() == (1 if closed else 2)
    if not closed:
        assert window.tabs.widget(0) is tab
        assert tab.dirty


def test_close_save_cancel_keeps_unsaved_tab(window, monkeypatch):
    tab = window.tabs.widget(0)
    tab.title_edit.setText("CHANGED")
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Save)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args, **kwargs: ("", ""))
    assert not window.close_tab(0)
    assert window.tabs.widget(0) is tab
    assert tab.path is None
    assert tab.dirty


def test_close_save_writes_changes_before_closing(window, tmp_path, monkeypatch):
    path = tmp_path / "existing.dsx"
    dsx.dump(dsx.Document(dsx.Metadata("BEFORE", ("Author",), 1, date.today())), path)
    tab = window.open_path(path)
    tab.title_edit.setText("AFTER")
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Save)
    assert window.close_tab(window.tabs.indexOf(tab))
    assert dsx.load(path).metadata.title == "AFTER"
    assert window.tabs.count() == 1


def test_inline_clipboard_preserves_citations_and_literal_markers(qtbot):
    source = editor.InlineEditor()
    target = editor.InlineEditor()
    qtbot.addWidget(source)
    qtbot.addWidget(target)
    expected = (dsx.Text("Literal [1], cite "), dsx.Citation(1), dsx.Text("."))
    source.set_inlines(expected)
    source.selectAll()
    target.insertFromMimeData(source.createMimeDataFromSelection())
    assert target.inlines() == expected
    target.undo()
    assert target.inlines() == ()
    target.redo()
    assert target.inlines() == expected


def test_plaintext_paste_does_not_invent_citations(qtbot):
    widget = editor.InlineEditor()
    qtbot.addWidget(widget)
    data = QMimeData()
    data.setText("Literal [1].")
    widget.insertFromMimeData(data)
    assert widget.inlines() == (dsx.Text("Literal [1]."),)


def test_structural_split_undo_redo_is_atomic(window):
    tab = window.tabs.widget(0)
    block = tab.record_editors[0]
    block.editor.set_inlines((dsx.Text("Hello world"),))
    cursor = block.editor.textCursor()
    cursor.setPosition(6)
    block.editor.setTextCursor(cursor)
    tab.split_record(block)
    expected = (dsx.Paragraph((dsx.Text("Hello"),)), dsx.Paragraph((dsx.Text("world"),)))
    assert tab.document().records == expected
    tab.undo()
    assert tab.document().records == (dsx.Paragraph((dsx.Text("Hello world"),)),)
    assert len(tab.record_editors) == 1
    tab.redo()
    assert tab.document().records == expected


def test_add_remove_and_type_change_are_undoable(window):
    tab = window.tabs.widget(0)
    record = dsx.AsciiBlock("ASCII")
    tab.add_record(record)
    assert tab.document().records == (record,)
    tab.undo()
    assert tab.document().records == ()
    tab.redo()
    tab.change_kind(tab.record_editors[-1], "Paragraph")
    assert tab.document().records == (dsx.Paragraph((dsx.Text("ASCII"),)),)
    tab.undo()
    assert tab.document().records == (record,)
    tab.remove_record(tab.record_editors[-1])
    assert tab.document().records == ()
    tab.undo()
    assert tab.document().records == (record,)


def test_lowercase_title_typing_undo_restores_placeholder(window, qtbot):
    window.show()
    tab = window.tabs.widget(0)
    tab.title_edit.setFocus()
    qtbot.keyClicks(tab.title_edit, "hello")
    assert tab.title_edit.text() == "HELLO"
    tab.undo()
    assert tab.title_edit.text() == ""
    assert not tab.dirty
    tab.redo()
    assert tab.title_edit.text() == "HELLO"
