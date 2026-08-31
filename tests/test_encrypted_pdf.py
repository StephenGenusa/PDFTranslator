"""Password-protected PDFs.

An encrypted PDF opens without error and reports its page count, but yields no
text, so the run used to produce a full-length copy of the original with nothing
translated and nothing said about why.
"""

from __future__ import annotations

import pymupdf
import pytest
from typer.testing import CliRunner

from pdftranslator.cli import app

runner = CliRunner()


def _encrypted(path, *, user_pw=None, owner_pw="owner"):
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 144), "Slaptas tekstas puslapyje", fontsize=12)
        kwargs = {"encryption": pymupdf.PDF_ENCRYPT_AES_256, "owner_pw": owner_pw}
        if user_pw:
            kwargs["user_pw"] = user_pw
        doc.save(path, **kwargs)


def test_a_password_protected_pdf_is_reported_not_silently_copied(tmp_path):
    """Breaks if needs_pass stops being checked: the user gets 380 untranslated
    pages and no explanation."""
    src = tmp_path / "locked.pdf"
    _encrypted(src, user_pw="secret")
    result = runner.invoke(app, ["-i", str(src), "-o", str(tmp_path / "out.pdf"),
                                 "--cache-dir", str(tmp_path / "c")])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "password" in result.output.lower()


def test_the_right_password_unlocks_it(tmp_path, monkeypatch):
    from pdftranslator import translate as translate_mod

    def reply(**kwargs):
        message = type("M", (), {"content": '{"0": "Secret text on the page", "1": "s"}'})()
        choice = type("C", (), {"message": message, "finish_reason": "stop"})()
        return type("R", (), {"choices": [choice], "_hidden_params": {}})()

    monkeypatch.setattr(translate_mod.litellm, "completion", reply)
    src = tmp_path / "locked.pdf"
    _encrypted(src, user_pw="secret")
    out = tmp_path / "out.pdf"
    result = runner.invoke(app, ["-i", str(src), "-o", str(out),
                                 "--cache-dir", str(tmp_path / "c"),
                                 "--password", "secret"])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_a_wrong_password_is_reported_clearly(tmp_path):
    src = tmp_path / "locked.pdf"
    _encrypted(src, user_pw="secret")
    result = runner.invoke(app, ["-i", str(src), "-o", str(tmp_path / "out.pdf"),
                                 "--cache-dir", str(tmp_path / "c"),
                                 "--password", "wrong"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_an_owner_password_only_pdf_is_processed_normally(tmp_path, monkeypatch):
    """Permissions-restricted but readable. Rejecting these would be wrong --
    they are common and open without a password."""
    from pdftranslator import translate as translate_mod

    def reply(**kwargs):
        message = type("M", (), {"content": '{"0": "Protected text", "1": "s"}'})()
        choice = type("C", (), {"message": message, "finish_reason": "stop"})()
        return type("R", (), {"choices": [choice], "_hidden_params": {}})()

    monkeypatch.setattr(translate_mod.litellm, "completion", reply)
    src = tmp_path / "restricted.pdf"
    _encrypted(src, user_pw=None)
    out = tmp_path / "out.pdf"
    result = runner.invoke(app, ["-i", str(src), "-o", str(out),
                                 "--cache-dir", str(tmp_path / "c")])
    assert result.exit_code == 0, result.output
    assert out.exists()
