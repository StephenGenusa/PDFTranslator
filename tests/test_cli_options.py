"""CLI options that must actually reach the translator.

Translator accepts request_pause / max_tokens / max_attempts, but the CLI used
to hardcode all three. max_attempts governs the API retry, so a user being rate
limited had no way to turn retries up, or to slow the request rate down, without
editing the source.

Each test drives the real CLI and observes the effect at the transport boundary.
"""

from __future__ import annotations

import fitz
import pytest
from typer.testing import CliRunner

from pdftranslator import translate as translate_mod
from pdftranslator.cli import app

runner = CliRunner()


class _Reply:
    def __init__(self, content: str) -> None:
        message = type("Message", (), {"content": content})()
        self.choices = [type("Choice", (), {"message": message})()]


@pytest.fixture
def book(tmp_path):
    path = tmp_path / "in.pdf"
    with fitz.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 144), "Labas rytas visiems", fontsize=12)
        doc.save(path)
    return path


@pytest.fixture
def provider(monkeypatch):
    """Records every call; fails by default so retries are observable."""
    calls: list[dict] = []
    sleeps: list[float] = []

    def fake(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(translate_mod.litellm, "completion", fake)
    monkeypatch.setattr(translate_mod.time, "sleep", sleeps.append)
    return calls, sleeps


def _run(book, tmp_path, *extra):
    return runner.invoke(
        app,
        ["-i", str(book), "-o", str(tmp_path / "out.pdf"),
         "--cache-dir", str(tmp_path / "cache"), *extra],
    )


def test_max_attempts_controls_the_number_of_api_calls(book, tmp_path, provider):
    """Breaks if --max-attempts stops reaching Translator."""
    calls, _ = provider
    result = _run(book, tmp_path, "--max-attempts", "4")
    assert result.exit_code == 0, result.output
    assert len(calls) == 4, f"expected 4 attempts, got {len(calls)}"


def test_max_attempts_of_one_disables_retrying(book, tmp_path, provider):
    calls, _ = provider
    _run(book, tmp_path, "--max-attempts", "1")
    assert len(calls) == 1


def test_default_max_attempts_is_unchanged(book, tmp_path, provider):
    calls, _ = provider
    _run(book, tmp_path)
    assert len(calls) == 3


def test_request_pause_is_applied_before_each_call(book, tmp_path, provider):
    calls, sleeps = provider
    _run(book, tmp_path, "--request-pause", "0.25", "--max-attempts", "2")
    assert sleeps.count(0.25) == 2, sleeps


def test_max_tokens_reaches_the_provider(book, tmp_path, provider):
    calls, _ = provider
    _run(book, tmp_path, "--max-tokens", "512", "--max-attempts", "1")
    assert calls[0]["max_tokens"] == 512


def test_all_three_options_are_documented_in_help():
    result = runner.invoke(app, ["--help"])
    for flag in ("--request-pause", "--max-tokens", "--max-attempts"):
        assert flag in result.output, flag
