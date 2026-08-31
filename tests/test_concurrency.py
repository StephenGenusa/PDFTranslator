"""Translating pages concurrently, without exceeding the provider's limits."""

from __future__ import annotations

import threading
import time

import pymupdf
import pytest
from typer.testing import CliRunner

from pdftranslator import translate as translate_mod
from pdftranslator.cli import app
from pdftranslator.orchestrator import Orchestrator
from pdftranslator.translate import Translator

runner = CliRunner()


class _Headers(dict):
    def get(self, key, default=None):
        return super().get(key.lower(), default)


def _rate_limited(retry_after="0.2"):
    response = type("Response", (), {"headers": _Headers({"retry-after": retry_after})})()
    return translate_mod.litellm.RateLimitError(
        message="429", llm_provider="openrouter", model="m", response=response,
    )


def _book(path, pages):
    with pymupdf.open() as doc:
        for i in range(pages):
            page = doc.new_page(width=400, height=600)
            page.insert_text((40, 200), f"Puslapio {i + 1} tekstas yra cia", fontsize=11)
        doc.save(path)


def _reply(content):
    message = type("Message", (), {"content": content})()
    choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
    return type("R", (), {"choices": [choice], "_hidden_params": {}})()


def test_pages_are_translated_concurrently(tmp_path, monkeypatch):
    """Breaks if the page loop goes back to being strictly serial."""
    peak = 0
    active = 0
    lock = threading.Lock()

    def slow(**kwargs):
        nonlocal peak, active
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return _reply('{"0": "translated", "1": "s"}')

    monkeypatch.setattr(translate_mod.litellm, "completion", slow)
    src = tmp_path / "in.pdf"
    _book(src, 8)
    t = Translator(model="fake/x", target_lang="English", cache_dir=str(tmp_path / "c"))
    try:
        Orchestrator(t, mirror=False, concurrency=4).run(src, tmp_path / "out.pdf", "all")
    finally:
        t.close()
    assert peak > 1, f"never more than {peak} call(s) in flight"


def test_output_pages_stay_in_source_order(tmp_path, monkeypatch):
    """Concurrency must not reorder the book."""
    import random

    def jittery(**kwargs):
        time.sleep(random.uniform(0.001, 0.03))
        body = kwargs["messages"][1]["content"]
        # the page's own text is last; the preceding-context line comes first
        marker = body.rsplit("Puslapio ", 1)[1].split(" ")[0] if "Puslapio " in body else "?"
        return _reply('{"0": "PAGE %s", "1": "s"}' % marker)

    monkeypatch.setattr(translate_mod.litellm, "completion", jittery)
    src = tmp_path / "in.pdf"
    _book(src, 10)
    out = tmp_path / "out.pdf"
    t = Translator(model="fake/x", target_lang="English", cache_dir=str(tmp_path / "c"))
    try:
        Orchestrator(t, mirror=False, concurrency=5).run(src, out, "all")
    finally:
        t.close()
    with pymupdf.open(out) as doc:
        seen = [f"PAGE {i + 1}" in doc[i].get_text() for i in range(doc.page_count)]
    assert all(seen), seen


def test_a_rate_limit_notice_pauses_the_run(tmp_path, monkeypatch):
    """The provider said wait; the run must wait, not retry straight through."""
    calls = []

    def limited(**kwargs):
        calls.append(time.monotonic())
        if len(calls) <= 2:
            raise _rate_limited("0.2")
        return _reply('{"0": "translated", "1": "s"}')

    monkeypatch.setattr(translate_mod.litellm, "completion", limited)
    src = tmp_path / "in.pdf"
    _book(src, 3)
    t = Translator(model="fake/x", target_lang="English",
                   cache_dir=str(tmp_path / "c"), max_attempts=4)
    started = time.monotonic()
    try:
        Orchestrator(t, mirror=False, concurrency=3).run(src, tmp_path / "out.pdf", "all")
    finally:
        t.close()
    assert time.monotonic() - started >= 0.2, "did not honour the retry-after"
    assert t.stats["rate_limited"] >= 1


def test_concurrency_option_is_wired_through(tmp_path, monkeypatch):
    from pdftranslator import orchestrator as orch

    seen = []
    original = orch.Orchestrator

    class Recording(original):
        def __init__(self, translator, *, mirror, size_fidelity=None, concurrency=None):
            seen.append(concurrency)
            super().__init__(translator, mirror=mirror,
                             size_fidelity=size_fidelity, concurrency=concurrency)

    monkeypatch.setattr("pdftranslator.cli.Orchestrator", Recording)
    monkeypatch.setattr(translate_mod.litellm, "completion",
                        lambda **k: _reply('{"0": "x", "1": "s"}'))
    src = tmp_path / "in.pdf"
    _book(src, 2)
    result = runner.invoke(app, ["-i", str(src), "-o", str(tmp_path / "o.pdf"),
                                 "--cache-dir", str(tmp_path / "c"),
                                 "--concurrency", "6"])
    assert result.exit_code == 0, result.output
    assert seen == [6], seen


def test_concurrency_must_be_at_least_one(tmp_path):
    src = tmp_path / "in.pdf"
    _book(src, 1)
    result = runner.invoke(app, ["-i", str(src), "-o", str(tmp_path / "o.pdf"),
                                 "--cache-dir", str(tmp_path / "c"),
                                 "--concurrency", "0"])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
