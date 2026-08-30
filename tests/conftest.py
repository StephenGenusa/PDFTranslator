"""Shared test helpers."""

from __future__ import annotations

import pytest

from pdftranslator.translate import Translator


class ScriptedTranslator(Translator):
    """A real Translator driven by a scripted model.

    Everything except the network call is the production code path. Each entry
    in ``replies`` is either the raw text the model returns, or an Exception to
    raise from that call.
    """

    def __init__(self, replies, **kwargs):
        kwargs.setdefault("model", "fake/model")
        kwargs.setdefault("target_lang", "English")
        super().__init__(**kwargs)
        self._replies = list(replies)
        self.sent: list[list[dict]] = []

    def _complete(self, messages):
        self.sent.append(messages)
        self.stats["api_calls"] += 1
        if not self._replies:
            return ""
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "cache")


@pytest.fixture
def scripted(cache_dir):
    """Factory for ScriptedTranslators that are closed at teardown."""
    made: list[ScriptedTranslator] = []

    def _make(replies, **kwargs):
        kwargs.setdefault("cache_dir", cache_dir)
        t = ScriptedTranslator(replies, **kwargs)
        made.append(t)
        return t

    yield _make
    for t in made:
        t.close()
