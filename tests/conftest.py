"""Shared test helpers."""

from __future__ import annotations

import pytest

from pdftranslator import translate as translate_mod
from pdftranslator.translate import Translator


class _Reply:
    """Minimal stand-in for a litellm ChatCompletion response."""

    def __init__(self, content: str) -> None:
        message = type("Message", (), {"content": content})()
        choice = type("Choice", (), {"message": message, "finish_reason": "stop"})()
        self.choices = [choice]
        self._hidden_params: dict = {}


class ScriptedModel:
    """A scripted provider, patched in at the transport boundary.

    Replacing litellm.completion rather than Translator._complete keeps the
    retry/backoff logic under test real. Each entry in ``replies`` is either the
    raw text the model returns or an Exception to raise from that call.
    """

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.sent: list[list[dict]] = []
        self.slept: list[float] = []

    def __call__(self, **kwargs):
        self.sent.append(kwargs["messages"])
        if not self.replies:
            return _Reply("")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if hasattr(reply, "choices"):
            return reply          # a caller-built response carrying usage etc.
        return _Reply(reply)


@pytest.fixture
def cache_dir(tmp_path):
    return str(tmp_path / "cache")


@pytest.fixture
def scripted(cache_dir, monkeypatch):
    """Factory: a real Translator wired to a scripted provider.

    The returned Translator carries `.model` (the ScriptedModel) so tests can
    inspect the messages sent and the backoff waited.
    """
    made: list[Translator] = []

    def _make(replies, **kwargs):
        kwargs.setdefault("model", "fake/model")
        kwargs.setdefault("target_lang", "English")
        kwargs.setdefault("cache_dir", cache_dir)
        model = ScriptedModel(replies)
        monkeypatch.setattr(translate_mod.litellm, "completion", model)
        monkeypatch.setattr(translate_mod.time, "sleep", model.slept.append)
        t = Translator(**kwargs)
        t.model = model
        t.sent = model.sent
        made.append(t)
        return t

    yield _make
    for t in made:
        t.close()
