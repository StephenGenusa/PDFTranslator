"""The pure helpers behind batch parsing and the repair pass.

_looks_untranslated gates the repair pass, so it decides whether the run pays
for extra per-line calls or silently leaves an echoed line in the output.
"""

from __future__ import annotations

import pytest

from pdftranslator.translate import _extract_json_object, _looks_untranslated


class TestExtractJsonObject:
    def test_plain_object(self):
        assert _extract_json_object('{"0": "hello", "1": "world"}') == {
            "0": "hello", "1": "world"}

    def test_strips_a_json_code_fence(self):
        assert _extract_json_object('```json\n{"0": "hi"}\n```') == {"0": "hi"}

    def test_strips_a_bare_code_fence(self):
        assert _extract_json_object('```\n{"0": "hi"}\n```') == {"0": "hi"}

    def test_recovers_an_object_wrapped_in_prose(self):
        raw = 'Sure! Here is the translation: {"0": "hi"} Hope that helps.'
        assert _extract_json_object(raw) == {"0": "hi"}

    def test_keeps_braces_that_are_inside_string_values(self):
        raw = '{"0": "a brace } inside", "1": "b"}'
        assert _extract_json_object(raw) == {"0": "a brace } inside", "1": "b"}

    def test_nested_objects_survive(self):
        assert _extract_json_object('{"0": {"a": 1}}') == {"0": {"a": 1}}

    @pytest.mark.parametrize("raw", ["", "   ", "not json at all",
                                     "{not valid json}", '["a", "b"]', "null"])
    def test_unusable_input_returns_none(self, raw):
        assert _extract_json_object(raw) is None


class TestLooksUntranslated:
    @pytest.mark.parametrize("text", ["labas", "Labas rytas", "kalba"])
    def test_an_echoed_line_is_flagged(self, text):
        assert _looks_untranslated(text, text) is True

    def test_case_and_surrounding_space_are_ignored(self):
        assert _looks_untranslated("Labas", "  LABAS  ") is True

    def test_a_real_translation_is_not_flagged(self):
        assert _looks_untranslated("labas rytas", "good morning") is False

    @pytest.mark.parametrize("text", ["123", "1917", "OK", "A.", "-", "42.5"])
    def test_tokens_with_too_few_letters_are_left_alone(self, text):
        """Numbers and short tokens are legitimately identical in both
        languages; retrying them would just re-buy the same answer."""
        assert _looks_untranslated(text, text) is False

    def test_the_letter_threshold_boundary(self):
        assert _looks_untranslated("abc", "abc") is False   # 3 letters
        assert _looks_untranslated("abcd", "abcd") is True  # 4 letters

    def test_empty_source_is_not_flagged(self):
        assert _looks_untranslated("", "") is False
        assert _looks_untranslated("   ", "") is False

    def test_a_proper_noun_is_flagged_and_that_is_a_known_cost(self):
        """Names translate to themselves, so they trip the check and cost a
        retry that returns the same text. Documented, not desired."""
        assert _looks_untranslated("Nikolaj Starikov", "Nikolaj Starikov") is True
