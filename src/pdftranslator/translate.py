"""Whole-page, line-aligned translation via OpenRouter (through litellm).

Design mandate: send the *entire page* (pre-split into lines) so the model has
maximum context, and require it to return one translation per input line, in
order. The single most important correctness property is N-in == N-out: if the
model merges or drops a line, every line after it would be placed at the wrong
baseline. We therefore validate the count and, on mismatch, retry and finally
fall back to per-line translation, which cannot drift.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time

import litellm
from diskcache import Cache

litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_SHARED_ROLE = (
    "You are a professional literary translator. You translate a single page of "
    "a book that has been split into individually numbered lines to preserve its "
    "layout. Use the whole page as context to translate accurately and naturally. "
    "Translate each line independently in place: never merge, split, reorder, add, "
    "or drop lines, and keep each translation close in length to its source line. "
    "Preserve names, numbers and punctuation; if a line is purely a number, symbol "
    "or already in the target language, return it unchanged. "
)

# The batch call wants a JSON object back.
_SYSTEM_BATCH = _SHARED_ROLE + (
    "Output ONLY the requested JSON value — no prose, no commentary, no code fences."
)

# The per-line call wants one bare line. Reusing the batch prompt here told the
# model to answer in JSON while the user turn asked for a plain line; compliant
# models emitted the JSON, which was then stamped onto the page verbatim.
_SYSTEM_LINE = _SHARED_ROLE + (
    "Output ONLY the translated line itself — no prose, no commentary, no code "
    "fences, no quoting, no keys or wrappers of any kind."
)


class TranslationError(RuntimeError):
    pass


class Translator:
    def __init__(
        self,
        *,
        model: str,
        target_lang: str,
        api_key: str | None = None,
        api_base: str | None = None,
        cache_dir: str = ".translation_cache",
        max_tokens: int = 8192,
        max_attempts: int = 3,
        request_pause: float = 0.0,
    ) -> None:
        self.litellm_model = model  # full litellm model string, e.g. "openrouter/google/gemini-2.5-flash-lite"
        self.target_lang = target_lang
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.request_pause = request_pause
        self._cache = Cache(cache_dir)
        self.stats = {"pages": 0, "api_calls": 0, "cache_hits": 0, "fallbacks": 0}

    # ------------------------------------------------------------------- API
    def translate_page(
        self, lines: list[str], preceding_context: str | None = None
    ) -> list[str]:
        """Translate every line of one page; returns a list of the same length."""
        if not lines:
            return []

        cache_key = self._page_key(lines)
        cached = self._cache.get(cache_key)
        if isinstance(cached, list) and len(cached) == len(lines):
            self.stats["cache_hits"] += 1
            return self._repair_cached(lines, [str(v) for v in cached])

        # Keyed batch: the model returns an object keyed by line index, so a
        # dropped/extra line is realigned by key instead of shifting everything.
        partial = self._translate_batch(lines, preceding_context)
        missing = [i for i, r in enumerate(partial) if r is None]
        if missing:
            logger.warning(
                "Batch left %d/%d line(s) untranslated; filling per line.",
                len(missing), len(lines),
            )
            self.stats["fallbacks"] += 1
            page_context = "\n".join(lines)
            for i in missing:
                partial[i] = self._translate_one(lines[i], page_context, i, len(lines))

        result = [r if r is not None else lines[i] for i, r in enumerate(partial)]
        result = self._repair_untranslated(lines, result)
        self._cache.set(cache_key, result)
        self.stats["pages"] += 1
        return result

    def _repair_cached(self, lines: list[str], cached: list[str]) -> list[str]:
        """Strip protocol debris from an entry written before the guards existed.

        Falling back to the source line keeps the page readable at no cost; the
        warning tells the user how to get a real translation for it.
        """
        poisoned = [i for i, v in enumerate(cached) if _SENTINEL in v]
        if not poisoned:
            return cached
        logger.warning(
            "Cached translation for %d line(s) contains internal batch markers; "
            "using the source text instead. Re-run with --ignore-cache to "
            "translate them properly.",
            len(poisoned),
        )
        for i in poisoned:
            cached[i] = lines[i]
        return cached

    def _repair_untranslated(self, lines: list[str], result: list[str]) -> list[str]:
        """Retry any line that came back unchanged but clearly should have been
        translated (e.g. the model echoed the source instead of translating).

        Path-agnostic safety net: catches both batch echoes and fallback misses.
        Numbers, names and short tokens that legitimately stay the same are left
        alone (see :func:`_looks_untranslated`).
        """
        suspects = [i for i, (s, t) in enumerate(zip(lines, result)) if _looks_untranslated(s, t)]
        if not suspects:
            return result
        logger.debug("Repairing %d untranslated line(s): %s", len(suspects), suspects)
        page_context = "\n".join(lines)
        for i in suspects:
            fixed = self._translate_one(lines[i], page_context, i, len(lines))
            if fixed and not _looks_untranslated(lines[i], fixed):
                result[i] = fixed
        return result

    # -------------------------------------------------------------- internal
    def _translate_batch(
        self, lines: list[str], preceding_context: str | None
    ) -> list[str | None]:
        """Translate the page as a keyed object; returns one entry per line.

        Entries the model omitted (or that failed to parse) come back as ``None``
        so the caller can fill just those, rather than discarding the whole page.
        """
        context = ""
        if preceding_context:
            context = (
                "For continuity, the previous page ended with:\n"
                f"{preceding_context}\n\n"
            )
        # Pad with a sentinel so the real last line is never the model's final
        # key — the model consistently drops the last key in the batch.
        n = len(lines)
        padded = lines + [_SENTINEL]
        np_ = len(padded)
        numbered = {str(i): ln for i, ln in enumerate(padded)}
        user = (
            f"{context}Translate the VALUES of this JSON object into "
            f"{self.target_lang}. Return a JSON object with the exact same keys "
            f'("0" through "{np_ - 1}"), each value replaced by its '
            f"translation.\n\n{json.dumps(numbered, ensure_ascii=False)}"
        )

        best: list[str | None] = [None] * np_
        for attempt in range(self.max_attempts):
            raw = self._complete(
                [
                    {"role": "system", "content": _SYSTEM_BATCH},
                    {"role": "user", "content": user},
                ]
            )
            obj = _extract_json_object(raw)
            if obj:
                current = [_batch_value(obj.get(str(i))) for i in range(np_)]
                # Keep the most complete response across attempts.
                if sum(x is not None for x in current) > sum(x is not None for x in best):
                    best = current
                if all(x is not None for x in best[:n]):
                    return best[:n]
            logger.debug(
                "Attempt %d: %d/%d lines present", attempt + 1,
                sum(x is not None for x in best[:n]), n,
            )
            if attempt < self.max_attempts - 1:
                time.sleep(1.5 * (attempt + 1))
        missing_idx = [i for i, v in enumerate(best[:n]) if v is None]
        if missing_idx:
            logger.debug(
                "Missing keys after all attempts: %s. Raw response tail: %r",
                missing_idx, raw[-300:],
            )
        return best[:n]

    def _translate_individually(
        self, lines: list[str], preceding_context: str | None
    ) -> list[str]:
        """Guaranteed-aligned fallback: one call per line, original on failure.

        The full page is still supplied as context so quality stays reasonable.
        """
        page_context = "\n".join(lines)
        return [self._translate_one(line, page_context, i, len(lines)) for i, line in enumerate(lines)]

    def _translate_one(self, line: str, page_context: str, idx: int, total: int) -> str:
        """Translate a single line with the whole page as context.

        Lines that are essentially a number/symbol (e.g. a chapter number "1")
        are returned unchanged — asking the model to "translate" them in isolation
        tends to make it hallucinate a heading.
        """
        if sum(c.isalpha() for c in line) < 2:
            return line
        user = (
            f"Full page (for context only):\n{page_context}\n\n"
            f"Translate ONLY line {idx + 1} into {self.target_lang}. It is not yet "
            f"in {self.target_lang}; translate it fully. Return just the translated "
            f"line, nothing else.\n\nLine {idx + 1}: {line}"
        )
        try:
            raw = self._complete(
                [
                    {"role": "system", "content": _SYSTEM_LINE},
                    {"role": "user", "content": user},
                ]
            )
            return _clean_single(raw) or line
        except Exception as exc:  # noqa: BLE001 - fall back to source text
            logger.debug("Per-line translation failed (%s); keeping source", exc)
            return line

    def _complete(self, messages: list[dict]) -> str:
        self.stats["api_calls"] += 1
        if self.request_pause:
            time.sleep(self.request_pause)
        kwargs: dict = dict(
            model=self.litellm_model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.2,
        )
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        response = litellm.completion(**kwargs)
        return response.choices[0].message.content or ""

    def _page_key(self, lines: list[str]) -> str:
        raw = "\x1f".join(lines)
        digest = hashlib.sha256(
            f"v{_CACHE_VERSION}|{self.litellm_model}|{self.target_lang}|{raw}".encode("utf-8")
        ).hexdigest()
        return digest

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Translation cache cleared")

    def close(self) -> None:
        self._cache.close()


# Bump to invalidate cached page translations when translation logic changes.
_CACHE_VERSION = 3

# Dummy line appended to each batch so the model never treats the real last
# line as the end-of-response — it consistently drops the final key otherwise.
# A UUID-like token: clearly not prose, so the model won't absorb it into
# adjacent line translations the way a punctuation character (¶) would be.
_SENTINEL = "__batch_end_a3f2d8e1__"


def _looks_untranslated(source: str, translated: str) -> bool:
    """True if a translation is suspiciously identical to its source.

    Only flags lines with enough letters to be real words, so numbers, short
    tokens and bare punctuation that legitimately stay unchanged are ignored.
    Proper names may be flagged, but a retry simply returns the name again.
    """
    s, t = source.strip(), translated.strip()
    if not s or s.casefold() != t.casefold():
        return False
    return sum(c.isalpha() for c in s) >= 4


# --------------------------------------------------------------------- parsing
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _extract_json_object(raw: str) -> dict | None:
    if not raw:
        return None
    text = _FENCE_RE.sub("", raw).strip()
    try:
        val = json.loads(text)
        if isinstance(val, dict):
            return val
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            val = json.loads(text[start : end + 1])
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            return None
    return None


def _extract_json_array(raw: str) -> list | None:
    if not raw:
        return None
    text = _FENCE_RE.sub("", raw).strip()
    # Fast path.
    try:
        val = json.loads(text)
        if isinstance(val, list):
            return val
    except json.JSONDecodeError:
        pass
    # Recover the outermost [...] block.
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            val = json.loads(text[start : end + 1])
            if isinstance(val, list):
                return val
        except json.JSONDecodeError:
            return None
    return None


def _batch_value(value: object) -> str | None:
    """Accept one keyed batch value, or None if it is unusable.

    The model occasionally shifts keys and answers a real line with the
    end-of-batch sentinel; treating that as a translation stamps the raw token
    onto the page. Rejecting it sends the line down the per-line path instead.
    """
    if not isinstance(value, (str, int, float)):
        return None
    text = str(value)
    return None if _SENTINEL in text else text


def _clean_single(raw: str) -> str:
    """Reduce a per-line reply to bare text, or "" if nothing usable remains.

    Returning "" makes the caller keep the source line, which is always better
    than stamping protocol debris onto the page.
    """
    text = _FENCE_RE.sub("", raw or "").strip()

    # A single string wrapped in a 1-element array.
    if text.startswith("[") and text.endswith("]"):
        arr = _extract_json_array(text)
        return str(arr[0]).strip() if len(arr or []) == 1 else ""

    # A JSON object: only unambiguous when it holds exactly one value.
    if text.startswith("{"):
        obj = _extract_json_object(text)
        if obj is None or len(obj) != 1:
            return ""
        only = next(iter(obj.values()))
        if not isinstance(only, (str, int, float)):
            return ""
        text = str(only).strip()

    text = text.strip().strip('"').strip()
    return "" if _SENTINEL in text else text
