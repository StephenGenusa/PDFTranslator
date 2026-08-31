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
import threading
import time

import litellm
from diskcache import Cache

from .ratelimit import RateLimitGate, retry_after_seconds

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


_BATCH_USER_TEMPLATE = (
    "{context}Translate the VALUES of this JSON object into {target_lang}. "
    'Return a JSON object with the exact same keys ("0" through "{last_key}"), '
    "each value replaced by its translation.\n\n{payload}"
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
        ignore_cache: bool = False,
        max_concurrency: int = 1,
    ) -> None:
        self.litellm_model = model  # full litellm model string, e.g. "openrouter/google/gemini-2.5-flash-lite"
        self.target_lang = target_lang
        self.api_key = api_key
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.request_pause = request_pause
        # Skip *reading* the cache; results are still written, so only the pages
        # actually processed are refreshed and the rest of the book is untouched.
        self.ignore_cache = ignore_cache
        self._cache = Cache(cache_dir)
        # Shared by every worker: a rate-limit notice holds all of them.
        self.gate = RateLimitGate(max_concurrency)
        self._stats_lock = threading.Lock()
        self.stats = {
            "pages": 0,
            "api_calls": 0,
            "cache_hits": 0,
            "fallbacks": 0,
            "api_errors": 0,
            "cache_migrated": 0,
            "cache_pruned": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": 0.0,
            "truncated": 0,
            "misaligned": 0,
            "rate_limited": 0,
        }

    # ------------------------------------------------------------------- API
    def translate_page(
        self, lines: list[str], preceding_context: str | None = None
    ) -> list[str]:
        """Translate every line of one page; returns a list of the same length."""
        if not lines:
            return []

        cache_key = self._page_key(lines)
        cached = None if self.ignore_cache else self._lookup(lines, cache_key)
        if isinstance(cached, list) and len(cached) == len(lines):
            self._bump("cache_hits")
            return self._repair_cached(lines, [str(v) for v in cached])

        # Keyed batch: the model returns an object keyed by line index, so a
        # dropped/extra line is realigned by key instead of shifting everything.
        partial, api_failed = self._translate_batch(lines, preceding_context)
        missing = [i for i, r in enumerate(partial) if r is None]
        if missing and api_failed:
            logger.error(
                "Provider unreachable; keeping the source text for %d line(s) "
                "and NOT caching this page.",
                len(missing),
            )
        elif missing:
            logger.warning(
                "Batch left %d/%d line(s) untranslated; filling per line.",
                len(missing), len(lines),
            )
            self._bump("fallbacks")
            page_context = "\n".join(lines)
            for i in missing:
                partial[i] = self._translate_one(lines[i], page_context, i, len(lines))

        # Lines still unresolved fall back to their source text. That is a
        # failure, not a translation, so the page must not be cached: a cached
        # page is never retried, which would make a transient outage permanent.
        unresolved = [i for i, r in enumerate(partial) if r is None]
        result = [r if r is not None else lines[i] for i, r in enumerate(partial)]
        if not api_failed:
            # Skip the repair pass during an outage: every line would look
            # untranslated and fan the same failure out into one call per line.
            result = self._repair_untranslated(lines, result)
        self._bump("pages")

        if unresolved:
            logger.warning(
                "%d/%d line(s) could not be translated; keeping the source text "
                "and NOT caching this page so a later run retries it.",
                len(unresolved), len(lines),
            )
            return result

        self._cache.set(cache_key, result)
        return result

    def _repair_cached(self, lines: list[str], cached: list[str]) -> list[str]:
        """Repair an entry written before the batch-value guards existed.

        Two defects are found in older entries: protocol debris (the end-of-batch
        sentinel), and an empty translation for a line that has source text --
        which renders as nothing at all, leaving the original showing through.
        Falling back to the source line keeps the page consistent at no cost; the
        warning says how to get a real translation for it.
        """
        broken = [
            i for i, value in enumerate(cached)
            if _SENTINEL in value or (lines[i].strip() and not value.strip())
        ]
        if not broken:
            return cached
        logger.warning(
            "Cached translation for %d line(s) is unusable (empty, or carrying "
            "internal batch markers); using the source text instead. Re-run "
            "those pages with --ignore-cache to translate them properly.",
            len(broken),
        )
        for i in broken:
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
    ) -> tuple[list[str | None], bool]:
        """Translate the page as a keyed object; returns (entries, api_failed).

        Entries the model omitted (or that failed to parse) come back as ``None``
        so the caller can fill just those, rather than discarding the whole page.
        ``api_failed`` says the provider could not be reached at all, which tells
        the caller not to fan the same outage out over one call per line.
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
        user = _BATCH_USER_TEMPLATE.format(
            context=context,
            target_lang=self.target_lang,
            last_key=np_ - 1,
            payload=json.dumps(numbered, ensure_ascii=False),
        )

        best: list[str | None] = [None] * np_
        raw = ""
        for attempt in range(self.max_attempts):
            try:
                raw = self._complete(
                    [
                        {"role": "system", "content": _SYSTEM_BATCH},
                        {"role": "user", "content": user},
                    ]
                )
            except TranslationError as exc:
                logger.error("Batch translation unavailable: %s", exc)
                return best[:n], True
            obj = _extract_json_object(raw)
            if obj:
                current = [_batch_value(obj.get(str(i))) for i in range(np_)]
                # Keep the most complete response across attempts.
                if _misaligned(lines, current[:n]):
                    self._bump("misaligned")
                    logger.warning(
                        "Batch looks misaligned (a translation reused, or badly "
                        "the wrong length for its line); retrying."
                    )
                    if attempt < self.max_attempts - 1:
                        time.sleep(_RETRY_BACKOFF * (attempt + 1))
                        continue
                    # Out of attempts: per-line translation cannot misalign.
                    return [None] * n, False
                if sum(x is not None for x in current) > sum(x is not None for x in best):
                    best = current
                if all(x is not None for x in best[:n]):
                    return best[:n], False
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
        return best[:n], False

    def _translate_one(
        self, line: str, page_context: str, idx: int, total: int
    ) -> str | None:
        """Translate a single line with the whole page as context.

        Returns ``None`` when the line could not be translated, so the caller can
        tell a genuine failure apart from a line that legitimately came back
        unchanged. Lines that are essentially a number/symbol (e.g. a chapter
        number "1") are returned unchanged — asking the model to "translate" them
        in isolation tends to make it hallucinate a heading.
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
            return _clean_single(raw) or None
        except (TranslationError, Exception) as exc:  # noqa: BLE001
            logger.debug("Per-line translation failed (%s)", exc)
            return None

    def _complete(self, messages: list[dict]) -> str:
        """Call the model, retrying transient provider errors.

        A book is hundreds of sequential calls, so a 429 or a timeout is routine.
        Letting it escape used to cost the whole page.
        """
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

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            self._bump("api_calls")
            if self.request_pause:
                time.sleep(self.request_pause)
            try:
                with self.gate.slot():
                    response = litellm.completion(**kwargs)
                self.gate.record_success()
                self._record_usage(response)
                if _finish_reason(response) == "length":
                    # The response hit max_tokens, so its tail is missing. This
                    # is the likeliest cause of the "model drops the last key"
                    # behaviour the sentinel padding works around.
                    self._bump("truncated")
                    logger.warning(
                        "Response was cut off at max_tokens (%d); the tail of "
                        "the page is missing. Raise --max-tokens for this "
                        "document.", self.max_tokens,
                    )
                return response.choices[0].message.content or ""
            except litellm.RateLimitError as exc:
                # Not a transient error: the provider told us to wait. Hold
                # every worker for as long as it asked.
                last_error = exc
                self._bump("api_errors")
                self._bump("rate_limited")
                self.gate.penalise(retry_after_seconds(exc))
            except Exception as exc:  # noqa: BLE001 - any provider error is retryable
                last_error = exc
                self._bump("api_errors")
                logger.warning(
                    "API call failed (attempt %d/%d): %s",
                    attempt + 1, self.max_attempts, exc,
                )
                if attempt < self.max_attempts - 1:
                    time.sleep(_RETRY_BACKOFF * (attempt + 1))

        raise TranslationError(
            f"provider unreachable after {self.max_attempts} attempt(s): {last_error}"
        ) from last_error

    def _bump(self, key: str, amount: float = 1) -> None:
        """Counters are touched from several worker threads."""
        with self._stats_lock:
            self.stats[key] += amount

    def _record_usage(self, response: object) -> None:
        """Accumulate tokens and what the provider says the call cost.

        The charge is read from the response rather than derived from a price
        table, which does not know every model routed through OpenRouter.
        """
        usage = getattr(response, "usage", None)
        if usage is not None:
            self._bump("prompt_tokens",
                       int(getattr(usage, "prompt_tokens", 0) or 0))
            self._bump("completion_tokens",
                       int(getattr(usage, "completion_tokens", 0) or 0))
        cost = (getattr(response, "_hidden_params", None) or {}).get("response_cost")
        if cost is None:
            try:
                cost = litellm.completion_cost(completion_response=response)
            except Exception:  # noqa: BLE001 - unmapped model, no price known
                cost = None
        if cost:
            self._bump("cost_usd", float(cost))

    def _page_key(self, lines: list[str]) -> str:
        """Cache identity for a page.

        Covers the model, the target language, the source text and the prompt
        the translation will be produced under. Whitespace is normalised because
        PyMuPDF 1.28 collapses runs of spaces that 1.27 preserved -- an artifact
        of justified text, not content -- and hashing the raw strings meant a
        library upgrade missed every entry and re-paid for the whole book.

        The prompt is included because a translation only means what the
        instructions that produced it meant; editing one used to reuse the older
        results silently unless somebody bumped _CACHE_VERSION by hand.
        max_tokens is deliberately excluded: a response limit does not change
        what a correct translation is, and keying on it would throw away good
        work every time it was tuned.
        """
        return self._digest(
            f"{_prompt_fingerprint()}|" + "\x1f".join(_normalise(ln) for ln in lines)
        )

    def _superseded_page_keys(self, lines: list[str]) -> list[str]:
        """Keys earlier versions stored this page under, newest scheme first.

        Consulted so that translations already paid for are never orphaned when
        the scheme changes.
        """
        return [
            self._digest("\x1f".join(_normalise(ln) for ln in lines)),  # pre-prompt
            self._digest("\x1f".join(lines)),                           # pre-normalisation
        ]

    def _digest(self, raw: str) -> str:
        return hashlib.sha256(
            f"v{_CACHE_VERSION}|{self.litellm_model}|{self.target_lang}|{raw}".encode("utf-8")
        ).hexdigest()

    def set_max_concurrency(self, max_concurrency: int) -> None:
        """Match the gate to the number of workers that will actually run."""
        self.gate.set_max(max_concurrency)

    def estimate_page_tokens(self, lines: list[str], include_output: bool = True) -> int:
        """Tokens one page is expected to consume, both directions.

        A translation is produced as well as consumed, and comes back at
        roughly the length of the source, so the output is counted too.
        """
        if not lines:
            return 0
        text = _SYSTEM_BATCH + "\n" + json.dumps(
            {str(i): line for i, line in enumerate(lines)}, ensure_ascii=False
        )
        try:
            prompt = int(litellm.token_counter(model=self.litellm_model, text=text))
        except Exception:  # noqa: BLE001 - unknown tokeniser, fall back to chars
            prompt = max(1, len(text) // 4)
        return prompt * 2 if include_output else prompt

    def estimate_cost(self, tokens: int) -> float | None:
        """Dollar cost for a token count, or None when the model has no price.

        Many models routed through OpenRouter are not in litellm's price table.
        Returning None says so plainly rather than inventing a figure; the run
        still reports what the provider actually charged.
        """
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=self.litellm_model,
                prompt_tokens=tokens // 2,
                completion_tokens=tokens // 2,
            )
        except Exception:  # noqa: BLE001 - unmapped model
            return None
        total = (prompt_cost or 0.0) + (completion_cost or 0.0)
        return total or None

    def has_cached(self, lines: list[str]) -> bool:
        """Is a usable translation already stored for this page?

        Consults both key schemes and mutates nothing, so coverage can be
        reported without rewriting the cache. Checking only the current scheme
        is what once made 311 already-translated pages look missing.
        """
        for key in [self._page_key(lines), *self._superseded_page_keys(lines)]:
            value = self._cache.get(key)
            if isinstance(value, list) and len(value) == len(lines):
                return True
        return False

    def cached_looks_misaligned(self, lines: list[str]) -> bool:
        """Does the stored translation for this page look shifted?

        Read-only. Entries written before misalignment detection existed still
        carry the damage, and serving them silently hides it.
        """
        for key in [self._page_key(lines), *self._superseded_page_keys(lines)]:
            value = self._cache.get(key)
            if isinstance(value, list) and len(value) == len(lines):
                return _misaligned(lines, [str(v) for v in value])
        return False

    def _lookup(self, lines: list[str], cache_key: str) -> list | None:
        """Read the cache, converting entries written under an older scheme.

        A cached translation is paid-for work, so an old key is removed only
        after the new one has been written and read back. An entry that cannot
        be re-keyed is never touched -- it may belong to another document.
        """
        superseded = [k for k in self._superseded_page_keys(lines) if k != cache_key]

        cached = self._cache.get(cache_key)
        if cached is not None:
            # An earlier version may have left a duplicate behind. Drop it only
            # because the current key already holds this very translation; if
            # the two disagree that is unexplained, so keep both.
            for old_key in superseded:
                if self._cache.get(old_key) == cached:
                    self._cache.delete(old_key)
                    self._bump("cache_pruned")
            return cached

        for old_key in superseded:
            found = self._cache.get(old_key)
            if found is None:
                continue
            self._cache.set(cache_key, found)
            if self._cache.get(cache_key) == found:
                self._cache.delete(old_key)   # only now is the old copy safe to drop
                self._bump("cache_migrated")
            else:
                logger.warning(
                    "Could not rewrite a cached page under the current key; "
                    "keeping the existing entry."
                )
            return found
        return None

    def close(self) -> None:
        self._cache.close()


def _prompt_fingerprint() -> str:
    """Short digest of the instructions a translation is produced under.

    Read from the module at call time, so editing a prompt takes effect at once
    instead of depending on someone remembering to bump _CACHE_VERSION.
    """
    blob = "\x1e".join((_SYSTEM_BATCH, _SYSTEM_LINE, _BATCH_USER_TEMPLATE))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _finish_reason(response: object) -> str | None:
    choices = getattr(response, "choices", None) or []
    return getattr(choices[0], "finish_reason", None) if choices else None


_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """Collapse whitespace runs so extraction quirks do not change identity."""
    return _WHITESPACE_RE.sub(" ", text).strip()


# A batch is judged misaligned when a translation is reused for two different
# source lines, or a line's translation is wildly the wrong length for it. Only
# lines with enough text are length-checked: numbers and markers legitimately
# change length a great deal.
_ALIGN_MIN_CHARS = 12
_ALIGN_MIN_RATIO = 0.25
_ALIGN_MAX_RATIO = 4.0
_ALIGN_MIN_ODD_LINES = 2
_ALIGN_ODD_SHARE = 0.10


def _misaligned(lines: list[str], values: list[str | None]) -> bool:
    """Do the values look like they landed under the wrong keys?

    The keyed scheme realigns by key, but the model can still fill the wrong
    key. Counts match in that case, so nothing else catches it.
    """
    seen: dict[str, str] = {}
    for source, value in zip(lines, values):
        if value is None or len(value.strip()) <= _ALIGN_MIN_CHARS:
            continue
        # the same translation for two *different* source lines
        previous = seen.setdefault(value.strip(), source)
        if previous != source:
            return True
    # A single odd line is plausible -- a name, a number, an aside. A shifted
    # batch throws several lines out at once, so require more than one before
    # condemning the page and paying for a retry.
    checked = odd = 0
    for source, value in zip(lines, values):
        if value is None:
            continue
        length = len(source.strip())
        if length < _ALIGN_MIN_CHARS:
            continue
        checked += 1
        ratio = len(value.strip()) / length
        if not _ALIGN_MIN_RATIO <= ratio <= _ALIGN_MAX_RATIO:
            odd += 1
    return odd >= _ALIGN_MIN_ODD_LINES and odd >= checked * _ALIGN_ODD_SHARE


# Seconds between transport retries; multiplied by the attempt number.
_RETRY_BACKOFF = 1.5

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

    Two ways a value is unusable. The model occasionally shifts keys and answers
    a real line with the end-of-batch sentinel, which would stamp the raw token
    onto the page. And it sometimes returns an empty string, which is worse than
    it looks: the renderer draws nothing at all for an empty line -- no mask --
    so the untranslated original shows through from the base layer with no
    warning. Rejecting both sends the line down the per-line path instead.
    """
    if not isinstance(value, (str, int, float)):
        return None
    text = str(value)
    if _SENTINEL in text or not text.strip():
        return None
    return text


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
