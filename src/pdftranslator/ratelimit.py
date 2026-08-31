"""Shared concurrency gate that obeys the provider's rate-limit notices.

A 429 is an instruction, not a transient error to retry through. When one worker
is told to back off, every worker waits -- continuing to hammer with the other
threads is what turns a brief limit into a sustained one -- and the gate narrows
how many requests may be in flight, widening again only after sustained success.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

# Headers providers use to say when it is safe to try again, in preference order.
_RETRY_HEADERS = (
    "retry-after",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "x-ratelimit-reset",
)

# Never wait longer than this on one notice, however large the number given.
MAX_BACKOFF_SECONDS = 300.0

# Successful calls needed before the gate widens again.
_SUCCESSES_TO_WIDEN = 20


def retry_after_seconds(error: object) -> float | None:
    """How long the provider asked us to wait, or None if it did not say.

    Accepts both forms the Retry-After header takes: a number of seconds, or an
    HTTP date.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    for name in _RETRY_HEADERS:
        try:
            raw = headers.get(name)
        except Exception:  # noqa: BLE001 - odd header mapping
            raw = None
        if not raw:
            continue
        seconds = _parse_delay(str(raw))
        if seconds is not None:
            return max(0.0, min(seconds, MAX_BACKOFF_SECONDS))
    return None


def _parse_delay(raw: str) -> float | None:
    raw = raw.strip()
    try:
        return float(raw)
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    return when.timestamp() - time.time()


class RateLimitGate:
    """Bounds in-flight requests and pauses them all when a limit is hit."""

    def __init__(self, max_concurrency: int) -> None:
        self._max = max(1, int(max_concurrency))
        self._target = self._max
        self._active = 0
        self._successes = 0
        self._paused_until = 0.0
        self._lock = threading.Lock()
        self._free = threading.Condition(self._lock)

    def set_max(self, max_concurrency: int) -> None:
        """Set the ceiling. The orchestrator owns how many workers there are,
        so it tells the gate rather than the two guessing separately."""
        with self._free:
            self._max = max(1, int(max_concurrency))
            self._target = self._max
            self._successes = 0
            self._free.notify_all()

    @property
    def target(self) -> int:
        with self._lock:
            return self._target

    @contextmanager
    def slot(self):
        """Hold one of the permitted in-flight slots."""
        self._acquire()
        try:
            yield
        finally:
            self._release()

    def _acquire(self) -> None:
        while True:
            with self._free:
                now = time.monotonic()
                wait = self._paused_until - now
                if wait <= 0 and self._active < self._target:
                    self._active += 1
                    return
                self._free.wait(timeout=max(0.01, min(wait, 0.5)) if wait > 0 else 0.5)

    def _release(self) -> None:
        with self._free:
            self._active -= 1
            self._free.notify_all()

    def penalise(self, seconds: float | None) -> None:
        """Record a rate-limit notice: hold every worker, and narrow the gate."""
        delay = MAX_BACKOFF_SECONDS if seconds is None else min(
            max(0.0, seconds), MAX_BACKOFF_SECONDS
        )
        with self._free:
            self._paused_until = max(self._paused_until, time.monotonic() + delay)
            self._successes = 0
            if self._target > 1:
                self._target -= 1
                logger.warning(
                    "Rate limited; pausing %.1fs and reducing concurrency to %d",
                    delay, self._target,
                )
            else:
                logger.warning("Rate limited; pausing %.1fs", delay)
            self._free.notify_all()

    def record_success(self) -> None:
        """Widen again once the provider has been comfortable for a while."""
        with self._free:
            if self._target >= self._max:
                return
            self._successes += 1
            if self._successes >= _SUCCESSES_TO_WIDEN:
                self._successes = 0
                self._target += 1
                logger.info("Rate limit eased; concurrency back to %d", self._target)
                self._free.notify_all()
