"""The shared gate that keeps concurrent workers inside the provider's limits.

A rate-limit response is an instruction, not an error to retry through. When one
worker is told to back off, every worker waits: hammering with the other threads
is what turns a brief limit into a sustained one.
"""

from __future__ import annotations

import threading
import time

import pytest

from pdftranslator.ratelimit import RateLimitGate, retry_after_seconds


class _Headers(dict):
    def get(self, key, default=None):
        return super().get(key.lower(), default)


def _error(headers):
    response = type("Response", (), {"headers": _Headers(headers)})()
    return type("RateLimitError", (Exception,), {"response": response,
                                                 "status_code": 429})()


# ------------------------------------------------------- reading the notice
def test_retry_after_in_seconds_is_read():
    assert retry_after_seconds(_error({"retry-after": "7"})) == pytest.approx(7.0)


def test_retry_after_as_an_http_date_is_read():
    from email.utils import formatdate
    when = formatdate(time.time() + 30, usegmt=True)
    seconds = retry_after_seconds(_error({"retry-after": when}))
    assert 20 <= seconds <= 40, seconds


def test_a_reset_header_is_used_when_retry_after_is_absent():
    assert retry_after_seconds(
        _error({"x-ratelimit-reset-requests": "12"})) == pytest.approx(12.0)


def test_no_usable_header_yields_none():
    assert retry_after_seconds(_error({})) is None
    assert retry_after_seconds(Exception("plain")) is None


def test_an_absurd_value_is_capped():
    """A provider asking for an hour should not wedge the run indefinitely."""
    capped = retry_after_seconds(_error({"retry-after": "99999"}))
    assert capped is not None and capped <= 300


# --------------------------------------------------------------- the gate
def test_a_pause_holds_every_worker(monkeypatch):
    """Breaks if a 429 only slows the thread that received it."""
    gate = RateLimitGate(max_concurrency=4)
    gate.penalise(0.25)
    started = time.monotonic()
    results = []

    def worker():
        with gate.slot():
            results.append(time.monotonic() - started)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(results) == 4
    assert min(results) >= 0.2, results


def test_concurrency_is_capped_at_the_configured_maximum():
    gate = RateLimitGate(max_concurrency=2)
    peak = 0
    active = 0
    lock = threading.Lock()

    def worker():
        nonlocal peak, active
        with gate.slot():
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert peak <= 2, peak


def test_being_limited_narrows_concurrency():
    gate = RateLimitGate(max_concurrency=4)
    assert gate.target == 4
    gate.penalise(0.0)
    assert gate.target < 4, "should back off after being rate limited"


def test_concurrency_recovers_after_sustained_success():
    gate = RateLimitGate(max_concurrency=4)
    gate.penalise(0.0)
    narrowed = gate.target
    for _ in range(100):
        gate.record_success()
    assert gate.target > narrowed


def test_concurrency_never_drops_below_one():
    gate = RateLimitGate(max_concurrency=2)
    for _ in range(10):
        gate.penalise(0.0)
    assert gate.target >= 1
