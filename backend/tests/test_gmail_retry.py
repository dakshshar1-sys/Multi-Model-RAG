"""
Gmail availability must recover from a transient start-up failure.

Background: the backend came up during a network blip, the OAuth token refresh
failed, and `available` stayed False for 29 hours because it was computed once in
the constructor. The client now retries on demand, at most once per cooldown, and
only for transient errors (never for "not authorized" / "invalid token").

Run:  cd backend && python -m pytest tests/test_gmail_retry.py -v
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actions.gmail_client import GmailClient


def _client_with(monkeypatch, error, calls):
    """Build a client whose _connect records calls and leaves it in `error` state."""
    def fake_connect(self):
        import time
        calls.append(1)
        self._last_attempt = time.monotonic()
        self.service = None
        self._init_error = error
    monkeypatch.setattr(GmailClient, "_connect", fake_connect)
    return GmailClient(credentials_path="/nonexistent", token_path="/nonexistent")


def test_transient_failure_retries_after_cooldown(monkeypatch):
    calls = []
    c = _client_with(monkeypatch, "Gmail init failed: Network is unreachable", calls)
    assert len(calls) == 1
    assert c.available is False and len(calls) == 1, "no retry inside the cooldown"
    c._last_attempt -= GmailClient.RETRY_COOLDOWN_S + 1
    assert c.available is False and len(calls) == 2, "one retry once the cooldown has passed"
    assert c.available is False and len(calls) == 2, "and not again immediately"


def test_transient_failure_recovers_when_the_retry_succeeds(monkeypatch):
    calls = []
    c = _client_with(monkeypatch, "Gmail init failed: Network is unreachable", calls)
    def good_connect(self):
        calls.append(1)
        self.service = object()
        self._init_error = None
    monkeypatch.setattr(GmailClient, "_connect", good_connect)
    c._last_attempt -= GmailClient.RETRY_COOLDOWN_S + 1
    assert c.available is True
    assert c._init_error is None


def test_not_authorized_never_retries(monkeypatch):
    calls = []
    c = _client_with(monkeypatch, "Gmail is not authorized yet. Run: python -m actions.authorize", calls)
    c._last_attempt -= 10_000
    assert c.available is False and len(calls) == 1
