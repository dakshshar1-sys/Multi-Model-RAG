"""OLLAMA_TEMPERATURE is an opt-in evaluation setting: unset must change nothing."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.llm_provider import _sampling_overrides


def test_unset_or_blank_changes_nothing(monkeypatch):
    monkeypatch.delenv("OLLAMA_TEMPERATURE", raising=False)
    assert _sampling_overrides() == {}
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "  ")
    assert _sampling_overrides() == {}


def test_zero_means_greedy_and_garbage_is_ignored(monkeypatch):
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0")
    assert _sampling_overrides() == {"temperature": 0.0}
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.35")
    assert _sampling_overrides() == {"temperature": 0.35}
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "-1")
    assert _sampling_overrides() == {"temperature": 0.0}, "clamped, never negative"
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "cold")
    assert _sampling_overrides() == {}
