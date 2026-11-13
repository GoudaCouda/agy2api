import sys
from agy2api.tui import fuzzy_score, run_model_manager_tui
import agy2api.tui as tui


def test_fuzzy_score():
    # Exact match has highest score
    assert fuzzy_score("gpt-4o", "gpt-4o") > fuzzy_score("gpt-4o", "openai/gpt-4o")
    # Substring matches
    assert fuzzy_score("sonnet", "anthropic/claude-3.5-sonnet") > 0
    # Multiple tokens
    assert fuzzy_score("claude sonnet", "anthropic/claude-3.5-sonnet") > 0
    # Subsequence match
    assert fuzzy_score("c35s", "claude-3-5-sonnet") > 0
    # Non-match returns 0
    assert fuzzy_score("xyz123", "openai/gpt-4o") == 0


def test_non_interactive_fallback():
    upstream = ["model-1", "model-2", "model-3"]
    # In non-interactive environment (e.g. pytest without tty), it returns initial_selected directly
    res = run_model_manager_tui(upstream, initial_selected=["model-1", "model-2"])
    assert "model-1" in res
    assert "model-2" in res


def test_interactive_tui_keystrokes(monkeypatch):
    # Simulated keystrokes:
    # 1. Type 's', 'o', 'n' -> matches 'anthropic/claude-3.5-sonnet'
    # 2. Press ENTER -> adds sonnet to selected
    # 3. Press TAB -> switches to selected list
    # 4. Press DELETE -> deletes slot 0 ('initial-model')
    # 5. Press ESC -> saves and exits
    keys = iter(["s", "o", "n", "ENTER", "TAB", "DELETE", "ESC"])
    monkeypatch.setattr(tui, "get_key", lambda: next(keys))

    class MockIO:
        def isatty(self):
            return True
        def write(self, s):
            pass
        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", MockIO())
    monkeypatch.setattr(sys, "stdin", MockIO())

    models = ["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "deepseek/deepseek-r1"]
    res = run_model_manager_tui(models, initial_selected=["initial-model"])

    assert "anthropic/claude-3.5-sonnet" in res
    assert "initial-model" not in res
    assert len(res) == 1


def test_interactive_tui_enforces_32_limit(monkeypatch):
    # Try adding a 33rd model
    initial = [f"model-{i}" for i in range(32)]
    keys = iter(["g", "p", "t", "ENTER", "ESC"])
    monkeypatch.setattr(tui, "get_key", lambda: next(keys))

    class MockIO:
        def isatty(self):
            return True
        def write(self, s):
            pass
        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", MockIO())
    monkeypatch.setattr(sys, "stdin", MockIO())

    models = ["openai/gpt-4o"] + initial
    res = run_model_manager_tui(models, initial_selected=initial)

    assert len(res) == 32
    assert "openai/gpt-4o" not in res
