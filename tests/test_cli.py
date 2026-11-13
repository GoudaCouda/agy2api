import os
import sys
from pathlib import Path
import pytest
from agy2api.config import Config
from agy2api.cli import main, find_agy_binary, find_ide_binary

def test_find_binaries():
    # On this machine, agy.exe and Antigravity IDE.exe exist
    agy = find_agy_binary()
    assert agy is not None
    assert agy.is_file()

    ide = find_ide_binary()
    assert ide is not None
    assert ide.is_file()

def test_cli_status_and_stop(capsys, monkeypatch, tmp_path):
    state_file = tmp_path / "daemon.json"
    monkeypatch.setattr("agy2api.daemon.default_state_path", lambda: state_file)

    # Test status when not running
    exit_code = main(["--status"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "not running" in captured.lower()

    # Test stop when not running
    exit_code = main(["--stop"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "stopped" in captured.lower() or "not running" in captured.lower()

def test_cli_forwarding_invocation(tmp_path, monkeypatch):
    # Mock agy binary with a script that echos args and environment
    script_path = tmp_path / "mock_agy.bat" if sys.platform == "win32" else tmp_path / "mock_agy.sh"
    if sys.platform == "win32":
        script_path.write_text("@echo off\necho MOCK_AGY_CALLED %*\necho GATEWAY=%AGY_LLM_GATEWAY_URL%\nexit /b 42", encoding="utf-8")
    else:
        script_path.write_text("#!/bin/sh\necho MOCK_AGY_CALLED \"$@\"\necho GATEWAY=$AGY_LLM_GATEWAY_URL\nexit 42", encoding="utf-8")
        script_path.chmod(0o755)

    monkeypatch.setattr("agy2api.cli.find_agy_binary", lambda: script_path)

    # Mock daemon to avoid spawning a background server during test
    class MockDaemon:
        def __init__(self, *args, **kwargs):
            self.port = 4040
        def is_running(self):
            return True
        def start(self, *args, **kwargs):
            return True

    monkeypatch.setattr("agy2api.cli.DaemonManager", MockDaemon)

    # Provide configured settings
    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"openai": {"baseUrl": "https://api.test.com/v1", "apiKey": "sk-test", "defaultModel": "gpt-4o"}}', encoding="utf-8")
    monkeypatch.setattr("agy2api.config.default_settings_path", lambda: settings_file)

    exit_code = main(["test-arg", "--flag"])
    assert exit_code == 42


def test_cli_help(capsys):
    exit_code = main(["--help"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "agy2api" in captured
    assert "--model" in captured
    assert "--ide" in captured
    assert "--app" in captured


def test_cli_app_launch(tmp_path, monkeypatch, capsys):
    mock_app = tmp_path / "Antigravity.exe"
    mock_app.write_text("dummy binary", encoding="utf-8")
    monkeypatch.setattr("agy2api.cli.find_app_binary", lambda: mock_app)

    class MockDaemon:
        def __init__(self, *args, **kwargs):
            self.port = 4040
        def is_running(self):
            return True
        def start(self, *args, **kwargs):
            return True

    monkeypatch.setattr("agy2api.cli.DaemonManager", MockDaemon)
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: None)

    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"openai": {"baseUrl": "https://api.test.com/v1", "apiKey": "sk-test", "defaultModel": "gpt-4o"}}', encoding="utf-8")
    monkeypatch.setattr("agy2api.config.default_settings_path", lambda: settings_file)

    exit_code = main(["--app"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Launched Antigravity App" in captured


def test_cli_model_override(tmp_path, monkeypatch):
    script_path = tmp_path / "mock_agy.bat" if sys.platform == "win32" else tmp_path / "mock_agy.sh"
    if sys.platform == "win32":
        script_path.write_text("@echo off\necho MODEL=%ANTIGRAVITY_DEFAULT_MODEL%\necho ARGS=%*\nexit /b 0", encoding="utf-8")
    else:
        script_path.write_text("#!/bin/sh\necho MODEL=$ANTIGRAVITY_DEFAULT_MODEL\necho ARGS=\"$@\"\nexit 0", encoding="utf-8")
        script_path.chmod(0o755)

    monkeypatch.setattr("agy2api.cli.find_agy_binary", lambda: script_path)

    class MockDaemon:
        def __init__(self, *args, **kwargs):
            self.port = 4040
        def is_running(self):
            return True
        def start(self, *args, **kwargs):
            return True

    monkeypatch.setattr("agy2api.cli.DaemonManager", MockDaemon)

    settings_file = tmp_path / "settings.json"
    settings_file.write_text('{"openai": {"baseUrl": "https://api.test.com/v1", "apiKey": "sk-test", "defaultModel": "gpt-4o"}}', encoding="utf-8")
    monkeypatch.setattr("agy2api.config.default_settings_path", lambda: settings_file)

    exit_code = main(["--model", "deepseek/deepseek-r1", "do-work"])
    assert exit_code == 0
