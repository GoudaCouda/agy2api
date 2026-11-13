import json
import os
import pytest
from pathlib import Path
from agy2api.config import Config, load_config, save_config

def test_load_config_from_settings_json(tmp_path):
    settings_file = tmp_path / "settings.json"
    initial_data = {
        "agentMode": "plan",
        "model": "Gemini 3.8 Flash (Medium)",
        "openai": {
            "baseUrl": "https://api.deepseek.com/v1",
            "apiKey": "sk-test-deepseek-key",
            "defaultModel": "deepseek-chat"
        }
    }
    settings_file.write_text(json.dumps(initial_data), encoding="utf-8")

    config = load_config(settings_path=settings_file)
    assert config.base_url == "https://api.deepseek.com/v1"
    assert config.api_key == "sk-test-deepseek-key"
    assert config.default_model == "deepseek-chat"

def test_load_config_env_var_overrides(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    initial_data = {
        "openai": {
            "baseUrl": "https://file-url.com",
            "apiKey": "sk-file-key",
            "defaultModel": "file-model"
        }
    }
    settings_file.write_text(json.dumps(initial_data), encoding="utf-8")

    monkeypatch.setenv("OPENAI_BASE_URL", "https://env-url.com/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
    monkeypatch.setenv("OPENAI_MODEL_NAME", "env-model")

    config = load_config(settings_path=settings_file)
    assert config.base_url == "https://env-url.com/v1"
    assert config.api_key == "sk-env-key"
    assert config.default_model == "env-model"

def test_save_config_preserves_other_settings(tmp_path):
    settings_file = tmp_path / "settings.json"
    initial_data = {
        "agentMode": "plan",
        "model": "Gemini 3.8 Flash (Medium)",
        "trustedWorkspaces": ["C:\\test"]
    }
    settings_file.write_text(json.dumps(initial_data), encoding="utf-8")

    new_config = Config(
        base_url="https://api.openai.com/v1",
        api_key="sk-new-key",
        default_model="gpt-4o"
    )
    save_config(new_config, settings_path=settings_file)

    saved_data = json.loads(settings_file.read_text(encoding="utf-8"))
    assert saved_data["agentMode"] == "plan"
    assert saved_data["trustedWorkspaces"] == ["C:\\test"]
    assert saved_data["agy2api"] == {
        "baseUrl": "https://api.openai.com/v1",
        "apiKey": "sk-new-key",
        "defaultModel": "gpt-4o"
    }

def test_is_configured_helper():
    incomplete = Config(base_url="https://api.openai.com/v1", api_key="", default_model="")
    assert not incomplete.is_configured()

    complete = Config(base_url="https://api.openai.com/v1", api_key="sk-123", default_model="gpt-4o")
    assert complete.is_configured()

def test_verify_models_endpoint_success(monkeypatch):
    from agy2api.config import verify_models_endpoint
    import httpx

    def fake_get(url, headers, timeout):
        assert url == "https://api.openai.com/v1/models"
        assert headers.get("Authorization") == "Bearer sk-valid"
        return httpx.Response(
            status_code=200,
            json={"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr("httpx.Client.get", lambda self, url, **kwargs: fake_get(url, kwargs.get("headers", {}), kwargs.get("timeout")))
    ok, err, models = verify_models_endpoint("https://api.openai.com/v1", "sk-valid")
    assert ok is True
    assert err is None
    assert models == ["gpt-4o", "gpt-4o-mini"]

def test_verify_models_endpoint_failure(monkeypatch):
    from agy2api.config import verify_models_endpoint
    import httpx

    def fake_get(url, headers, timeout):
        return httpx.Response(
            status_code=401,
            text='{"error": "Invalid API Key"}',
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr("httpx.Client.get", lambda self, url, **kwargs: fake_get(url, kwargs.get("headers", {}), kwargs.get("timeout")))
    ok, err, models = verify_models_endpoint("https://api.openai.com/v1", "sk-bad")
    assert ok is False
    assert "HTTP 401" in err
    assert models == []

def test_verify_models_endpoint_returns_error_on_empty_models(monkeypatch):
    from agy2api.config import verify_models_endpoint
    import httpx

    def fake_get(url, headers, timeout):
        return httpx.Response(
            status_code=200,
            json={"data": [], "object": "list"},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr("httpx.Client.get", lambda self, url, **kwargs: fake_get(url, kwargs.get("headers", {}), kwargs.get("timeout")))
    ok, err, models = verify_models_endpoint("https://api.openai.com/v1", "sk-valid")
    assert ok is False
    assert "no models" in err.lower()
    assert models == []

def test_interactive_setup_fails_when_no_models(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    user_inputs = iter(["https://api.openai.com/v1", "sk-test"])
    monkeypatch.setattr("builtins.input", lambda _: next(user_inputs))

    monkeypatch.setattr("agy2api.config.verify_models_endpoint", lambda base_url, api_key, **kwargs: (False, "Endpoint returned no models", []))

    from agy2api.config import interactive_setup
    with pytest.raises(ValueError, match="no models"):
        interactive_setup(settings_path=settings_file)

def test_interactive_setup(tmp_path, monkeypatch):
    settings_file = tmp_path / "settings.json"
    user_inputs = iter(["https://api.groq.com/openai/v1", "gsk-testkey", "n", "llama-3.3-70b-versatile"])
    monkeypatch.setattr("builtins.input", lambda _: next(user_inputs))

    # Mock verify_models_endpoint to return simulated models
    verified_calls = []
    def fake_verify(base_url, api_key, **kwargs):
        verified_calls.append((base_url, api_key))
        return True, None, ["llama-3.3-70b-versatile"]

    monkeypatch.setattr("agy2api.config.verify_models_endpoint", fake_verify)

    from agy2api.config import interactive_setup
    cfg = interactive_setup(settings_path=settings_file)
    assert verified_calls == [("https://api.groq.com/openai/v1", "gsk-testkey")]
    assert cfg.base_url == "https://api.groq.com/openai/v1"
    assert cfg.api_key == "gsk-testkey"
    assert cfg.default_model == "llama-3.3-70b-versatile"

    # Verify written to file
    saved = load_config(settings_path=settings_file)
    assert saved.base_url == "https://api.groq.com/openai/v1"
    assert saved.api_key == "gsk-testkey"
    assert saved.default_model == "llama-3.3-70b-versatile"


