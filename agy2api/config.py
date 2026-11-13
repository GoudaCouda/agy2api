from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple
import httpx

DEFAULT_PORT = 4040
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = ""

def default_settings_path() -> Path:
    root_settings = Path.home() / ".gemini" / "settings.json"
    if root_settings.is_file():
        return root_settings
    legacy_settings = Path.home() / ".gemini" / "antigravity-cli" / "settings.json"
    if legacy_settings.is_file():
        return legacy_settings
    return root_settings

@dataclass
class Config:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    default_model: str = DEFAULT_MODEL
    port: int = DEFAULT_PORT
    selected_models: List[str] = field(default_factory=list)
    configured: bool = False

    def is_configured(self) -> bool:
        return self.configured or bool(self.api_key)

def load_config(settings_path: Path | None = None) -> Config:
    path = settings_path or default_settings_path()
    base_url = DEFAULT_BASE_URL
    api_key = ""
    default_model = DEFAULT_MODEL
    port = DEFAULT_PORT
    selected_models: List[str] = []
    configured = False

    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            # Check agy2api first, fallback to legacy openai section
            section = data.get("agy2api")
            if not isinstance(section, dict):
                section = data.get("openai", {})

            if isinstance(section, dict) and section:
                configured = True
                base_url = section.get("baseUrl", base_url)
                api_key = section.get("apiKey", api_key)
                default_model = section.get("defaultModel", default_model)
                port = int(section.get("port", port))
                raw_selected = section.get("selectedModels") or section.get("models")
                if isinstance(raw_selected, list):
                    selected_models = [str(m).strip() for m in raw_selected if str(m).strip()]
        except Exception:
            pass

    # Env var overrides
    env_base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if env_base:
        base_url = env_base
        configured = True

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        api_key = env_key
        configured = True

    env_model = os.getenv("OPENAI_MODEL_NAME") or os.getenv("OPENAI_MODEL")
    if env_model:
        default_model = env_model

    env_port = os.getenv("AGY2API_PORT") or os.getenv("agy2api_PORT")
    if env_port:
        try:
            port = int(env_port)
        except ValueError:
            pass

    return Config(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        default_model=default_model,
        port=port,
        selected_models=selected_models,
        configured=configured,
    )

def save_config(config: Config, settings_path: Path | None = None) -> None:
    path = settings_path or default_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}

    section = {
        "baseUrl": config.base_url,
        "apiKey": config.api_key,
        "defaultModel": config.default_model,
    }
    if config.port != DEFAULT_PORT:
        section["port"] = config.port

    if config.selected_models:
        section["selectedModels"] = config.selected_models[:32]
    elif "selectedModels" in section:
        del section["selectedModels"]

    data["agy2api"] = section
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def verify_models_endpoint(base_url: str, api_key: str, timeout: float = 6.0) -> Tuple[bool, Optional[str], List[str]]:
    """Test the upstream /models endpoint to verify connectivity and credentials.

    Returns (success, error_or_info_message, list_of_model_ids).
    If the endpoint returns no models, it reports an error.
    """
    url = f"{base_url.rstrip('/')}/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
            if resp.status_code == 200:
                models: List[str] = []
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        items = data.get("data") or data.get("models") or []
                    elif isinstance(data, list):
                        items = data
                    else:
                        items = []

                    for item in items:
                        if isinstance(item, dict):
                            mid = item.get("id") or item.get("name")
                            if mid:
                                models.append(str(mid))
                        elif isinstance(item, str):
                            models.append(item)
                except Exception:
                    pass

                if not models:
                    return False, "Endpoint returned HTTP 200 but no models were found", []

                return True, None, models
            else:
                snippet = resp.text[:150].strip().replace("\n", " ")
                return False, f"HTTP {resp.status_code} ({snippet})", []
    except Exception as exc:
        return False, str(exc), []

def interactive_setup(settings_path: Path | None = None) -> Config:
    current = load_config(settings_path=settings_path)
    print("\n\033[1;36m=== agy2api Setup Wizard ===\033[0m")
    print("\033[90mConfigure your upstream OpenAI-compatible provider.")
    print("(Press Enter to keep current value, or '-' to clear)\033[0m\n")

    base_url_input = input(f"Base URL [{current.base_url}]: ").strip()
    base_url = base_url_input if base_url_input else current.base_url
    base_url = base_url.rstrip("/")

    key_prompt = f"API Key [{'***' if current.api_key else 'optional'}]: "
    api_key_input = input(key_prompt).strip()
    if api_key_input in ("-", "none", "clear", "empty"):
        api_key = ""
    elif api_key_input:
        api_key = api_key_input
    else:
        api_key = current.api_key

    print(f"\n\033[94m[TESTING]\033[0m Verifying endpoint at {base_url}/models ...")
    ok, err, discovered_models = verify_models_endpoint(base_url=base_url, api_key=api_key)
    if not ok:
        print(f"\n\033[1;31m[ERROR]\033[0m Setup verification failed: {err}")
        raise ValueError(f"Setup verification failed: {err}")

    print("\033[1;32m[OK]\033[0m Successfully connected to /models! Setup verified.")
    print(f"\033[1;32m[OK]\033[0m Discovered {len(discovered_models)} available upstream model(s).\n")

    selected_models = list(current.selected_models)

    # Offer to run the 32-Model Manager TUI
    print("\033[1mManage up to 32 models for the Antigravity UI model picker.\033[0m")
    setup_models = input("Would you like to open the Model Manager TUI now? [Y/n]: ").strip().lower()
    if setup_models not in ("n", "no"):
        try:
            from agy2api.tui import run_model_manager_tui
            selected_models = run_model_manager_tui(
                upstream_models=discovered_models,
                initial_selected=selected_models,
            )
        except Exception as e:
            print(f"[WARN] Could not launch TUI ({e}). Continuing with standard setup.")
    elif not selected_models and discovered_models:
        selected_models = discovered_models[:32]

    new_config = Config(
        base_url=base_url,
        api_key=api_key,
        default_model="",
        port=current.port,
        selected_models=selected_models,
        configured=True,
    )
    save_config(new_config, settings_path=settings_path)
    print(f"\n\033[1;32m[OK]\033[0m Configuration saved to {settings_path or default_settings_path()}")

    try:
        from agy2api.daemon import DaemonManager
        daemon = DaemonManager(port=new_config.port)
        if daemon.is_running():
            print("\033[94m[INFO]\033[0m Restarting proxy daemon with new endpoint...")
            daemon.stop()
            daemon.start(config=new_config, settings_path=settings_path)
            print("\033[1;32m[OK]\033[0m Proxy daemon restarted with updated models.")
    except Exception:
        pass

    return new_config
