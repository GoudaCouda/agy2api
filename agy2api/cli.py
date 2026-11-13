import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional
from agy2api.config import interactive_setup, load_config
from agy2api.daemon import DaemonManager

def find_agy_binary() -> Optional[Path]:
    which_path = shutil.which("agy")
    if which_path:
        return Path(which_path)

    if sys.platform == "win32":
        local_app_data = os.getenv("LOCALAPPDATA")
        candidates = [
            Path(local_app_data) / "agy" / "bin" / "agy.exe" if local_app_data else None,
            Path.home() / "AppData" / "Local" / "agy" / "bin" / "agy.exe",
            Path.home() / ".gemini" / "antigravity-cli" / "bin" / "agy.exe",
        ]
        for candidate in candidates:
            if candidate and candidate.is_file():
                return candidate
    else:
        candidates = [
            Path.home() / ".local" / "bin" / "agy",
            Path.home() / ".gemini" / "antigravity-cli" / "bin" / "agy",
            Path("/usr/local/bin/agy"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    return None

def find_ide_binary() -> Optional[Path]:
    if sys.platform == "win32":
        local_app_data = os.getenv("LOCALAPPDATA")
        candidates = [
            Path(local_app_data) / "Programs" / "Antigravity IDE" / "Antigravity IDE.exe" if local_app_data else None,
            Path.home() / "AppData" / "Local" / "Programs" / "Antigravity IDE" / "Antigravity IDE.exe",
            Path.home() / "AppData" / "Local" / "Programs" / "Antigravity IDE" / "bin" / "antigravity-ide.cmd",
        ]
        for candidate in candidates:
            if candidate and candidate.is_file():
                return candidate

    which_path = shutil.which("antigravity-ide")
    if which_path:
        return Path(which_path)
    elif sys.platform == "darwin":
        candidates = [
            Path("/Applications/Antigravity IDE.app/Contents/MacOS/Antigravity IDE"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    else:
        candidates = [
            Path("/usr/bin/antigravity-ide"),
            Path("/usr/local/bin/antigravity-ide"),
            Path.home() / ".local" / "bin" / "antigravity-ide",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    return None

def find_app_binary() -> Optional[Path]:
    if sys.platform == "win32":
        local_app_data = os.getenv("LOCALAPPDATA")
        candidates = [
            Path(local_app_data) / "Programs" / "antigravity" / "Antigravity.exe" if local_app_data else None,
            Path.home() / "AppData" / "Local" / "Programs" / "antigravity" / "Antigravity.exe",
        ]
        for candidate in candidates:
            if candidate and candidate.is_file():
                return candidate

    which_path = shutil.which("antigravity")
    if which_path:
        return Path(which_path)
    elif sys.platform == "darwin":
        candidates = [
            Path("/Applications/Antigravity.app/Contents/MacOS/Antigravity"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    else:
        candidates = [
            Path("/usr/bin/antigravity"),
            Path("/usr/local/bin/antigravity"),
            Path.home() / ".local" / "bin" / "antigravity",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

    return None

def print_help() -> None:
    print("""agy2api: Run Google Antigravity with any OpenAI-compatible provider (e.g. OpenRouter, DeepSeek, vLLM).

Usage:
  agy2api [options] [-- agy_options...]
  agy2api models [query]

Subcommands:
  models [query]        Search and list available models from upstream provider

Options:
  --model <model_id>    Set or override upstream model for this session
  --model-setup         Open the interactive 32-Model Manager TUI
  --ide                 Launch the Antigravity Desktop IDE (VS Code-based) with the proxy
  --app                 Launch the Antigravity Desktop App (Electron client) with the proxy
  --detached            Run in detached background mode instead of binding to terminal
  --status              Show proxy daemon running status and active configuration
  --stop                Stop running background proxy daemon
  --setup               Run interactive provider setup wizard
  --logs                Display recent lines from proxy log
  --help, -h            Show this help message

All other flags and arguments are forwarded directly to the Antigravity CLI (agy).
""")


def setup_main() -> int:
    try:
        interactive_setup()
        return 0
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


def main(args: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if args is None else list(args)

    if "--help" in argv or "-h" in argv:
        print_help()
        return 0

    cfg = load_config()
    daemon = DaemonManager(port=cfg.port)

    if "--setup" in argv or (argv and argv[0] in ("setup", "agy-setup")):
        try:
            interactive_setup()
            return 0
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1

    if "--model-setup" in argv or (argv and argv[0] in ("model-setup", "models-setup")):
        from agy2api.config import save_config, verify_models_endpoint
        from agy2api.tui import run_model_manager_tui
        if not cfg.is_configured():
            print("[ERROR] Upstream provider credentials not configured yet. Please run 'agy2api --setup' first.", file=sys.stderr)
            return 1

        print(f"Fetching upstream models from {cfg.base_url}...")
        ok, err, discovered_models = verify_models_endpoint(base_url=cfg.base_url, api_key=cfg.api_key)
        if not ok or not discovered_models:
            print(f"[ERROR] Could not fetch models from upstream ({err}).", file=sys.stderr)
            return 1

        selected = run_model_manager_tui(
            upstream_models=discovered_models,
            initial_selected=cfg.selected_models,
            default_model=cfg.default_model,
        )
        cfg.selected_models = selected
        save_config(cfg)
        print(f"[OK] Saved {len(selected)} selected model(s) to settings.")

        # Notify daemon if running
        if daemon.is_running():
            try:
                req_reload = urllib.request.Request(
                    f"http://127.0.0.1:{cfg.port}/_reload_config",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_reload, timeout=1.5) as _:
                    pass
                print("[OK] Updated running proxy daemon with new model list.")
            except Exception:
                pass
        return 0

    # Extract optional --model override
    model_override: Optional[str] = None
    if "--model" in argv:
        idx = argv.index("--model")
        if idx + 1 < len(argv):
            model_override = argv[idx + 1]
            # Keep --model in argv so agy also receives it, or update cfg
            cfg.default_model = model_override

    if "--status" in argv:
        state = daemon.read_state()
        if daemon.is_running():
            print(f"[STATUS] agy2api proxy is RUNNING on port {daemon.port}")
            if state:
                print(f"  PID: {state.get('pid')}")
            print(f"  Upstream: {cfg.base_url}")
            print(f"  Default Model: {cfg.default_model}")
        else:
            print("[STATUS] agy2api proxy is NOT running.")
            print(f"  Configured Upstream: {cfg.base_url}")
            print(f"  Configured Model: {cfg.default_model}")
        return 0

    if "--logs" in argv:
        log_path = Path.home() / ".gemini" / "antigravity-cli" / "proxy.log"
        if log_path.is_file():
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            tail = lines[-50:] if len(lines) > 50 else lines
            print(f"=== Last {len(tail)} lines of {log_path} ===")
            for line in tail:
                print(line)
        else:
            print("[INFO] No proxy log file found yet.")
        return 0

    if "--stop" in argv:
        if daemon.is_running():
            daemon.stop()
            print("[OK] agy2api proxy daemon stopped.")
        else:
            print("[INFO] agy2api proxy was not running.")
        return 0

    if argv and argv[0] == "models":
        import asyncio
        from agy2api.catalog import ModelCatalog
        raw_terms = [a for a in argv[1:] if not a.startswith("-")]
        query = " ".join(raw_terms) if raw_terms else None
        if query:
            print(f"Searching models matching '{query}' from {cfg.base_url}...")
        else:
            print(f"Fetching available models from {cfg.base_url}...")
        catalog = ModelCatalog(cfg)
        models = asyncio.run(catalog.get_models(query=query, force_refresh=True))
        print(f"\n{'MODEL ID':<50} {'DISPLAY NAME':<40} {'CONTEXT'}")
        print("-" * 100)
        for m in models:
            mid = m["name"].removeprefix("models/")
            dname = m.get("displayName") or mid
            ctx = m.get("inputTokenLimit", 0)
            ctx_str = f"{ctx // 1000}k" if ctx >= 1000 else str(ctx)
            print(f"{mid:<50} {dname[:38]:<40} {ctx_str}")
        print(f"\nTotal: {len(models)} models.")
        return 0

    # Ensure configured
    if not cfg.is_configured():
        print("[NOTICE] Upstream OpenAI provider credentials are not yet configured.")
        try:
            cfg = interactive_setup()
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        daemon = DaemonManager(port=cfg.port)

    # Ensure proxy daemon is running
    if daemon.is_running():
        # Verify running daemon is using current upstream base_url
        daemon_restarted = False
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{cfg.port}/health")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                hdata = json.loads(resp.read().decode("utf-8"))
                running_url = (hdata.get("baseUrl") or "").rstrip("/")
                if running_url and running_url != cfg.base_url.rstrip("/"):
                    print(f"[INFO] Upstream endpoint changed ({running_url} -> {cfg.base_url}). Restarting daemon...")
                    daemon.stop()
                    time.sleep(0.3)
                    daemon.start(config=cfg)
                    daemon_restarted = True
                    time.sleep(0.3)
        except Exception:
            pass

    detached_mode = "--detached" in argv
    if detached_mode:
        argv.remove("--detached")

    we_started_proxy = False
    if not daemon.is_running():
        started = daemon.start(config=cfg)
        if not started:
            print("[ERROR] Failed to start local translation proxy server.", file=sys.stderr)
            return 1
        we_started_proxy = not detached_mode
        # Give server a moment to bind if freshly started
        time.sleep(0.3)

    # Synchronize daemon active settings and refresh catalog cache
    try:
        active_model = model_override or cfg.default_model
        if active_model:
            req = urllib.request.Request(
                f"http://127.0.0.1:{cfg.port}/_set_model",
                data=json.dumps({"model": active_model}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=1.0) as _:
                pass

        # Trigger model cache update
        req = urllib.request.Request(
            f"http://127.0.0.1:{cfg.port}/_refresh_models",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=1.5) as _:
            pass
    except Exception:
        pass

    # Prepare environment for child process (Enterprise Gateway Mode)
    child_env = os.environ.copy()
    child_env["AGY_LLM_GATEWAY_URL"] = f"http://127.0.0.1:{cfg.port}"
    child_env["AGY_CLI_DISABLE_SAFETY_FILTERING"] = "true"
    child_env["AGY_CLI_DISABLE_AUTO_UPDATE"] = "1"
    if cfg.default_model:
        child_env["ANTIGRAVITY_DEFAULT_MODEL"] = cfg.default_model

    # Remove consumer Gemini API keys/endpoints that force agy into flat 32-model consumer mode
    child_env.pop("GOOGLE_GEMINI_BASE_URL", None)
    child_env.pop("GEMINI_API_KEY", None)
    child_env.pop("GOOGLE_API_KEY", None)

    # Launch IDE if --ide passed
    if "--ide" in argv:
        argv.remove("--ide")
        ide_path = find_ide_binary()
        if not ide_path:
            print("[ERROR] Could not find Antigravity IDE installation on this system.", file=sys.stderr)
            if we_started_proxy:
                daemon.stop()
            return 1

        flags = 0
        if detached_mode and sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            [str(ide_path)] + argv,
            env=child_env,
            creationflags=flags,
            close_fds=(sys.platform != "win32"),
        )
        print(f"[OK] Launched Antigravity IDE with proxy at http://127.0.0.1:{cfg.port}")

        if not detached_mode and proc:
            print("Proxy is bound to this terminal session. Closing the window or pressing Ctrl+C will stop the proxy.")
            try:
                proc.wait()
            except KeyboardInterrupt:
                pass
            finally:
                if we_started_proxy:
                    daemon.stop()
                    print("[OK] Proxy stopped.")
        return 0

    # Launch Desktop App if --app passed
    if "--app" in argv:
        argv.remove("--app")
        app_path = find_app_binary()
        if not app_path:
            print("[ERROR] Could not find Antigravity App installation on this system.", file=sys.stderr)
            if we_started_proxy:
                daemon.stop()
            return 1

        flags = 0
        if detached_mode and sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

        proc = subprocess.Popen(
            [str(app_path)] + argv,
            env=child_env,
            creationflags=flags,
            close_fds=(sys.platform != "win32"),
        )
        print(f"[OK] Launched Antigravity App with proxy at http://127.0.0.1:{cfg.port}")

        if not detached_mode and proc:
            print("Proxy is bound to this terminal session. Closing the window or pressing Ctrl+C will stop the proxy.")
            try:
                proc.wait()
            except KeyboardInterrupt:
                pass
            finally:
                if we_started_proxy:
                    daemon.stop()
                    print("[OK] Proxy stopped.")
        return 0

    # Forward to agy CLI
    agy_path = find_agy_binary()
    if not agy_path:
        print("[ERROR] Could not find 'agy' executable on this system.", file=sys.stderr)
        if we_started_proxy:
            daemon.stop()
        return 1

    cli_args = list(argv)
    try:
        proc = subprocess.run([str(agy_path)] + cli_args, env=child_env)
        return proc.returncode
    finally:
        if we_started_proxy:
            daemon.stop()

if __name__ == "__main__":
    sys.exit(main())
