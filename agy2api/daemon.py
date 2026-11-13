import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional
import httpx
from agy2api.config import Config, load_config

def default_state_path() -> Path:
    return Path.home() / ".gemini" / "antigravity-cli" / "agy2api.state.json"

class DaemonManager:
    def __init__(self, port: int = 4040, state_file: Optional[Path] = None):
        self.port = port
        self.state_file = state_file or default_state_path()

    def read_state(self) -> Optional[Dict[str, Any]]:
        if self.state_file.is_file():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def save_state(self, pid: int, port: int) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pid": pid,
            "port": port,
            "started_at": time.time(),
        }
        self.state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def clear_state(self) -> None:
        if self.state_file.is_file():
            try:
                self.state_file.unlink()
            except Exception:
                pass

    def check_health(self, timeout: float = 1.0) -> bool:
        try:
            resp = httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=timeout)
            return resp.status_code == 200 and resp.json().get("status") == "ok"
        except Exception:
            return False

    def is_running(self) -> bool:
        state = self.read_state()
        if not state:
            return False
        port = state.get("port", self.port)
        self.port = port
        if self.check_health(timeout=1.0):
            return True
        # If health check fails, clean stale state file
        self.clear_state()
        return False

    def start(self, config: Optional[Config] = None, settings_path: Optional[Path] = None) -> bool:
        if self.is_running():
            return True

        cfg = config or load_config(settings_path=settings_path)
        self.port = cfg.port

        py_exe = sys.executable
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

        cmd = [
            py_exe,
            "-u",
            "-m",
            "agy2api.daemon",
            "--serve",
            "--port",
            str(cfg.port),
        ]
        if settings_path:
            cmd.extend(["--settings", str(settings_path)])

        proc = subprocess.Popen(
            cmd,
            creationflags=flags,
            close_fds=(sys.platform != "win32"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

        self.save_state(pid=proc.pid, port=cfg.port)

        # Wait for health check
        start_time = time.time()
        while time.time() - start_time < 6.0:
            if self.check_health(timeout=0.5):
                return True
            time.sleep(0.2)

        return False

    def stop(self) -> bool:
        state = self.read_state()
        port = state.get("port", self.port) if state else self.port
        stopped = False
        try:
            resp = httpx.post(f"http://127.0.0.1:{port}/_shutdown", timeout=2.0)
            if resp.status_code == 200:
                stopped = True
        except Exception:
            pass

        # Check PID if still running on Windows/Unix
        if state and "pid" in state:
            pid = state["pid"]
            try:
                import signal
                os.kill(pid, signal.SIGTERM)
                stopped = True
            except Exception:
                pass

        # Wait briefly for process to terminate and port to free
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if not self.check_health(timeout=0.2):
                break
            time.sleep(0.1)

        self.clear_state()
        return stopped

def run_server_cli():
    parser = argparse.ArgumentParser(description="agy2api background daemon")
    parser.add_argument("--serve", action="store_true", help="Run the HTTP server")
    parser.add_argument("--port", type=int, default=4040, help="Port to bind")
    parser.add_argument("--settings", type=str, default=None, help="Path to settings.json")
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        from agy2api.server import create_app
        settings_p = Path(args.settings) if args.settings else None
        cfg = load_config(settings_path=settings_p)
        cfg.port = args.port
        app = create_app(cfg)
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")

if __name__ == "__main__":
    run_server_cli()
