import json
import os
import pytest
import httpx
from pathlib import Path
from agy2api.config import Config
from agy2api.server import create_app
from agy2api.daemon import DaemonManager

@pytest.mark.asyncio
async def test_health_and_shutdown_endpoints():
    config = Config(
        base_url="https://api.test-provider.com/v1",
        api_key="sk-test-key",
        default_model="test-model"
    )
    app = create_app(config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"

def test_daemon_state_file_management(tmp_path):
    state_file = tmp_path / "daemon.json"
    mgr = DaemonManager(port=4040, state_file=state_file)

    assert not mgr.is_running()

    mgr.save_state(pid=99999, port=4040)
    assert state_file.is_file()
    state = mgr.read_state()
    assert state["pid"] == 99999
    assert state["port"] == 4040

    mgr.clear_state()
    assert not state_file.is_file()
