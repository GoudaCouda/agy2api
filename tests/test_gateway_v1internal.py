import json
import pytest
import httpx
from agy2api.config import Config
from agy2api.server import create_app


@pytest.fixture
def mock_config():
    return Config(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test-key",
        default_model="anthropic/claude-3.7-sonnet",
        port=4040,
    )


@pytest.mark.asyncio
async def test_v1internal_list_experiments_unlocks_models(mock_config):
    app = create_app(mock_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1internal:listExperiments", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "flags" in data
        flag_map = {f["name"]: f.get("boolValue") for f in data["flags"]}
        assert flag_map.get("show-all-models-in-gemini-app") is True
        assert flag_map.get("SDLCAgents__enable_rest_model_connection") is True
        assert flag_map.get("disable-teamwork-forced-flash-model") is True


@pytest.mark.asyncio
async def test_v1internal_fetch_available_models(mock_config):
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": "anthropic/claude-3.7-sonnet",
                            "name": "Claude 3.7 Sonnet",
                            "context_length": 200000,
                        },
                        {
                            "id": "deepseek/deepseek-r1",
                            "name": "DeepSeek R1",
                            "context_length": 128000,
                        },
                    ]
                },
            )
        return httpx.Response(404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(mock_config, http_client=mock_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1internal:fetchAvailableModels", json={"project": "aicode-consumers"})
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert "anthropic/claude-3.7-sonnet" in data["models"]
        assert "deepseek/deepseek-r1" in data["models"]
        assert "agentModelSorts" in data
        sort_groups = data["agentModelSorts"][0]["groups"][0]["modelIds"]
        assert "anthropic/claude-3.7-sonnet" in sort_groups


@pytest.mark.asyncio
async def test_v1internal_load_code_assist_and_quota(mock_config):
    app = create_app(mock_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1internal:loadCodeAssist", json={})
        assert resp.status_code == 200
        assert resp.json().get("currentTier") == "standard-tier"

        resp_quota = await client.post("/v1internal:retrieveUserQuota", json={})
        assert resp_quota.status_code == 200
        assert "buckets" in resp_quota.json()


@pytest.mark.asyncio
async def test_v1internal_stream_generate_content_wrapped(mock_config):
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "deepseek/deepseek-r1"
        assert body["messages"][-1]["content"] == "Hello from CloudCode"

        async def sse_gen():
            yield b'data: {"choices":[{"index":0,"delta":{"content":"Hi there!"},"finish_reason":"stop"}]}\n\n'
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=sse_gen())

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(mock_config, http_client=mock_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        cloudcode_payload = {
            "project": "aicode-consumers",
            "model": "models/deepseek/deepseek-r1",
            "request": {
                "contents": [{"parts": [{"text": "Hello from CloudCode"}]}],
            },
        }
        resp = await client.post("/v1internal:streamGenerateContent?alt=sse", json=cloudcode_payload)
        assert resp.status_code == 200
        lines = [line for line in resp.text.split("\n") if line.startswith("data: ")]
        assert len(lines) >= 1
        frame = json.loads(lines[0][len("data: "):])
        assert frame["candidates"][0]["content"]["parts"][0]["text"] == "Hi there!"
