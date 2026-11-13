import json
import pytest
import httpx
from agy2api.config import Config
from agy2api.server import create_app

@pytest.mark.asyncio
async def test_get_models_list():
    config = Config(
        base_url="https://api.test-provider.com/v1",
        api_key="sk-test-key",
        default_model="deepseek-chat"
    )

    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.test-provider.com/v1/models"
        return httpx.Response(
            status_code=200,
            json={
                "data": [
                    {"id": "deepseek-chat", "object": "model"},
                    {"id": "deepseek-coder", "object": "model"}
                ]
            }
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(config, http_client=mock_client)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        res = await client.get("/v1beta/models")
        assert res.status_code == 200
        data = res.json()
        assert "models" in data
        model_names = [m["name"] for m in data["models"]]
        assert "models/deepseek-chat" in model_names
        assert "models/deepseek-coder" in model_names

        # Single model endpoint
        res_single = await client.get("/v1beta/models/deepseek-chat")
        assert res_single.status_code == 200
        single_data = res_single.json()
        assert single_data["name"] == "models/deepseek-chat"

@pytest.mark.asyncio
async def test_fallback_model_routing():
    config = Config(
        base_url="https://api.test-provider.com/v1",
        api_key="sk-test-key",
        default_model="deepseek-chat"
    )

    captured_model = None

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_model
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            captured_model = body["model"]
            return httpx.Response(
                status_code=200,
                json={
                    "choices": [{"message": {"role": "assistant", "content": "ok"}}]
                }
            )
        return httpx.Response(status_code=404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(config, http_client=mock_client)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Request with a standard Gemini model name -> should fall back to default_model ("deepseek-chat")
        await client.post(
            "/v1beta/models/gemini-2.5-flash:generateContent",
            json={"contents": [{"parts": [{"text": "hi"}]}]}
        )
        assert captured_model == "deepseek-chat"

        # Request with an explicit custom model -> should route to that model
        await client.post(
            "/v1beta/models/qwen-2.5-coder:generateContent",
            json={"contents": [{"parts": [{"text": "hi"}]}]}
        )
        assert captured_model == "qwen-2.5-coder"
