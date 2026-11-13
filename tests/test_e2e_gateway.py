import json
import pytest
import httpx
from agy2api.config import Config
from agy2api.server import create_app


@pytest.fixture
def test_config():
    return Config(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test-key",
        default_model="meta-llama/llama-3.3-70b-instruct",
        port=4040,
    )


@pytest.mark.asyncio
async def test_e2e_openrouter_catalog_and_selection(test_config):
    # Mock OpenRouter /models and /chat/completions
    recorded_requests = []

    def mock_handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        if request.url.path.endswith("/models"):
            return httpx.Response(
                status_code=200,
                json={
                    "data": [
                        {
                            "id": "anthropic/claude-3.7-sonnet",
                            "name": "Anthropic: Claude 3.7 Sonnet",
                            "context_length": 200000,
                        },
                        {
                            "id": "deepseek/deepseek-r1",
                            "name": "DeepSeek: R1 (Reasoning)",
                            "context_length": 128000,
                        },
                    ]
                },
            )
        elif request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            model_used = body["model"]
            return httpx.Response(
                status_code=200,
                json={
                    "id": "gen-123",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Response from {model_used}",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )
        return httpx.Response(status_code=404)

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(test_config, http_client=mock_client)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Test /v1beta/models returns OpenRouter models
        models_resp = await client.get("/v1beta/models")
        assert models_resp.status_code == 200
        catalog = models_resp.json()["models"]
        names = [m["name"] for m in catalog]
        assert "models/anthropic/claude-3.7-sonnet" in names
        assert "models/deepseek/deepseek-r1" in names

        # 2. Test request with default model
        gen_resp = await client.post(
            "/v1beta/models/default:generateContent",
            json={"contents": [{"parts": [{"text": "Hello"}]}]},
        )
        assert gen_resp.status_code == 200
        data = gen_resp.json()
        assert "Response from meta-llama/llama-3.3-70b-instruct" in data["candidates"][0]["content"]["parts"][0]["text"]

        # 3. Test request with explicit OpenRouter model
        gen_resp2 = await client.post(
            "/v1beta/models/deepseek/deepseek-r1:generateContent",
            json={"contents": [{"parts": [{"text": "Think deep"}]}]},
        )
        assert gen_resp2.status_code == 200
        data2 = gen_resp2.json()
        assert "Response from deepseek/deepseek-r1" in data2["candidates"][0]["content"]["parts"][0]["text"]

        # 4. Test request with X-Upstream-Model header override
        gen_resp3 = await client.post(
            "/v1beta/models/gemini-3.8-flash:generateContent",
            headers={"x-upstream-model": "anthropic/claude-3.7-sonnet"},
            json={"contents": [{"parts": [{"text": "Claude test"}]}]},
        )
        assert gen_resp3.status_code == 200
        data3 = gen_resp3.json()
        assert "Response from anthropic/claude-3.7-sonnet" in data3["candidates"][0]["content"]["parts"][0]["text"]


@pytest.mark.asyncio
async def test_e2e_streaming_and_tool_calls(test_config):
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        assert "tools" in body
        assert body["tools"][0]["function"]["name"] == "execute_command"

        async def sse_gen():
            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_01",
                                        "type": "function",
                                        "function": {"name": "execute_command", "arguments": "{\"cmd\": "},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": "\"dir\"}"},
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ]
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_gen(),
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(test_config, http_client=mock_client)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "contents": [{"parts": [{"text": "run dir"}]}],
            "tools": [{
                "functionDeclarations": [{
                    "name": "execute_command",
                    "description": "Run shell command",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {"cmd": {"type": "STRING"}},
                        "required": ["cmd"],
                    },
                }]
            }],
        }
        resp = await client.post(
            "/v1beta/models/deepseek/deepseek-chat:streamGenerateContent?alt=sse",
            json=payload,
        )
        assert resp.status_code == 200
        lines = [line for line in resp.text.split("\n") if line.startswith("data: ")]
        assert len(lines) >= 1
        last_frame = json.loads(lines[-1][len("data: "):])
        fn_call = last_frame["candidates"][0]["content"]["parts"][0]["functionCall"]
        assert fn_call["name"] == "execute_command"
        assert fn_call["args"] == {"cmd": "dir"}


@pytest.mark.asyncio
async def test_e2e_dynamic_model_switch_endpoint(test_config):
    app = create_app(test_config)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Check initial default
        health = await client.get("/health")
        assert health.json()["defaultModel"] == "meta-llama/llama-3.3-70b-instruct"

        # Update model via _set_model
        set_resp = await client.post("/_set_model", json={"model": "qwen/qwen-2.5-coder-32b"})
        assert set_resp.status_code == 200
        assert set_resp.json()["defaultModel"] == "qwen/qwen-2.5-coder-32b"

        # Check health reflects new default
        health_after = await client.get("/health")
        assert health_after.json()["defaultModel"] == "qwen/qwen-2.5-coder-32b"


@pytest.mark.asyncio
async def test_stream_upstream_error_inline(test_config):
    # Simulate an expired API key returning 401
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=401,
            json={"error": {"code": 401, "message": "Invalid API key or expired token."}},
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(test_config, http_client=mock_client)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1beta/models/deepseek/deepseek-chat:streamGenerateContent?alt=sse",
            json={"contents": [{"parts": [{"text": "Hello"}]}]},
        )
        assert resp.status_code == 200
        # The error must be formatted as candidate model text so agy displays it inline!
        lines = [line for line in resp.text.split("\n") if line.startswith("data: ")]
        assert len(lines) >= 1
        frame = json.loads(lines[0][len("data: "):])
        assert "candidates" in frame
        text = frame["candidates"][0]["content"]["parts"][0]["text"]
        assert "Upstream Error" in text
        assert "Invalid API key or expired token" in text
