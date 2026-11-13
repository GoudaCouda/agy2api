import json
import pytest
import httpx
from agy2api.config import Config
from agy2api.server import create_app

@pytest.mark.asyncio
async def test_stream_generate_content():
    config = Config(
        base_url="https://api.test-provider.com/v1",
        api_key="sk-test-key",
        default_model="test-model"
    )

    # Mock handler that simulates OpenAI SSE stream
    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.test-provider.com/v1/chat/completions"
        import json
        body = json.loads(request.content)
        assert body["stream"] is True

        async def sse_generator():
            chunks = [
                {"choices": [{"delta": {"content": "Hello "}, "index": 0, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "world!"}, "index": 0, "finish_reason": "stop"}]}
            ]
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_generator()
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(config, http_client=mock_client)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        gemini_payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Say hello"}]}
            ]
        }
        res = await client.post("/v1beta/models/test-model:streamGenerateContent?alt=sse", json=gemini_payload)
        assert res.status_code == 200
        assert "text/event-stream" in res.headers["content-type"]

        body = res.text
        lines = [line for line in body.split("\n") if line.startswith("data: ")]
        assert len(lines) == 2

        frame1 = json.loads(lines[0][len("data: "):])
        assert frame1["candidates"][0]["content"]["parts"][0]["text"] == "Hello "

        frame2 = json.loads(lines[1][len("data: "):])
        assert frame2["candidates"][0]["content"]["parts"][0]["text"] == "world!"
        assert frame2["candidates"][0]["finishReason"] == "STOP"

@pytest.mark.asyncio
async def test_stream_generate_content_with_tool_call_chunks():
    config = Config(
        base_url="https://api.test-provider.com/v1",
        api_key="sk-test-key",
        default_model="test-model"
    )

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        async def sse_generator():
            chunks = [
                # First chunk introduces tool call with partial json argument
                {
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "id": "call_123",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": "{\"pa"}
                            }]
                        },
                        "finish_reason": None
                    }]
                },
                # Second chunk streams the rest of the arguments
                {
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "tool_calls": [{
                                "index": 0,
                                "function": {"arguments": "th\": \"test.txt\"}"}
                            }]
                        },
                        "finish_reason": "tool_calls"
                    }]
                }
            ]
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_generator()
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(config, http_client=mock_client)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        gemini_payload = {
            "contents": [{"role": "user", "parts": [{"text": "Read test.txt"}]}]
        }
        res = await client.post("/v1beta/models/test-model:streamGenerateContent?alt=sse", json=gemini_payload)
        assert res.status_code == 200

        body = res.text
        lines = [line for line in body.split("\n") if line.startswith("data: ")]
        assert len(lines) == 1

        frame = json.loads(lines[0][len("data: "):])
        cand = frame["candidates"][0]
        assert cand["content"]["role"] == "model"
        parts = cand["content"]["parts"]
        assert len(parts) == 1
        assert "functionCall" in parts[0]
        assert parts[0]["functionCall"]["name"] == "read_file"
        assert parts[0]["functionCall"]["args"] == {"path": "test.txt"}
        assert cand["finishReason"] == "STOP"


@pytest.mark.asyncio
async def test_stream_reasoning_tokens():
    config = Config(
        base_url="https://api.test-provider.com/v1",
        api_key="sk-test-key",
        default_model="test-model"
    )

    def mock_handler(request: httpx.Request) -> httpx.Response:
        async def sse_generator():
            chunks = [
                {
                    "id": "chatcmpl-reason",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"reasoning_content": "Thinking step 1..."},
                        "finish_reason": None
                    }]
                },
                {
                    "id": "chatcmpl-reason",
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "Final answer."},
                        "finish_reason": "stop"
                    }]
                }
            ]
            for c in chunks:
                yield f"data: {json.dumps(c)}\n\n".encode("utf-8")
            yield b"data: [DONE]\n\n"

        return httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=sse_generator()
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(config, http_client=mock_client)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        gemini_payload = {
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}]
        }
        res = await client.post("/v1beta/models/test-model:streamGenerateContent?alt=sse", json=gemini_payload)
        assert res.status_code == 200

        lines = [line for line in res.text.split("\n") if line.startswith("data: ")]
        assert len(lines) == 2

        frame1 = json.loads(lines[0][len("data: "):])
        assert frame1["candidates"][0]["content"]["parts"][0]["thought"] is True
        assert frame1["candidates"][0]["content"]["parts"][0]["text"] == "Thinking step 1..."

        frame2 = json.loads(lines[1][len("data: "):])
        assert frame2["candidates"][0]["content"]["parts"][0]["text"] == "Final answer."
        assert frame2["candidates"][0]["finishReason"] == "STOP"

