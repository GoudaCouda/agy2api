import pytest
import httpx
from agy2api.config import Config
from agy2api.server import create_app

@pytest.mark.asyncio
async def test_generate_content_non_streaming():
    config = Config(
        base_url="https://api.test-provider.com/v1",
        api_key="sk-test-key",
        default_model="test-model"
    )

    # Mock handler for the upstream OpenAI endpoint
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.test-provider.com/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-test-key"
        import json
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["messages"] == [
            {"role": "system", "content": "System prompt text."},
            {"role": "user", "content": "Hello assistant"},
            {"role": "assistant", "content": "Hello user"},
            {"role": "user", "content": "How are you?"}
        ]
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 1234567,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "I am doing well, thank you!"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 8,
                    "total_tokens": 23
                }
            }
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(config, http_client=mock_client)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        gemini_payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Hello assistant"}]},
                {"role": "model", "parts": [{"text": "Hello user"}]},
                {"role": "user", "parts": [{"text": "How are you?"}]}
            ],
            "system_instruction": {
                "parts": [{"text": "System prompt text."}]
            }
        }
        res = await client.post("/v1beta/models/test-model:generateContent", json=gemini_payload)
        assert res.status_code == 200
        data = res.json()
        assert "candidates" in data
        assert len(data["candidates"]) == 1
        candidate = data["candidates"][0]
        assert candidate["content"]["role"] == "model"
        assert candidate["content"]["parts"] == [{"text": "I am doing well, thank you!"}]
        assert candidate["finishReason"] == "STOP"
        assert data["usageMetadata"]["promptTokenCount"] == 15
        assert data["usageMetadata"]["candidatesTokenCount"] == 8
        assert data["usageMetadata"]["totalTokenCount"] == 23
