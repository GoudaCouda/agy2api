import json
import pytest
import httpx
from agy2api.config import Config
from agy2api.server import create_app

@pytest.mark.asyncio
async def test_tool_declarations_and_function_call():
    config = Config(
        base_url="https://api.test-provider.com/v1",
        api_key="sk-test-key",
        default_model="test-model"
    )

    def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "tools" in body
        tools = body["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "read_file"
        assert tools[0]["function"]["parameters"]["type"] == "object"

        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-tool",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_abc",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "{\"path\": \"test.txt\"}"
                                    }
                                }
                            ]
                        },
                        "finish_reason": "tool_calls"
                    }
                ]
            }
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(config, http_client=mock_client)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        gemini_payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Read test.txt"}]}
            ],
            "tools": [
                {
                    "functionDeclarations": [
                        {
                            "name": "read_file",
                            "description": "Reads a file",
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {
                                    "path": {"type": "STRING", "description": "File path"}
                                },
                                "required": ["path"]
                            }
                        }
                    ]
                }
            ]
        }
        res = await client.post("/v1beta/models/test-model:generateContent", json=gemini_payload)
        assert res.status_code == 200
        data = res.json()
        assert "candidates" in data
        candidate = data["candidates"][0]
        parts = candidate["content"]["parts"]
        assert len(parts) == 1
        assert "functionCall" in parts[0]
        fn_call = parts[0]["functionCall"]
        assert fn_call["name"] == "read_file"
        assert fn_call["args"] == {"path": "test.txt"}

@pytest.mark.asyncio
async def test_function_response_to_tool_message():
    config = Config(
        base_url="https://api.test-provider.com/v1",
        api_key="sk-test-key",
        default_model="test-model"
    )

    def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        messages = body["messages"]
        # Expect system or user, then assistant tool_call, then tool response
        assert any(m["role"] == "tool" and "file contents" in m["content"] for m in messages)

        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-after-tool",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The file contains: file contents."
                        },
                        "finish_reason": "stop"
                    }
                ]
            }
        )

    mock_client = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))
    app = create_app(config, http_client=mock_client)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        gemini_payload = {
            "contents": [
                {"role": "user", "parts": [{"text": "Read test.txt"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "read_file",
                                "args": {"path": "test.txt"}
                            }
                        }
                    ]
                },
                {
                    "role": "function",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "read_file",
                                "response": {"result": "file contents"}
                            }
                        }
                    ]
                }
            ]
        }
        res = await client.post("/v1beta/models/test-model:generateContent", json=gemini_payload)
        assert res.status_code == 200
        data = res.json()
        assert data["candidates"][0]["content"]["parts"][0]["text"] == "The file contains: file contents."


@pytest.mark.asyncio
async def test_parallel_tool_calls_fifo_mapping():
    from agy2api.translator import gemini_to_openai_messages
    gemini_contents = [
        {"role": "user", "parts": [{"text": "Read two files"}]},
        {
            "role": "model",
            "parts": [
                {"functionCall": {"name": "read_file", "args": {"path": "a.txt"}}},
                {"functionCall": {"name": "read_file", "args": {"path": "b.txt"}}},
            ],
        },
        {
            "role": "user",
            "parts": [
                {"functionResponse": {"name": "read_file", "response": {"result": "content A"}}},
                {"functionResponse": {"name": "read_file", "response": {"result": "content B"}}},
            ],
        },
    ]

    messages = gemini_to_openai_messages(gemini_contents)
    assistant_msg = next(m for m in messages if m["role"] == "assistant")
    tool_msgs = [m for m in messages if m["role"] == "tool"]

    assert len(assistant_msg["tool_calls"]) == 2
    id_a = assistant_msg["tool_calls"][0]["id"]
    id_b = assistant_msg["tool_calls"][1]["id"]
    assert id_a != id_b

    assert len(tool_msgs) == 2
    assert tool_msgs[0]["tool_call_id"] == id_a
    assert "content A" in tool_msgs[0]["content"]
    assert tool_msgs[1]["tool_call_id"] == id_b
    assert "content B" in tool_msgs[1]["content"]


def test_normalize_schema_coerces_string_integers():
    from agy2api.translator import normalize_schema

    gemini_schema = {
        "type": "OBJECT",
        "properties": {
            "Message": {
                "type": "STRING",
                "description": "Content of the message.",
                "minLength": "1",
                "maxLength": "500",
            },
            "Recipient": {
                "type": "STRING",
                "minLength": "1",
            },
            "options": {
                "type": "ARRAY",
                "minItems": "2",
                "maxItems": "10",
                "items": {"type": "STRING"},
            },
            "score": {
                "type": "NUMBER",
                "minimum": "0.5",
                "maximum": "100",
            },
        },
        "required": ["Recipient", "Message"],
    }

    normalized = normalize_schema(gemini_schema)
    assert normalized["type"] == "object"
    assert normalized["properties"]["Message"]["type"] == "string"
    assert normalized["properties"]["Message"]["minLength"] == 1
    assert isinstance(normalized["properties"]["Message"]["minLength"], int)
    assert normalized["properties"]["Message"]["maxLength"] == 500
    assert isinstance(normalized["properties"]["Message"]["maxLength"], int)

    assert normalized["properties"]["Recipient"]["minLength"] == 1
    assert isinstance(normalized["properties"]["Recipient"]["minLength"], int)

    assert normalized["properties"]["options"]["minItems"] == 2
    assert isinstance(normalized["properties"]["options"]["minItems"], int)
    assert normalized["properties"]["options"]["maxItems"] == 10
    assert isinstance(normalized["properties"]["options"]["maxItems"], int)

    assert normalized["properties"]["score"]["minimum"] == 0.5
    assert normalized["properties"]["score"]["maximum"] == 100

