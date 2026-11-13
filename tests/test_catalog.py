import pytest
import httpx
from agy2api.config import Config
from agy2api.catalog import (
    ModelCatalog,
    build_gemini_model_descriptor,
    search_models,
)


@pytest.fixture
def mock_config():
    return Config(
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test-key",
        default_model="anthropic/claude-3.5-sonnet",
    )


def test_build_gemini_model_descriptor():
    desc = build_gemini_model_descriptor(
        model_id="deepseek/deepseek-r1",
        display_name="DeepSeek R1",
        description="Reasoning model",
        context_length=64000,
        max_output=4096,
    )
    assert desc["name"] == "models/deepseek/deepseek-r1"
    assert desc["displayName"] == "DeepSeek R1"
    assert desc["inputTokenLimit"] == 64000
    assert desc["outputTokenLimit"] == 4096
    assert "generateContent" in desc["supportedGenerationMethods"]


def test_resolve_target_model(mock_config):
    catalog = ModelCatalog(mock_config)
    # Explicit override wins
    assert catalog.resolve_target_model("models/gemini-3.8-flash", override_model="openai/gpt-4o") == "openai/gpt-4o"
    # Specific model retained
    assert catalog.resolve_target_model("models/meta-llama/llama-3-70b-instruct") == "meta-llama/llama-3-70b-instruct"
    # Action suffix stripped
    assert catalog.resolve_target_model("models/deepseek/deepseek-chat:streamGenerateContent") == "deepseek/deepseek-chat"
    # Empty string falls back to default
    assert catalog.resolve_target_model("") == "anthropic/claude-3.5-sonnet"


@pytest.mark.asyncio
async def test_get_models_from_upstream(mock_config):
    models_response = {
        "data": [
            {
                "id": "anthropic/claude-3.5-sonnet",
                "name": "Anthropic: Claude 3.5 Sonnet",
                "description": "Sonnet 3.5 model",
                "context_length": 200000,
            },
            {
                "id": "google/gemini-flash-1.5",
                "name": "Google: Gemini Flash 1.5",
                "context_length": 1000000,
            },
        ]
    }

    async def mock_handler(request: httpx.Request):
        assert request.url.path == "/api/v1/models"
        assert request.headers["Authorization"] == "Bearer sk-or-test-key"
        return httpx.Response(200, json=models_response)

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        catalog = ModelCatalog(mock_config, http_client=client)
        models = await catalog.get_models()

        assert len(models) == 2
        assert models[0]["name"] == "models/anthropic/claude-3.5-sonnet"
        assert models[0]["displayName"] == "Anthropic: Claude 3.5 Sonnet"
        assert models[0]["inputTokenLimit"] == 200000

        assert models[1]["name"] == "models/google/gemini-flash-1.5"
        assert models[1]["inputTokenLimit"] == 1000000

        # Verify caching works without calling handler again
        models_cached = await catalog.get_models()
        assert len(models_cached) == 2


@pytest.mark.asyncio
async def test_get_models_fallback_on_failure(mock_config):
    async def mock_handler(request: httpx.Request):
        return httpx.Response(500, text="Internal Server Error")

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        catalog = ModelCatalog(mock_config, http_client=client)
        models = await catalog.get_models()

        assert len(models) == 1
        assert models[0]["name"] == "models/anthropic/claude-3.5-sonnet"


@pytest.mark.asyncio
async def test_model_catalog_search_filter(mock_config):
    raw_data = [
        {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet", "description": "Flagship intelligence"},
        {"id": "deepseek/deepseek-r1", "name": "DeepSeek R1", "description": "Reasoning model"},
        {"id": "openai/gpt-4o", "name": "GPT-4o", "description": "Omni model"},
        {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B", "description": "Open weights"},
        {"id": "mistralai/mistral-large-2407", "name": "Mistral Large", "description": "Frontier Mistral"},
    ]

    async def mock_handler(request: httpx.Request):
        return httpx.Response(200, json={"data": raw_data})

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        catalog = ModelCatalog(mock_config, http_client=client)

        # Unfiltered returns all models
        all_models = await catalog.get_models()
        assert len(all_models) == 5

        # Search by single token
        claude_models = await catalog.get_models(query="claude")
        assert len(claude_models) == 1
        assert claude_models[0]["name"] == "models/anthropic/claude-3.5-sonnet"

        # Search by multi-token
        multi_models = await catalog.get_models(query="deepseek, gpt")
        assert len(multi_models) == 2
        names = [m["name"] for m in multi_models]
        assert "models/deepseek/deepseek-r1" in names
        assert "models/openai/gpt-4o" in names

        # Search in description
        reasoning_models = await catalog.get_models(query="reasoning")
        assert len(reasoning_models) == 1
        assert reasoning_models[0]["name"] == "models/deepseek/deepseek-r1"

        # Test catalog structure
        catalog_dict = await catalog.get_available_models_catalog()
        assert len(catalog_dict["models"]) == 5
        assert "defaultAgentModelId" in catalog_dict
        assert len(catalog_dict["agentModelSorts"]) == 1


def test_search_models_matching():
    models = [
        {"name": "models/anthropic/claude-3.5-sonnet", "displayName": "Claude 3.5 Sonnet"},
        {"name": "models/openai/gpt-4o", "displayName": "GPT-4o"},
    ]
    # Empty query returns all
    all_res = search_models(models, None)
    assert len(all_res) == 2

    # Query matching one
    claude_res = search_models(models, "claude")
    assert len(claude_res) == 1
    assert claude_res[0]["name"] == "models/anthropic/claude-3.5-sonnet"


@pytest.mark.asyncio
async def test_selected_models_curation():
    cfg = Config(
        base_url="https://test.com/v1",
        api_key="key",
        default_model="model-1",
        selected_models=["model-2", "model-4"],
    )

    raw_data = [{"id": f"model-{i}"} for i in range(1, 10)]

    async def mock_handler(request: httpx.Request):
        return httpx.Response(200, json={"data": raw_data})

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        catalog = ModelCatalog(cfg, http_client=client)
        models = await catalog.get_models()

        names = [m["name"].removeprefix("models/") for m in models]
        # Should contain model-2, model-4, and default model-1
        assert "model-2" in names
        assert "model-4" in names
        assert "model-1" in names
        assert len(names) == 3


@pytest.mark.asyncio
async def test_endpoint_change_clears_old_models():
    cfg = Config(
        base_url="https://provider1.com/v1",
        api_key="key1",
        default_model="model-1",
    )

    async def mock_handler(request: httpx.Request):
        if "provider1.com" in str(request.url):
            return httpx.Response(200, json={"data": [{"id": "provider1/exclusive-model"}]})
        elif "provider2.com" in str(request.url):
            return httpx.Response(200, json={"data": [{"id": "provider2/new-model"}]})
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        catalog = ModelCatalog(cfg, http_client=client)
        m1 = await catalog.get_models()
        assert any("exclusive-model" in m["name"] for m in m1)

        cfg.base_url = "https://provider2.com/v1"
        cfg.default_model = "provider2/new-model"

        m2 = await catalog.get_models()
        assert not any("exclusive-model" in m["name"] for m in m2)
        assert any("new-model" in m["name"] for m in m2)


@pytest.mark.asyncio
async def test_low_ttl_and_force_refresh(mock_config):
    calls = 0

    async def mock_handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"data": [{"id": f"model-call-{calls}"}]})

    transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        catalog = ModelCatalog(mock_config, http_client=client, cache_ttl_seconds=0.2)

        res1 = await catalog.get_models()
        assert calls == 1
        assert res1[0]["name"] == "models/model-call-1"

        res2 = await catalog.get_models()
        assert calls == 1
        assert res2[0]["name"] == "models/model-call-1"

        res3 = await catalog.get_models(force_refresh=True)
        assert calls == 2
        assert res3[0]["name"] == "models/model-call-2"
