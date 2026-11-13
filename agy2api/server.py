from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional, Tuple
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from agy2api.config import Config
from agy2api.catalog import ModelCatalog
from agy2api.framing import SSEAccumulator, format_gemini_sse
from agy2api.translator import (
    StreamingChunkTranslator,
    gemini_to_openai_request,
    openai_to_gemini_response,
)


def format_inline_error_candidate(status_code: int, raw_text: str) -> Dict[str, Any]:
    """Format upstream errors into a standard Gemini model candidate part so agy renders it inline."""
    err_msg = raw_text
    try:
        err_json = json.loads(raw_text)
        if isinstance(err_json, dict) and "error" in err_json:
            e = err_json["error"]
            err_msg = e.get("message") if isinstance(e, dict) else str(e)
    except Exception:
        pass

    formatted = f"❌ Upstream Error ({status_code}): {err_msg}"
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": f"\n\n```\n{formatted}\n```\n"}],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ]
    }


def resolve_model(
    requested_model: str,
    default_model: str,
    override_model: Optional[str] = None,
) -> str:
    """Resolve incoming model identifier to the actual upstream model ID."""
    if override_model:
        return override_model.strip()

    cleaned = requested_model.removeprefix("models/").split(":")[0].strip()
    if not cleaned:
        return default_model

    # Built-in Antigravity default names map to user's configured default model
    cleaned_lower = cleaned.lower()
    if cleaned_lower in ("default", "custom") or cleaned_lower.startswith(("gemini", "claude-sonnet-", "gpt-oss-")):
        return default_model

    return cleaned


def get_upstream_request_params(
    config: Config,
    request: Request,
    model_path: str,
    payload: Dict[str, Any],
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    # Unwrap CloudCode envelope if present {"project": "...", "model": "...", "request": {...}}
    if "request" in payload and isinstance(payload["request"], dict):
        if (not model_path or model_path.lower() in ("default", "custom")) and payload.get("model"):
            model_path = payload["model"]
        payload = payload["request"]

    override_header = request.headers.get("x-upstream-model")
    target_model = resolve_model(model_path, config.default_model, override_model=override_header)
    openai_req = gemini_to_openai_request(payload, target_model)

    url = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    return url, headers, openai_req


def create_app(
    config: Config,
    http_client: Optional[httpx.AsyncClient] = None,
) -> FastAPI:
    client = http_client or httpx.AsyncClient(timeout=120.0)
    catalog = ModelCatalog(config=config, http_client=client)

    log_file = os.path.expanduser("~/.gemini/agy2api.log")

    def log_line(msg: str):
        try:
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {msg}\n")
        except Exception:
            pass

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Prefetch available models on startup
        startup_prefetch = asyncio.create_task(catalog.get_models(client=client, force_refresh=True))
        yield
        if not startup_prefetch.done():
            startup_prefetch.cancel()

    app = FastAPI(title="agy2api-proxy", lifespan=lifespan)

    @app.middleware("http")
    async def track_activity(request: Request, call_next):
        path = request.url.path
        if not path.startswith("/health"):
            query = f"?{request.url.query}" if request.url.query else ""
            log_line(f">> {request.method} {path}{query}")
        response = await call_next(request)
        if not path.startswith("/health"):
            log_line(f"<< {request.method} {path} -> {response.status_code}")
        return response

    @app.get("/health")
    async def health() -> Response:
        return JSONResponse(content={
            "status": "ok",
            "port": config.port,
            "pid": os.getpid(),
            "defaultModel": config.default_model,
            "baseUrl": config.base_url,
        })

    @app.post("/_reload_config")
    async def reload_config(request: Request) -> Response:
        try:
            body = await request.body()
            if body:
                payload = json.loads(body)
                if "baseUrl" in payload:
                    config.base_url = payload["baseUrl"]
                if "apiKey" in payload:
                    config.api_key = payload["apiKey"]
                if "defaultModel" in payload:
                    config.default_model = payload["defaultModel"]
                if "selectedModels" in payload:
                    config.selected_models = payload["selectedModels"]
            else:
                from agy2api.config import load_config
                loaded = load_config()
                config.base_url = loaded.base_url
                config.api_key = loaded.api_key
                config.default_model = loaded.default_model
                config.selected_models = loaded.selected_models
        except Exception as e:
            logger.warning("Error in _reload_config: %s", e)

        catalog.invalidate_cache()
        models = await catalog.get_models(client=client, force_refresh=True)
        return JSONResponse(content={
            "status": "ok",
            "baseUrl": config.base_url,
            "defaultModel": config.default_model,
            "count": len(models),
        })

    @app.post("/_set_model")
    async def set_active_model(request: Request) -> Response:
        data = await request.json()
        model = data.get("model")
        if model:
            config.default_model = model
            return JSONResponse(content={"status": "ok", "defaultModel": config.default_model})
        return JSONResponse(status_code=400, content={"error": "Missing model field"})

    @app.post("/_refresh_models")
    async def refresh_models() -> Response:
        catalog.invalidate_cache()
        models = await catalog.get_models(client=client, force_refresh=True)
        return JSONResponse(content={"status": "ok", "count": len(models)})

    @app.post("/_shutdown")
    async def shutdown() -> Response:
        loop = asyncio.get_event_loop()
        loop.call_later(0.2, lambda: os._exit(0))
        return JSONResponse(content={"status": "shutting_down"})

    @app.post("/v1beta/models/{model_path:path}:generateContent")
    @app.post("/v1beta/{model_path:path}:generateContent")
    async def generate_content(model_path: str, request: Request) -> Response:
        payload = await request.json()
        url, headers, openai_req = get_upstream_request_params(config, request, model_path, payload)

        try:
            resp = await client.post(url, json=openai_req, headers=headers)
        except Exception as exc:
            err_candidate = format_inline_error_candidate(502, f"Failed connecting to upstream {url}: {exc}")
            return JSONResponse(content=err_candidate)

        if resp.status_code != 200:
            err_candidate = format_inline_error_candidate(resp.status_code, resp.text)
            return JSONResponse(content=err_candidate)

        gemini_resp = openai_to_gemini_response(resp.json())
        return JSONResponse(content=gemini_resp)

    @app.post("/v1beta/models/{model_path:path}:streamGenerateContent")
    @app.post("/v1beta/{model_path:path}:streamGenerateContent")
    async def stream_generate_content(model_path: str, request: Request) -> Response:
        payload = await request.json()
        url, headers, openai_req = get_upstream_request_params(config, request, model_path, payload)
        openai_req["stream"] = True

        async def stream_chunks() -> AsyncGenerator[bytes, None]:
            try:
                req = client.build_request("POST", url, json=openai_req, headers=headers)
                upstream_resp = await client.send(req, stream=True)
            except Exception as exc:
                err_candidate = format_inline_error_candidate(502, f"Failed connecting to upstream {url}: {exc}")
                yield format_gemini_sse(err_candidate).encode("utf-8")
                return

            if upstream_resp.status_code != 200:
                err_text = await upstream_resp.aread()
                err_candidate = format_inline_error_candidate(
                    upstream_resp.status_code,
                    err_text.decode("utf-8", "replace"),
                )
                yield format_gemini_sse(err_candidate).encode("utf-8")
                await upstream_resp.aclose()
                return

            translator = StreamingChunkTranslator()
            accumulator = SSEAccumulator()
            done = False

            try:
                async for raw_chunk in upstream_resp.aiter_text():
                    events = accumulator.feed(raw_chunk)
                    for ev in events:
                        if ev.data.strip() == "[DONE]":
                            done = True
                            break
                        try:
                            chunk_data = json.loads(ev.data)
                            gemini_frames = translator.process_chunk(chunk_data)
                            for frame in gemini_frames:
                                yield format_gemini_sse(frame).encode("utf-8")
                        except Exception:
                            continue
                    if done:
                        break

                flushed_events = accumulator.flush()
                for ev in flushed_events:
                    if ev.data.strip() == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(ev.data)
                        gemini_frames = translator.process_chunk(chunk_data)
                        for frame in gemini_frames:
                            yield format_gemini_sse(frame).encode("utf-8")
                    except Exception:
                        pass

                final_frame = translator.flush()
                if final_frame:
                    yield format_gemini_sse(final_frame).encode("utf-8")
            except Exception as exc:
                err_candidate = format_inline_error_candidate(502, f"Stream interrupted: {exc}")
                yield format_gemini_sse(err_candidate).encode("utf-8")
            finally:
                await upstream_resp.aclose()

        return StreamingResponse(stream_chunks(), media_type="text/event-stream")

    @app.post("/v1internal:listExperiments")
    async def list_experiments() -> Response:
        flags = [
            {"name": "show-all-models-in-gemini-app", "boolValue": True},
            {"name": "Chat__enable_chat_moa", "boolValue": True},
            {"name": "disable-teamwork-forced-flash-model", "boolValue": True},
            {"name": "dark-launch-aicode-quota-status", "boolValue": True},
            {"name": "SDLCAgents__enable_anthropic_model_connection", "boolValue": True},
            {"name": "SDLCAgents__enable_azure_model_connection", "boolValue": True},
            {"name": "SDLCAgents__enable_gemini_model_connection", "boolValue": True},
            {"name": "SDLCAgents__enable_rest_model_connection", "boolValue": True},
            {"name": "Chat__enable_chat_agentic_mcp_chat", "boolValue": True},
            {"name": "enable-deepagent", "boolValue": True},
            {"name": "enable-agent-team", "boolValue": True},
            {"name": "enable-concierge-agent", "boolValue": True},
            {"name": "Chat__enable_mcp_server", "boolValue": True},
        ]
        return JSONResponse(content={"flags": flags})

    @app.post("/v1internal:fetchAvailableModels")
    async def fetch_available_models(request: Request) -> Response:
        data = await catalog.get_available_models_catalog(client=client)
        return JSONResponse(content=data)

    @app.post("/v1internal:loadCodeAssist")
    async def load_code_assist() -> Response:
        return JSONResponse(content={
            "currentTier": "standard-tier",
            "allowedTiers": ["free-tier", "standard-tier", "g1-pro-tier"],
        })

    @app.post("/v1internal:retrieveUserQuota")
    @app.post("/v1internal:retrieveUserQuotaSummary")
    async def retrieve_user_quota() -> Response:
        return JSONResponse(content={"buckets": []})

    @app.post("/v1internal:fetchUserInfo")
    async def fetch_user_info() -> Response:
        return JSONResponse(content={
            "email": "user@antigravity.local",
            "userTier": "standard-tier",
        })

    @app.post("/v1internal:streamGenerateContent")
    async def stream_generate_content_internal(request: Request) -> Response:
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}
        model_name = payload.get("model", "")
        return await stream_generate_content(model_name, request)

    @app.post("/v1internal:generateContent")
    async def generate_content_internal(request: Request) -> Response:
        body = await request.body()
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}
        model_name = payload.get("model", "")
        return await generate_content(model_name, request)

    @app.get("/v1beta/models")
    @app.get("/v1/models")
    async def list_models(request: Request) -> Response:
        models_list = await catalog.get_models(client=client)
        clean_default = config.default_model.removeprefix("models/").lstrip("~")
        return JSONResponse(content={
            "models": models_list,
            "defaultAgentModelId": clean_default,
            "totalCount": len(models_list),
        })

    @app.get("/v1beta/models/{model_path:path}")
    @app.get("/v1/models/{model_path:path}")
    async def get_model_endpoint(model_path: str) -> Response:
        descriptor = await catalog.get_model(model_path, client=client)
        return JSONResponse(content=descriptor)

    return app
