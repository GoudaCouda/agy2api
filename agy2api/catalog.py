from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
import httpx

from agy2api.config import Config

logger = logging.getLogger(__name__)

DEFAULT_CACHE_TTL_SECONDS = 60.0  # 60s low TTL for responsive in-memory cache


def search_models(models: List[Dict[str, Any]], query: Optional[str]) -> List[Dict[str, Any]]:
    """Search models by query tokens (case-insensitive substring match)."""
    if not query or not query.strip():
        return models

    tokens = [t.strip().lower() for t in query.replace(",", " ").split() if t.strip()]
    if not tokens:
        return models

    matched: List[Dict[str, Any]] = []
    for m in models:
        mid = m.get("name", "").lower()
        dname = (m.get("displayName") or "").lower()
        desc = (m.get("description") or "").lower()
        target_text = f"{mid} {dname} {desc}"

        if any(token in target_text for token in tokens):
            matched.append(m)

    return matched


def build_gemini_model_descriptor(
    model_id: str,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    context_length: int = 131072,
    max_output: int = 8192,
) -> Dict[str, Any]:
    """Format a model identifier into a standard Google Gemini model descriptor."""
    clean_id = model_id.removeprefix("models/").lstrip("~")
    dname = display_name.strip() if display_name else clean_id
    return {
        "name": f"models/{clean_id}",
        "version": "001",
        "displayName": dname,
        "description": description or f"Model {dname}",
        "inputTokenLimit": context_length,
        "outputTokenLimit": max_output,
        "supportedGenerationMethods": [
            "generateContent",
            "countTokens",
            "streamGenerateContent",
        ],
    }


class ModelCatalog:
    """Manages cached upstream models and translates them into Gemini descriptors."""

    def __init__(
        self,
        config: Config,
        http_client: Optional[httpx.AsyncClient] = None,
        cache_ttl_seconds: float = DEFAULT_CACHE_TTL_SECONDS,
    ):
        self.config = config
        self.http_client = http_client
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cached_models: List[Dict[str, Any]] = []
        self._cache_timestamp: float = 0.0
        self._cached_base_url: str = config.base_url

    def _is_cache_valid(self) -> bool:
        if not self._cached_models:
            return False
        return (time.time() - self._cache_timestamp) < self.cache_ttl_seconds

    def invalidate_cache(self) -> None:
        self._cached_models = []
        self._cache_timestamp = 0.0

    def resolve_target_model(
        self,
        requested_model: str,
        override_model: Optional[str] = None,
    ) -> str:
        """Resolve client requested model string to the actual upstream model ID."""
        if override_model:
            return override_model.strip()

        clean = requested_model.removeprefix("models/").split(":")[0].strip()
        if not clean:
            return self.config.default_model

        if clean.lower() in ("default", "custom"):
            return self.config.default_model

        return clean

    async def get_models(
        self,
        client: Optional[httpx.AsyncClient] = None,
        force_refresh: bool = False,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch and return Gemini-compatible model descriptors for upstream models."""
        if self._cached_base_url.rstrip("/") != self.config.base_url.rstrip("/"):
            self._cached_models = []
            self._cache_timestamp = 0.0
            self._cached_base_url = self.config.base_url

        if not force_refresh and self._is_cache_valid():
            all_cached = list(self._cached_models)
            return self._select_models(all_cached, query=query)

        active_client = client or self.http_client
        upstream_models: List[Dict[str, Any]] = []

        headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        url = f"{self.config.base_url.rstrip('/')}/models"

        try:
            if active_client is not None:
                resp = await active_client.get(url, headers=headers, timeout=10.0)
            else:
                async with httpx.AsyncClient(timeout=10.0) as temp_client:
                    resp = await temp_client.get(url, headers=headers)

            if resp.status_code == 200:
                data = resp.json()
                raw_items = data.get("data", []) if isinstance(data, dict) else []
                for item in raw_items:
                    if isinstance(item, dict) and "id" in item:
                        mid = str(item["id"])
                        dname = str(item.get("name") or mid)
                        desc = item.get("description")
                        ctx = (
                            item.get("context_length")
                            or item.get("max_tokens")
                            or 131072
                        )
                        max_out = item.get("max_completion_tokens") or 8192
                        upstream_models.append(
                            build_gemini_model_descriptor(
                                model_id=mid,
                                display_name=dname,
                                description=desc,
                                context_length=ctx,
                                max_output=max_out,
                            )
                        )
            else:
                logger.warning(
                    "Upstream /models returned HTTP %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as exc:
            logger.warning("Failed to query upstream models from %s: %s", url, exc)

        if not upstream_models:
            if self._cached_models and not force_refresh:
                return self._select_models(list(self._cached_models), query=query)
            if self.config.default_model:
                upstream_models = [
                    build_gemini_model_descriptor(
                        model_id=self.config.default_model,
                        display_name=f"{self.config.default_model}",
                    )
                ]

        self._cached_models = upstream_models
        self._cache_timestamp = time.time()
        self._cached_base_url = self.config.base_url
        return self._select_models(list(self._cached_models), query=query)

    def _select_models(self, models: List[Dict[str, Any]], query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Select up to 32 curated models (or all if querying via CLI)."""
        if query:
            return search_models(models, query)

        if self.config.selected_models:
            selected_set = {m.lower().removeprefix("models/").lstrip("~") for m in self.config.selected_models}
            curated: List[Dict[str, Any]] = []
            for m in models:
                mid = m["name"].lower().removeprefix("models/").lstrip("~")
                if mid in selected_set:
                    curated.append(m)

            if self.config.default_model:
                clean_def = self.config.default_model.lower().removeprefix("models/").lstrip("~")
                if clean_def not in {c["name"].lower().removeprefix("models/").lstrip("~") for c in curated}:
                    for m in models:
                        if m["name"].lower().removeprefix("models/").lstrip("~") == clean_def:
                            curated.insert(0, m)
                            break

            return curated if curated else models[:32]

        return models[:32]

    async def get_model(
        self,
        model_name: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """Retrieve descriptor for a single model."""
        clean = model_name.removeprefix("models/").split(":")[0]
        models = await self.get_models(client=client)
        for m in models:
            if m["name"] == f"models/{clean}" or m["displayName"] == clean:
                return m

        return build_gemini_model_descriptor(clean)

    async def get_available_models_catalog(
        self,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """Format models for CloudCode v1internal:fetchAvailableModels schema (up to 32 models)."""
        models_list = await self.get_models(client=client)
        models_dict: Dict[str, Any] = {}
        model_ids: List[str] = []
        for m in models_list:
            clean_id = m["name"].removeprefix("models/")
            model_ids.append(clean_id)
            models_dict[clean_id] = {
                "displayName": m.get("displayName") or clean_id,
                "description": m.get("description") or f"Model {clean_id}",
                "maxTokens": m.get("inputTokenLimit", 131072),
                "maxOutputTokens": m.get("outputTokenLimit", 8192),
                "supportedGenerationMethods": m.get("supportedGenerationMethods", []),
            }

        clean_default = self.config.default_model.removeprefix("models/").lstrip("~")
        if not clean_default and model_ids:
            clean_default = model_ids[0]

        groups = [
            {
                "displayName": "Models",
                "modelIds": model_ids,
            }
        ]

        return {
            "models": models_dict,
            "defaultAgentModelId": clean_default,
            "agentModelSorts": [
                {
                    "displayName": "Alphabetical",
                    "groups": groups,
                }
            ],
            "tieredModelIds": {
                "flash": model_ids[:2] if len(model_ids) >= 2 else model_ids,
                "pro": model_ids[2:] if len(model_ids) > 2 else model_ids,
            }
        }
