"""Endpoint fetcher — fetches provider endpoints from OpenRouter API."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

from config import config
from cache import EndpointCache

logger = logging.getLogger(__name__)


class EndpointFetcher:
    """Fetch and cache provider endpoints from OpenRouter API."""

    def __init__(self, api_key: str, cache: EndpointCache):
        self.api_key = api_key
        self.cache = cache
        self._client = httpx.AsyncClient(timeout=30.0)

    async def fetch_model_endpoints(self, model_id: str) -> Optional[dict]:
        """Fetch endpoints for a model from OpenRouter API."""
        url = f"https://openrouter.ai/api/v1/models/{model_id}/endpoints"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with self._client as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    logger.error(f"Failed to fetch endpoints for {model_id}: {resp.status_code}")
                    return None

                data = resp.json()
                endpoints = data.get("data", {}).get("endpoints", [])

                # Transform to our format
                transformed = []
                for ep in endpoints:
                    tag = ep.get("tag") or ep.get("provider_id")
                    if not tag:
                        continue

                    pricing = ep.get("pricing") or {}
                    transformed.append({
                        "tag": tag,
                        "provider_id": ep.get("provider_id"),
                        "provider_name": ep.get("provider_name"),
                        "pricing": {
                            "prompt": pricing.get("prompt", "0"),
                            "completion": pricing.get("completion", "0"),
                            "input_cache_read": pricing.get("input_cache_read", "0"),
                        },
                        "supports_implicit_caching": ep.get("supports_implicit_caching", False),
                        "uptime_last_30m": ep.get("uptime_last_30m"),
                    })

                cache_data = {
                    "model_id": model_id,
                    "endpoints": transformed,
                    "fetched_at": datetime.utcnow().isoformat(),
                }

                # Save to cache
                self.cache.set(model_id, cache_data)
                logger.info(f"Fetched {len(transformed)} endpoints for {model_id}")

                return cache_data

        except httpx.RequestError as e:
            logger.error(f"Error fetching endpoints for {model_id}: {e}")
            return None

    async def refresh_all_models(self) -> dict:
        """Refresh endpoints for all configured models."""
        results = {}
        for model_id in config.models:
            data = await self.fetch_model_endpoints(model_id)
            results[model_id] = "success" if data else "failed"
        return results

    async def refresh_model(self, model_id: str) -> bool:
        """Refresh endpoints for a single model."""
        data = await self.fetch_model_endpoints(model_id)
        return data is not None

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()
