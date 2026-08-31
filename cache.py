"""Endpoint cache manager.

Loads/saves endpoint data from/to JSON files.
Cache key: data/or_endpoints_<model>.json
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EndpointCache:
    """In-memory cache with JSON persistence."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, dict] = {}  # model_id -> {endpoints, fetched_at}

    def _cache_path(self, model_id: str) -> Path:
        """Get cache file path for a model."""
        # Sanitize model_id for filename
        safe_name = model_id.replace("/", "_").replace(":", "_")
        return self.data_dir / f"or_endpoints_{safe_name}.json"

    def get(self, model_id: str) -> Optional[dict]:
        """Get cached endpoints for a model, or None if not cached."""
        return self._cache.get(model_id)

    def set(self, model_id: str, data: dict) -> None:
        """Cache endpoints for a model and persist to disk.

        Before overwriting, diffs the endpoint tags against the previous
        in-memory snapshot and logs any change at INFO (slug-watcher):
        OpenRouter reshapes slugs/quantizations over time (e.g.
        "deepseek/fp8" -> "deepseek" with quantization "unknown") and such
        changes used to silently alter routing behaviour.
        """
        old = self._cache.get(model_id)
        if old:
            old_tags = {e.get("tag") for e in (old.get("endpoints") or []) if e.get("tag")}
            new_tags = {e.get("tag") for e in (data.get("endpoints") or []) if e.get("tag")}
            appeared = new_tags - old_tags
            vanished = old_tags - new_tags
            if appeared:
                logger.info(f"[slug-watch] {model_id}: new endpoint tags: {sorted(appeared)}")
            if vanished:
                logger.info(f"[slug-watch] {model_id}: removed endpoint tags: {sorted(vanished)}")
        self._cache[model_id] = data
        self._save(model_id, data)

    def _save(self, model_id: str, data: dict) -> None:
        """Save cache to JSON file."""
        path = self._cache_path(model_id)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_disk(self, model_id: str) -> Optional[dict]:
        """Load cache from disk (bypasses in-memory)."""
        path = self._cache_path(model_id)
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        # Also update in-memory cache
        self._cache[model_id] = data
        return data

    def get_all_cached(self) -> dict[str, dict]:
        """Get all cached models (for debugging)."""
        return dict(self._cache)

    def clear(self, model_id: Optional[str] = None) -> None:
        """Clear cache for a model or all models."""
        if model_id:
            self._cache.pop(model_id, None)
            path = self._cache_path(model_id)
            if path.exists():
                path.unlink()
        else:
            self._cache.clear()
            for path in self.data_dir.glob("or_endpoints_*.json"):
                path.unlink()
