"""Price diff detector — detects price changes between refreshes."""

import copy
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PriceDiffDetector:
    """Detect price changes between endpoint cache snapshots."""

    def __init__(self, snapshot_dir: str = "data"):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._last_snapshot: dict[str, dict] = {}

    def save_snapshot(self, cache_data: dict) -> None:
        """Save a snapshot of current cache for diffing.

        Uses deep copy to avoid reference sharing — modifications to
        cache_data after this call must not affect the saved snapshot.
        """
        model_id = cache_data.get("model_id")
        if not model_id:
            return

        # Sanitize model_id for filename
        safe_name = model_id.replace("/", "_").replace(":", "_")
        snapshot_path = self.snapshot_dir / f"snapshot_{safe_name}.json"

        # Save deep copy to file
        with open(snapshot_path, "w") as f:
            json.dump(cache_data, f, indent=2)

        # Update in-memory last snapshot with deep copy
        self._last_snapshot[model_id] = copy.deepcopy(cache_data)
        logger.debug(f"Saved snapshot for {model_id}")

    def detect_changes(self, new_cache_data: dict) -> list[dict]:
        """Detect price changes between last snapshot and new data.

        Returns:
            List of price change events.
        """
        model_id = new_cache_data.get("model_id")
        if not model_id:
            return []

        last = self._last_snapshot.get(model_id)
        if not last:
            logger.debug(f"No previous snapshot for {model_id}, skipping diff")
            self.save_snapshot(new_cache_data)
            return []

        events = []
        new_endpoints = {ep["tag"]: ep for ep in new_cache_data.get("endpoints", [])}
        last_endpoints = {ep["tag"]: ep for ep in last.get("endpoints", [])}

        # Check all providers present in both snapshots
        for tag in set(new_endpoints.keys()) & set(last_endpoints.keys()):
            new_ep = new_endpoints[tag]
            last_ep = last_endpoints[tag]

            changes = []
            for field in ["prompt", "completion", "input_cache_read"]:
                new_val = float(new_ep.get("pricing", {}).get(field, 0) or 0)
                last_val = float(last_ep.get("pricing", {}).get(field, 0) or 0)
                if last_val == 0:
                    continue  # Skip if last value was 0 (missing)

                diff = abs(new_val - last_val)
                rel_diff = diff / last_val if last_val > 0 else 0

                if rel_diff > 0.01:  # 1% threshold
                    changes.append({
                        "field": field,
                        "old_value": last_val,
                        "new_value": new_val,
                        "diff": diff,
                        "rel_diff": rel_diff,
                    })

            if changes:
                events.append({
                    "provider": tag,
                    "model": model_id,
                    "changes": changes,
                })
                logger.info(f"Price change detected for {tag}: {changes}")

        # Save new snapshot
        self.save_snapshot(new_cache_data)

        return events

    def get_last_snapshot(self, model_id: str) -> Optional[dict]:
        """Get last snapshot for a model."""
        return self._last_snapshot.get(model_id)
