"""Shared test setup.

`routes.py` opens a relative-path FileHandler (logs/proxy.jsonl) at import
time, and `cache.py`/`price_diff.py` may touch `data/`. Ensure those
directories exist before any test module is imported.
"""

import os

os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)
