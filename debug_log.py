"""Debug request logging — full RAW capture of in/out request bodies.

Activated by `server.debug_log: true` (or env DEBUG_REQUEST_LOG=1), read
DYNAMICALLY on every chat completion so the dashboard toggle takes effect
immediately after config reload, without a service restart.

Writes one JSON Lines file per purpose:
  logs/requests.jsonl   — correlated records (req_id):
      req_in : FULL raw body as received from the client + per-message
               fingerprints + attribution headers (NEVER Authorization)
      req_out: FULL upstream body actually sent to OpenRouter (with the
               provider pin) for every attempt
      res    : status + usage (cached tokens!) + provider_response

Privacy: Authorization is never recorded. Bodies are conversation content —
the UI warns loudly when the flag is on and Delete All Logs covers this file.

Rotation mirrors proxy.jsonl (size-based inline, see _check_rotation()).
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from threading import Lock

from config import config

logger = logging.getLogger(__name__)

_requests_logger = None
_lock = Lock()
_log_path = Path("logs") / "requests.jsonl"

# Rotation threshold, same order as proxy.jsonl handling
_MAX_BYTES = 10 * 1024 * 1024

# Never record these request headers (secrets)
_FORBIDDEN_HEADERS = {"authorization", "proxy-authorization", "cookie", "x-api-key"}


def _ensure_logger():
    """Lazy logger creation: only when debug mode first turns on."""
    global _requests_logger
    with _lock:
        if _requests_logger is not None:
            return _requests_logger
        os.makedirs("logs", exist_ok=True)
        lg = logging.getLogger("requests.debug")
        lg.setLevel(logging.INFO)
        lg.propagate = False
        handler = logging.FileHandler(_log_path)
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(handler)
        _requests_logger = lg
        return lg


def _check_rotation():
    """Size-based inline rotation; close+reopen the logger's handler."""
    global _requests_logger
    try:
        if not _log_path.exists() or _log_path.stat().st_size < _MAX_BYTES:
            return
        ts = datetime.now().strftime("%Y%m%d")
        rotated = Path(f"logs/requests.jsonl-{ts}")
        n = 0
        while rotated.exists():
            n += 1
            rotated = Path(f"logs/requests.jsonl-{ts}-{n}")
        with _lock:
            if _requests_logger:
                for h in list(_requests_logger.handlers):
                    h.close()
                    _requests_logger.removeHandler(h)
                _requests_logger = None
        os.rename(_log_path, rotated)
    except OSError as e:
        logger.warning(f"Debug log rotation failed: {e}")


def is_enabled() -> bool:
    """Dynamic read — no caching, so the config reload is immediate."""
    return bool(config.debug_log_enabled)


def status() -> dict:
    """Expose current state to /api/status (UI banner)."""
    active = is_enabled()
    size = 0
    mtime = None
    try:
        if _log_path.exists():
            st = _log_path.stat()
            size = st.st_size
            mtime = datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")
    except OSError:
        pass
    return {"enabled": active, "file": str(_log_path), "size_bytes": size,
            "last_write": mtime}


# ---------------------------------------------------------------------------
# Canonical message hashing (cache = prefix of messages, so fingerprint them)
# ---------------------------------------------------------------------------

def _canon(o) -> str:
    """Recursive canonical JSON: dict keys sorted, lists in order.

    json.dumps(sort_keys=True) sorts only top-level and nested dicts but the
    recursion must also survive non-string keys/tuples — this helper keeps
    it simple and total (repr fallback).
    """
    try:
        return json.dumps(o, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
    except (TypeError, ValueError):
        if isinstance(o, dict):
            return "{" + ",".join(
                f"{_canon(str(k))}:{_canon(v)}" for k, v in sorted(o.items())
            ) + "}"
        if isinstance(o, (list, tuple)):
            return "[" + ",".join(_canon(v) for v in o) + "]"
        return repr(o)


def canonical_messages_hash(messages) -> str:
    """sha256 over a canonical JSON of the messages array.

    Key order is irrelevant, content and order of messages are significant:
    cache = prefix of messages, so equal messages -> equal hash.
    """
    return hashlib.sha256(_canon(messages).encode("utf-8")).hexdigest()


def messages_fingerprint(messages) -> list[dict]:
    """Compact per-message fingerprint list: index, role, char length, hash8."""
    out = []
    for i, msg in enumerate(messages or []):
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str):
            try:
                content_s = json.dumps(content, sort_keys=True,
                                       ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError):
                content_s = repr(content)
        else:
            content_s = content
        out.append({
            "i": i,
            "role": msg.get("role") if isinstance(msg, dict) else "?",
            "chars": len(content_s),
            "h": hashlib.sha256(content_s.encode("utf-8")).hexdigest()[:8],
        })
    return out


def _trim(s: str, limit: int) -> str:
    """Apply the body char cap with an explicit truncation marker."""
    if limit and len(s) > limit:
        return s[:limit] + f"\n...[TRUNCATED at {limit} chars, original {len(s)} chars]"
    return s


def write_record(record_type: str, payload: dict) -> None:
    """Write one JSON line if (dynamically) enabled. Never raise."""
    try:
        if record_type == "req_in":
            enabled_now = is_enabled()
            # Snap-and-check: req_in only when the flag was on AT ARRIVAL.
            if not enabled_now:
                return
            _check_rotation()
            lg = _ensure_logger()
        else:
            # req_out / res follow their req_in — record if flag currently on;
            # records stay correlated via req_id either way.
            if not is_enabled():
                return
            lg = _requests_logger
            if lg is None:
                return
        entry = {"ts": datetime.now().isoformat(), "type": record_type,
                 **payload}
        lg.info(json.dumps(entry, ensure_ascii=False))
    except Exception as e:  # never break proxying because of debugging
        logger.warning(f"Debug log write failed: {e}")


# ---------------------------------------------------------------------------
# High-level hooks called from routes.py
# ---------------------------------------------------------------------------

def hook_request_in(req_id: str, body_dict: dict, client_headers) -> None:
    """Record the raw incoming body + safe metadata. Snapshot BEFORE any change."""
    if not is_enabled():
        return
    headers_safe = {
        k: v for k, v in getattr(client_headers, "items", lambda: [])()
        if k.lower() not in _FORBIDDEN_HEADERS
    }
    messages = body_dict.get("messages") or []
    payload = {
        "req_id": req_id,
        "session_id": body_dict.get("session_id"),
        "model": body_dict.get("model"),
        "stream": body_dict.get("stream", False),
        "body_raw": _trim(json.dumps(body_dict, ensure_ascii=False),
                          config.debug_max_body_chars),
        "msg_count": len(messages),
        "total_msg_chars": sum(
            len(m.get("content") or "") if isinstance(m.get("content"), str)
            else -1 for m in messages),
        "messages_hash": canonical_messages_hash(messages),
        "fingerprint": messages_fingerprint(messages),
        "client_headers": headers_safe,
    }
    write_record("req_in", payload)


def hook_request_out(req_id: str, upstream_body: dict, provider_slug: str,
                     attempt: int) -> None:
    """Record the exact upstream body sent for this attempt."""
    messages = upstream_body.get("messages") or []
    serialized = json.dumps(upstream_body, ensure_ascii=False)
    payload = {
        "req_id": req_id,
        "attempt": attempt,
        "provider_pin": provider_slug,
        "body_out": _trim(serialized, config.debug_max_body_chars),
        "body_out_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "messages_hash": canonical_messages_hash(messages),
        "msg_count": len(messages),
    }
    write_record("req_out", payload)


def hook_response(req_id: str, provider_slug: str, status_code: int,
                  usage: dict | None, provider_response=None,
                  model_response=None, latency_ms: int | None = None,
                  error: str | None = None) -> None:
    """Record the upstream response outcome (usage carries cached_tokens)."""
    payload = {
        "req_id": req_id,
        "provider": provider_slug,
        "status": status_code,
        "usage": usage,
        "cached_tokens": (usage or {}).get("prompt_tokens_details", {}).get("cached_tokens"),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "provider_response": provider_response,
        "model_response": model_response,
        "latency_ms": latency_ms,
        "error": (error[:300] if isinstance(error, str) else error),
    }
    write_record("res", payload)
