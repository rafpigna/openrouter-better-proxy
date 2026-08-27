"""Web dashboard routes — FastAPI endpoints for dashboard UI.

IMPORTANTE: usare SEMPRE Path(__file__).parent per path assoluti,
mai path relativi (il servizio potrebbe essere lanciato da ovunque).
"""

import asyncio
import csv
import json
import logging
import os
import shutil
import sys
import signal
import time
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import yaml
from config import config
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# OpenRouter prices are per-token; the dashboard displays them in dollars per
# MILLION tokens (×1e6, no rounding). Conversion happens only at the display
# boundary (catalog + endpoints API), never on stored/cached data.
PER_MILLION = 1_000_000


def _to_per_million(value) -> float:
    """Convert a per-token price (string or number) to $/M tokens.

    Pure scale conversion — no rounding. Returns 0.0 for missing/empty input.
    """
    try:
        return float(value or 0) * PER_MILLION
    except (TypeError, ValueError):
        return 0.0

# ---------------------------------------------------------------------------
# Globals injected by main.py at startup
# ---------------------------------------------------------------------------

_scheduler: Optional[Any] = None        # RefreshScheduler
_router_instance: Optional[Any] = None  # Router
_log_handler: Optional[Any] = None      # SSELogHandler
_fetcher: Optional[Any] = None          # EndpointFetcher
_start_time: float = time.monotonic()
_config_lock = asyncio.Lock()           # protects config file operations

# Base dir: parent of this file (= project root)
BASE_DIR = Path(__file__).parent

# Config file path
CONFIG_YAML_PATH = BASE_DIR / "routing_config.yaml"


def init_web_routes(
    scheduler: Any,
    router_instance: Any,
    log_handler: Any,
    fetcher: Any = None,
) -> None:
    """Inject dependencies after startup."""
    global _scheduler, _router_instance, _log_handler, _fetcher
    _scheduler = scheduler
    _router_instance = router_instance
    _log_handler = log_handler
    _fetcher = fetcher


def increment_request_count() -> None:
    """Called from routes.py on each /v1/chat/completions."""
    pass  # handled via direct import in routes.py


def get_request_count() -> int:
    """Current request counter (set by routes.py)."""
    from routes import REQUEST_COUNT
    return REQUEST_COUNT


def get_error_count() -> int:
    """Current error counter (set by routes.py)."""
    from routes import ERROR_COUNT
    return ERROR_COUNT


# ===========================================================================
# API Router
# ===========================================================================

router = APIRouter(tags=["dashboard"])


# ---------------------------------------------------------------------------
# Dashboard HTML page
# ---------------------------------------------------------------------------


@router.get("/dashboard/", response_class=FileResponse)
async def dashboard_index():
    """Serve the dashboard HTML page."""
    html_path = BASE_DIR / "web" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard page not found")
    return FileResponse(html_path)


# ---------------------------------------------------------------------------
# SSE log streaming
# ---------------------------------------------------------------------------


@router.get("/api/logs")
async def api_logs():
    """SSE streaming endpoint for live logs."""
    if _log_handler is None:
        raise HTTPException(status_code=404, detail="SSE live log disabled (set server.sse_log: true or SSE_LOG_ENABLED=1)")

    from log_handler import sse_log_generator

    async def event_stream():
        try:
            async for event in sse_log_generator(_log_handler):
                yield event
        except asyncio.CancelledError:
            pass

    return StreamingResponse(
        content=event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Config CRUD
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8787
    dashboard: bool = True
    debug_log: bool = False


class MaxPriceModel(BaseModel):
    input: float = Field(0.0, ge=0.0)
    completion: float = Field(0.0, ge=0.0)
    cache: Optional[float] = Field(0.0, ge=0.0)


class ModelConfig(BaseModel):
    quantizations: list[str] = Field(..., min_length=1)
    providers: list[str] = Field(..., min_length=1)
    max_price: MaxPriceModel = Field(...)


class MigrationConfig(BaseModel):
    enabled: bool = True
    hysteresis_mult: float = Field(3.0, ge=0.0)
    est_turns_per_session: int = Field(50, ge=1)
    r_cache_estimate: int = Field(300000, ge=0)
    out_per_turn_estimate: int = Field(40000, ge=0)


class RefreshConfig(BaseModel):
    interval_minutes: int = Field(30, ge=1, le=1440)
    # In PERCENT POINTS (10 = 10%, default 1 = 1%).
    price_change_threshold: float = Field(1.0, ge=0.0)
    times: list[Any] = []


class HealthConfig(BaseModel):
    initial_cooldown_seconds: int = Field(300, ge=0)
    consecutive_threshold: int = Field(3, ge=1)
    escalation_seconds: list[int] = Field([3600, 43200], min_length=1)
    max_cooldown_seconds: int = Field(43200, ge=0)


class RetryConfig(BaseModel):
    max_attempts: int = Field(1, ge=1)
    delay_seconds: float = Field(0.0, ge=0.0)


class AttributionConfig(BaseModel):
    mode: str = Field("passthrough")
    headers: dict[str, str] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}

    @field_validator("mode")
    @classmethod
    def _mode_valid(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in ("passthrough", "fallback", "force"):
            raise ValueError("attribution.mode must be passthrough|fallback|force")
        return v


class ConfigSchema(BaseModel):
    server: ServerConfig = ServerConfig()
    models: dict[str, ModelConfig] = Field(..., min_length=0)
    migration: MigrationConfig = MigrationConfig()
    refresh: RefreshConfig = RefreshConfig()
    health: HealthConfig = HealthConfig()
    retry: RetryConfig = RetryConfig()
    attribution: AttributionConfig = AttributionConfig()


async def _read_config_yaml() -> str:
    """Read raw config file content."""
    return await asyncio.to_thread(
        lambda: CONFIG_YAML_PATH.read_text() if CONFIG_YAML_PATH.exists() else ""
    )


async def _read_config_dict() -> dict:
    """Parse config YAML to dict."""
    raw = await _read_config_yaml()
    return yaml.safe_load(raw) or {} if raw else {}


def _validate_config_schema(data: dict) -> ConfigSchema:
    """Raise HTTPException(400) on invalid config."""
    try:
        return ConfigSchema(**data)
    except ValidationError as e:
        errors = []
        for err in e.errors():
            path = " -> ".join(str(p) for p in err["loc"])
            errors.append(f"{path}: {err['msg']}")
        raise HTTPException(
            status_code=400,
            detail={"error": "Invalid config", "details": errors},
        )


async def _backup_config() -> Path:
    """Create a timestamped backup of routing_config.yaml."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = BASE_DIR / "data"
    backup_dir.mkdir(exist_ok=True)
    dest = backup_dir / f"routing_config.yaml.backup.{timestamp}"
    await asyncio.to_thread(
        lambda: shutil.copy2(CONFIG_YAML_PATH, dest)
    )
    return dest


@router.get("/api/config")
async def get_config():
    """Return current config as JSON."""
    data = await _read_config_dict()
    return data


@router.get("/api/config/raw")
async def get_config_raw():
    """Return raw YAML text."""
    raw = await _read_config_yaml()
    return PlainTextResponse(raw, media_type="text/plain")


@router.put("/api/config")
async def put_config(request: Request):
    """Validate and save updated config."""
    raw_bytes = await request.body()
    raw_text = raw_bytes.decode("utf-8").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty config body")

    # Parse YAML
    try:
        data = yaml.safe_load(raw_text)
        if not isinstance(data, dict):
            raise ValueError("Root must be a mapping")
    except (yaml.YAMLError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid YAML: {e}"
        )

    # Validate with pydantic
    _validate_config_schema(data)

    # Backup and write (protected by async lock)
    async with _config_lock:
        await _backup_config()
        await asyncio.to_thread(
            lambda: CONFIG_YAML_PATH.write_text(raw_text)
        )

    # Reload config in the running process
    from config import config
    config.load()

    return {
        "status": "ok",
        "message": "Config saved and reloaded",
        "backup": True,
    }


# ---------------------------------------------------------------------------
# Extended status
# ---------------------------------------------------------------------------


@router.get("/api/status")
async def api_status():
    """Extended status: merges router status + service info."""
    from main import get_scheduler

    scheduler = get_scheduler()
    uptime_seconds = int(time.monotonic() - _start_time)

    base_status = {}
    if _router_instance is not None:
        base_status = _router_instance.get_status()

    # Migration info from scheduler
    migration_info = {}
    if scheduler is not None:
        migration_info = {
            "config": {
                "enabled": config.migration_enabled,
                "hysteresis_mult": config.hysteresis_mult,
                "est_turns_per_session": config.est_turns_per_session,
                "r_cache_estimate": config.r_cache_estimate,
                "out_per_turn_estimate": config.out_per_turn_estimate,
            },
            "last_events": scheduler.get_migration_log()[-10:] if hasattr(scheduler, "get_migration_log") else [],
        }

    return {
        **base_status,
        "service": {
            "uptime_seconds": uptime_seconds,
            "dashboard": config.dashboard_enabled,
            "version": "0.1.0",
        },
        "migration": migration_info,
        "scheduler": {
            "running": getattr(scheduler, "_running", False),
            "next_refresh": scheduler.next_run.replace(tzinfo=timezone.utc).isoformat() if scheduler is not None and getattr(scheduler, "next_run", None) else None,
        } if scheduler is not None else {"running": False, "next_refresh": None},
        "debug_log": _debug_log_status(),
    }


def _debug_log_status() -> dict:
    """Debug-log state for the UI banner (import-guarded for tests)."""
    try:
        import debug_log
        return debug_log.status()
    except Exception:
        return {"enabled": bool(config.debug_log_enabled), "file": None,
                "size_bytes": 0, "last_write": None}


# ---------------------------------------------------------------------------
# Debug log toggle — surgical YAML edit (preserves comments/formatting)
# ---------------------------------------------------------------------------

@router.post("/api/debug/toggle")
async def api_debug_toggle(request: Request):
    """Enable/disable `server.debug_log` in routing_config.yaml.

    The flag is edited SURGICALLY in the raw YAML text (never yaml.dump —
    it would destroy comments and formatting), then config.load() makes the
    change effective immediately (dynamic read, no restart).
    Body: {"enabled": true|false}
    """
    data = await request.json()
    enabled = bool(data.get("enabled"))

    async with _config_lock:
        await _backup_config()

        def _apply():
            text = CONFIG_YAML_PATH.read_text(encoding="utf-8")
            lines_cfg = text.splitlines(keepends=True)
            value = "true" if enabled else "false"
            target = "debug_log: " + value
            in_server = False
            found = False
            out = []
            for line in lines_cfg:
                stripped = line.rstrip("\r\n")
                if stripped and not stripped[0].isspace() and not stripped.startswith("#"):
                    in_server = (stripped == "server:")
                if in_server and stripped.split("#", 1)[0].strip().startswith("debug_log:"):
                    indent = line[: len(line) - len(line.lstrip())]
                    eol = line[len(stripped):] or "\n"
                    comment = ("  #" + stripped.split("#", 1)[1]) if "#" in stripped else ""
                    out.append(indent + target + comment + eol)
                    found = True
                else:
                    out.append(line)
            if not found:
                # Insert right after the `server:` key with matching indentation
                new_lines = []
                inserted = False
                for line in lines_cfg:
                    new_lines.append(line)
                    if not inserted and line.rstrip("\r\n").rstrip() == "server:":
                        eol = "\r\n" if line.endswith("\r\n") else "\n"
                        new_lines.append("    debug_log: " + value + eol)
                        inserted = True
                if not inserted:
                    raise HTTPException(status_code=400, detail="server: section not found in config")
                text = "".join(new_lines)
            else:
                text = "".join(out)
            CONFIG_YAML_PATH.write_text(text, encoding="utf-8")
            return text, True

        await asyncio.to_thread(_apply)

    from config import config as live_cfg
    live_cfg.load()

    import debug_log
    return {"status": "ok", "enabled": debug_log.is_enabled(),
            "message": f"Debug request log {'ENABLED' if debug_log.is_enabled() else 'disabled'}"}


# ---------------------------------------------------------------------------
# System metrics (OS + proxy)
# ---------------------------------------------------------------------------


@router.get("/api/system")
async def api_system():
    """Return system metrics (OS + proxy)."""
    import psutil

    # OS metrics
    cpu_percent = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(".")
    # In a container, psutil.boot_time() returns the HOST kernel boot (shared /proc/stat btime),
    # not the container start. Use PID 1 create_time for the real LXC boot time.
    try:
        boot_ts = psutil.Process(1).create_time()
    except Exception:
        boot_ts = psutil.boot_time()
    boot_time = datetime.fromtimestamp(boot_ts, tz=timezone.utc).isoformat()

    os_data = {
        "platform": "Linux" if psutil.LINUX else os.name,
        "hostname": socket.gethostname(),
        "boot_time": boot_time,
    }
    memory_data = {
        "total_gb": round(mem.total / (1024 ** 3), 1),
        "available_gb": round(mem.available / (1024 ** 3), 1),
        "percent": mem.percent,
    }
    cpu_data = {
        "percent": cpu_percent,
        "count": psutil.cpu_count(),
    }
    disk_data = {
        "total_gb": round(disk.total / (1024 ** 3), 1),
        "used_gb": round(disk.used / (1024 ** 3), 1),
        "free_gb": round(disk.free / (1024 ** 3), 1),
        "percent": disk.percent,
    }

    # Proxy metrics
    sessions_count = 0
    providers_in_cooldown = 0
    if _router_instance is not None:
        status = _router_instance.get_status()
        sessions_count = len(status.get("sessions", {}))
        providers_in_cooldown = len(
            [p for p, s in status.get("backoff", {}).items()
             if s.get("in_cooldown", False)]
        )
    cached_models = list(
        _router_instance.endpoint_cache.get_all_cached().keys()
        if _router_instance is not None
        else []
    )

    return {
        "os": os_data,
        "memory": memory_data,
        "cpu": cpu_data,
        "disk": disk_data,
        "proxy": {
            "total_requests": get_request_count(),
            "total_errors": get_error_count(),
            "active_sessions": sessions_count,
            "providers_in_cooldown": providers_in_cooldown,
            "cached_models": cached_models,
            "migration_enabled": config.migration_enabled,
        },
    }


# ---------------------------------------------------------------------------
# Manual refresh
# ---------------------------------------------------------------------------


@router.post("/api/refresh")
async def api_refresh():
    """Trigger manual price refresh (delegates to scheduler)."""
    from main import get_scheduler
    scheduler = get_scheduler()
    if scheduler is None:
        raise HTTPException(status_code=500, detail="Scheduler not initialized")

    result = await scheduler.manual_refresh()
    return {
        "status": "ok",
        "refreshed_models": list(result.keys() if result else []),
        "message": f"Refreshed {len(result) if result else 0} models",
    }


# ---------------------------------------------------------------------------#
# Service restart (200 OK -> 500ms delay -> SIGTERM)
# ---------------------------------------------------------------------------


@router.post("/api/restart")
async def api_restart():
    """Restart the service.
    Returns 200 OK immediately, then sends SIGTERM after 500ms delay.
    The process manager (systemd) should restart the service automatically.
    """
    async def _delayed_shutdown():
        await asyncio.sleep(0.5)
        logger.info("Delayed shutdown triggered")
        # Disconnect all SSE clients first
        if _log_handler is not None:
            _log_handler.disconnect_all()
        os._exit(0)  # Force exit — systemd Restart=always riavvia
    asyncio.create_task(_delayed_shutdown())
    return {
        "status": "ok",
        "message": "Service restarting in 500ms",
    }


# ---------------------------------------------------------------------------
# Log download endpoint
# ---------------------------------------------------------------------------


@router.get("/api/logs/download")
async def api_logs_download(
    format: str = "jsonl",
    source: str = "proxy",
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = Query(None, alias="to"),
):
    """Download logs in various formats (jsonl, csv, txt)."""
    log_path = BASE_DIR / "logs" / ("proxy.jsonl" if source == "proxy" else "app.log")
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    # No filter: return raw file directly (only for jsonl/proxy or txt/app)
    if not from_ and not to:
        if format == "jsonl":
            if source == "proxy":
                return FileResponse(
                    log_path,
                    media_type="application/x-ndjson",
                    filename="proxy.jsonl",
                )
            else:
                return FileResponse(
                    log_path,
                    media_type="text/plain",
                    filename="app.log",
                )
        elif format == "txt" and source != "proxy":
            return FileResponse(
                log_path,
                media_type="text/plain",
                filename=f"{source}.log",
            )
        # For csv and txt/proxy, fall through to conversion below

    # Read, filter, convert
    raw = await asyncio.to_thread(lambda: log_path.read_text(encoding="utf-8", errors="replace"))
    lines = [l for l in raw.splitlines() if l.strip()]

    if source == "proxy" and (from_ or to):
        # Filter proxy.jsonl by time range
        filtered = []
        for line in lines:
            try:
                entry = json.loads(line)
                ts = entry.get("ts", "")
                if from_ and ts < from_:
                    continue
                if to and ts > to:
                    continue
                filtered.append(entry)
            except json.JSONDecodeError:
                continue
    else:
        filtered = [json.loads(l) for l in lines if l.strip()] if source == "proxy" else lines

    if format == "csv":
        # Convert to CSV
        import io
        output = io.StringIO()
        writer = csv.writer(output, quoting=csv.QUOTE_ALL)
        if source == "proxy":
            fields = ["ts", "type", "model", "provider", "tier", "session_id", "stream",
                      "status", "tokens_in", "tokens_out", "tokens_cached",
                      "tokens_reasoning", "cost", "latency_ms", "error",
                      "provider_response", "model_response"]
            writer.writerow(fields)
            for entry in filtered:
                writer.writerow([entry.get(f) for f in fields])
        else:
            writer.writerow(["line"])
            for line in filtered:
                writer.writerow([line])
        content = output.getvalue()
        return PlainTextResponse(
            content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=proxy.csv" if source == "proxy" else "attachment; filename=app.csv"},
        )
    elif format == "txt" and source == "proxy":
        # TXT from proxy data: format as readable lines
        content = "\n".join(
            f"{e.get('ts','')} [{e.get('status','')}] {e.get('model','')} → {e.get('provider','')} "
            f"in={e.get('tokens_in','')} out={e.get('tokens_out','')} cost={e.get('cost','')} "
            f"lat={e.get('latency_ms','')}ms"
            for e in filtered
        )
        return PlainTextResponse(
            content,
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=proxy.txt"},
        )
    elif format == "txt":
        # Raw app.log text
        return PlainTextResponse(
            "\n".join(filtered),
            media_type="text/plain",
            headers={"Content-Disposition": "attachment; filename=app.log"},
        )
    else:
        # jsonl (filtered)
        content = "\n".join(json.dumps(e) for e in filtered) + "\n"
        return PlainTextResponse(
            content,
            media_type="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=proxy.jsonl"},
        )


# ---------------------------------------------------------------------------
# Log recent (polling endpoint for dashboard)
# ---------------------------------------------------------------------------


@router.get("/api/logs/recent")
async def api_logs_recent(limit: int = Query(50, ge=1, le=500)):
    """Return the last N structured log entries (polling for dashboard)."""
    log_path = BASE_DIR / "logs" / "proxy.jsonl"
    if not log_path.exists():
        return {"data": [], "total": 0}

    raw = await asyncio.to_thread(lambda: log_path.read_text(encoding="utf-8", errors="replace"))
    lines = [l for l in raw.splitlines() if l.strip()]
    tail = lines[-limit:]

    entries = []
    for line in tail:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    return {"data": entries, "total": len(entries)}


# ---------------------------------------------------------------------------
# Model catalog (fetch on-demand from OpenRouter API)
# ---------------------------------------------------------------------------


@router.get("/api/models/catalog")
async def api_models_catalog():
    """Fetch and return the filtered OpenRouter model catalog.
    On-demand, no server-side caching. Timeout 30s.
    """
    api_key = config.openrouter_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to fetch model catalog: {e}",
            )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter returned {resp.status_code} for model catalog",
        )

    data = resp.json()
    raw_models = data.get("data", [])

    filtered = []
    for m in raw_models:
        pricing = m.get("pricing", {})
        prompt_price = pricing.get("prompt", "0")
        completion_price = pricing.get("completion", "0")
        free = prompt_price == "0" and completion_price == "0"

        filtered.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "context_length": m.get("context_length"),
            "pricing": {
                "prompt": _to_per_million(prompt_price),
                "completion": _to_per_million(completion_price),
                "input_cache_read": _to_per_million(pricing.get("input_cache_read", "0")),
            },
            "free": free,
            "architecture": {
                "modality": m.get("architecture", {}).get("modality", "text->text"),
            },
            "reasoning": m.get("reasoning", {"mandatory": False}),
        })

    return {"data": filtered, "total": len(filtered)}


# ---------------------------------------------------------------------------
# Model endpoints (reuse EndpointFetcher)
# ---------------------------------------------------------------------------


@router.get("/api/models/{model_id:path}/endpoints")
async def api_model_endpoints(model_id: str):
    """Return provider endpoints and pricing for a specific model.

    Pricing is converted to $/M tokens for display. The underlying cache
    (fetcher) keeps per-token values untouched.
    """
    if _fetcher is None:
        raise HTTPException(status_code=500, detail="Endpoint fetcher not initialized")

    result = await _fetcher.fetch_model_endpoints(model_id)
    if result is None:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to fetch endpoints for {model_id}",
        )

    # Convert pricing to $/M on a display copy — never mutate the cached data.
    display = dict(result)
    display_endpoints = []
    for ep in result.get("endpoints", []):
        ep_copy = dict(ep)
        pr = ep.get("pricing") or {}
        ep_copy["pricing"] = {
            "prompt": _to_per_million(pr.get("prompt", "0")),
            "completion": _to_per_million(pr.get("completion", "0")),
            "input_cache_read": _to_per_million(pr.get("input_cache_read", "0")),
        }
        display_endpoints.append(ep_copy)
    display["endpoints"] = display_endpoints

    return display


# ---------------------------------------------------------------------------
# Config: delete model
# ---------------------------------------------------------------------------


@router.delete("/api/config/models/{model_id:path}")
async def api_config_delete_model(model_id: str):
    """Remove a model from config, with backup and reload."""
    async with _config_lock:
        config_dict = await _read_config_dict()
        models = config_dict.get("models", {})
        if model_id not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found in config",
            )
        del models[model_id]
        config_dict["models"] = models

        # Backup before writing
        await _backup_config()

        # Write updated config
        raw = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
        await asyncio.to_thread(lambda: CONFIG_YAML_PATH.write_text(raw))

    # Reload config in the running process
    from config import config
    config.load()

    return {
        "status": "ok",
        "message": f"Model '{model_id}' removed and config reloaded",
        "backup": True,
    }


# ---------------------------------------------------------------------------
# Config: export (download routing_config.yaml)
# ---------------------------------------------------------------------------


@router.api_route("/api/config/export", methods=["GET", "POST"])
async def api_config_export():
    """Download the current routing_config.yaml."""
    if not CONFIG_YAML_PATH.exists():
        raise HTTPException(status_code=404, detail="Config file not found")
    return FileResponse(
        CONFIG_YAML_PATH,
        media_type="application/x-yaml",
        filename="routing_config.yaml",
    )


# ---------------------------------------------------------------------------
# Logs: delete all log files (incl. rotated) — GDPR compliance
# ---------------------------------------------------------------------------


@router.delete("/api/logs")
async def api_logs_delete_all():
    """Delete all log files, including rotated/compressed versions (GDPR).

    Active log files (app.log, proxy.jsonl, requests.jsonl) are TRUNCATED
    (kept inode) so the FileHandler keeps writing to the same file;
    rotated/compressed versions (*.log-*, *.jsonl-*, *.gz) are physically
    removed.
    """
    import glob
    import os
    logs_dir = BASE_DIR / "logs"
    truncated = []
    removed = []
    # Active files held open by FileHandler: truncate to 0 (keep inode)
    for name in ("app.log", "proxy.jsonl", "requests.jsonl"):
        p = logs_dir / name
        if p.exists():
            try:
                with open(p, "w"):
                    pass
                truncated.append(name)
            except OSError as e:
                logger.error(f"Log truncate failed {name}: {e}")
    # Rotated/compressed versions: remove physically
    for pat in ("app.log-*", "app.log.*", "proxy.jsonl-*", "proxy.jsonl.*",
                "requests.jsonl-*", "requests.jsonl.*", "*.gz"):
        for f in glob.glob(str(logs_dir / pat)):
            try:
                os.remove(f)
                removed.append(os.path.basename(f))
            except OSError:
                pass
    result = {"truncated": truncated, "removed": removed}
    logger.info(f"Log purge: truncated {truncated}, removed {removed}")
    return {"status": "ok", "deleted": result, "message": f"Log purge: truncated {len(truncated)}, removed {len(removed)}"}


# ---------------------------------------------------------------------------
# Config: import (upload YAML, backup, validate, reload)
# ---------------------------------------------------------------------------


@router.post("/api/config/import")
async def api_config_import(request: Request):
    """Import a replacement routing_config.yaml with backup and validation."""
    raw_bytes = await request.body()
    raw_text = raw_bytes.decode("utf-8").strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Empty config body")

    # Parse YAML
    try:
        data = yaml.safe_load(raw_text)
        if not isinstance(data, dict):
            raise ValueError("Root must be a mapping")
    except (yaml.YAMLError, ValueError) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid YAML: {e}"
        )

    # Validate with pydantic
    _validate_config_schema(data)

    # Backup and write (protected by async lock)
    async with _config_lock:
        await _backup_config()
        await asyncio.to_thread(
            lambda: CONFIG_YAML_PATH.write_text(raw_text)
        )

    # Reload config in the running process
    from config import config
    config.load()

    return {
        "status": "ok",
        "message": "Config imported and reloaded",
        "backup": True,
    }