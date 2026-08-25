"""HTTP routes — FastAPI endpoints with SSE streaming support."""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from router import Router
from cache import EndpointCache
from config import config

logger = logging.getLogger(__name__)

# Create router (separate from main app for lifespan compatibility)
router = APIRouter()

# Router instance (injected by main)
_router: Optional[Router] = None
_endpoint_cache: Optional[EndpointCache] = None

# Request counter (incremented on each /v1/chat/completions)
REQUEST_COUNT = 0
ERROR_COUNT = 0

# Proxy logger (writes structured JSON Lines to logs/proxy.jsonl)
_proxy_logger = logging.getLogger("proxy")
_proxy_logger.setLevel(logging.INFO)
_proxy_logger.propagate = False
_handler = logging.FileHandler("logs/proxy.jsonl")
_handler.setFormatter(logging.Formatter("%(message)s"))
_proxy_logger.addHandler(_handler)


def _write_proxy_log(
    model_id: str,
    provider_slug: str,
    tier: str,
    session_id: str | None,
    stream: bool,
    status_code: int,
    usage: dict | None,
    latency_ms: int | None,
    error_message: str | None,
    provider_response: str | None,
    model_response: str | None,
) -> None:
    """Write a structured JSON line to proxy.jsonl."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "type": "error" if error_message or (status_code != 200 and status_code != 0) else "request",
        "model": model_id,
        "provider": provider_slug,
        "tier": tier,
        "session_id": session_id,
        "stream": stream,
        "status": status_code,
        "tokens_in": usage.get("prompt_tokens") if usage else None,
        "tokens_out": usage.get("completion_tokens") if usage else None,
        "tokens_cached": usage.get("prompt_tokens_details", {}).get("cached_tokens") if usage else None,
        "tokens_reasoning": usage.get("completion_tokens_details", {}).get("reasoning_tokens") if usage else None,
        "cost": usage.get("cost") if usage else None,
        "latency_ms": latency_ms,
        "error": error_message,
        "provider_response": provider_response,
        "model_response": model_response,
    }
    _proxy_logger.info(json.dumps(entry))


def init_routes(router_instance: Router, endpoint_cache: EndpointCache) -> None:
    """Inject router and cache into routes module."""
    global _router, _endpoint_cache
    _router = router_instance
    _endpoint_cache = endpoint_cache


@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@router.get("/v1/models")
async def list_models():
    """Return curated model list from config."""
    from pydantic import BaseModel

    class ModelResponse(BaseModel):
        id: str
        object: str = "model"
        created: int = 0
        owned_by: str = "router"

    models = []
    for model_id in config.models:
        models.append(ModelResponse(id=model_id))
    return {"data": models, "object": "list"}


async def _forward_stream(
    body: dict,
    provider_slug: str,
    tier: str,
) -> AsyncGenerator[str, None]:
    """Forward streaming response from OpenRouter as SSE."""
    global ERROR_COUNT
    start = time.monotonic()
    model_id = body.get("model", "")
    session_id = body.get("session_id")
    api_key = config.openrouter_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = "https://openrouter.ai/api/v1/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    logger.error(f"Upstream error {resp.status_code}: {error_text}")
                    _router.record_error(provider_slug)
                    ERROR_COUNT += 1
                    latency_ms = int((time.monotonic() - start) * 1000)
                    _write_proxy_log(
                        model_id, provider_slug, tier, session_id, True,
                        resp.status_code, None, latency_ms, error_text.decode(), None, None,
                    )
                    # Return error as SSE event
                    error_data = {
                        "error": {
                            "message": f"Upstream error: {error_text.decode()}",
                            "type": "upstream_error",
                            "code": resp.status_code,
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                _router.record_success(provider_slug)

                usage = None
                provider_response = None
                model_response = None

                # Stream SSE chunks byte-by-byte
                # Upstream already sends "data: {...}" format, pass through as-is
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    # Skip "OPENROUTER PROCESSING" status messages from upstream
                    # These are transient and don't indicate an error
                    if "OPENROUTER PROCESSING" in line:
                        logger.debug(f"Skipping upstream status message: {line[:100]}")
                        continue
                    # Intercept usage in the chunk before [DONE]
                    if line.startswith("data: ") and line[6:] != "[DONE]":
                        try:
                            chunk = json.loads(line[6:])
                            if "usage" in chunk:
                                usage = chunk["usage"]
                            # Capture provider/model from response chunks
                            if provider_response is None and chunk.get("provider"):
                                provider_response = chunk.get("provider")
                            if model_response is None and chunk.get("model"):
                                model_response = chunk.get("model")
                        except json.JSONDecodeError:
                            pass
                    # Pass through the line as-is (upstream already has "data: " prefix)
                    yield f"{line}\n\n"

                # End of stream — write proxy log
                latency_ms = int((time.monotonic() - start) * 1000)
                _write_proxy_log(
                    model_id, provider_slug, tier, session_id, True,
                    200, usage, latency_ms, None, provider_response, model_response,
                )
                yield "[DONE]\n\n"

    except httpx.ConnectError as e:
        logger.error(f"Connect error to {provider_slug}: {e}")
        _router.record_error(provider_slug)
        ERROR_COUNT += 1
        latency_ms = int((time.monotonic() - start) * 1000)
        _write_proxy_log(
            model_id, provider_slug, tier, session_id, True,
            502, None, latency_ms, str(e), None, None,
        )
        error_data = {
            "error": {
                "message": f"Provider {provider_slug} unreachable: {e}",
                "type": "connect_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"

    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting to {provider_slug}: {e}")
        _router.record_error(provider_slug)
        ERROR_COUNT += 1
        latency_ms = int((time.monotonic() - start) * 1000)
        _write_proxy_log(
            model_id, provider_slug, tier, session_id, True,
            504, None, latency_ms, str(e), None, None,
        )
        error_data = {
            "error": {
                "message": f"Provider {provider_slug} timeout: {e}",
                "type": "timeout_error",
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
async def chat_completions(request: dict):
    """Route chat completion to best provider."""
    if _router is None:
        raise HTTPException(status_code=500, detail="Router not initialized")

    # Extract fields
    model_id = request.get("model", "")
    messages = request.get("messages", [])
    stream = request.get("stream", False)
    session_id = request.get("session_id")
    temperature = request.get("temperature")
    max_tokens = request.get("max_tokens")
    tools = request.get("tools")
    extra_body = request.get("extra_body", {})

    # Detect tools presence
    has_tools = bool(tools) or any(
        msg.get("role") == "system" and "tools" in str(msg.get("content", ""))
        for msg in messages
    )

    # Select provider
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    selection = _router.select_provider(
        model_id=model_id,
        session_id=session_id,
        tools=has_tools,
    )

    if not selection:
        raise HTTPException(
            status_code=400,
            detail=f"No valid provider found for model {model_id}",
        )

    provider_slug, tier = selection
    logger.info(f"Routing {model_id} → {provider_slug} (tier={tier})")

    # Build upstream request
    upstream_body = {
        "model": model_id,
        "messages": messages,
        "stream": stream,
        "provider": {"only": [provider_slug]},
    }

    # Forward compatible fields
    if temperature is not None:
        upstream_body["temperature"] = temperature
    if max_tokens is not None:
        upstream_body["max_tokens"] = max_tokens
    if tools:
        upstream_body["tools"] = tools

    # Forward session_id if present
    if session_id:
        upstream_body["session_id"] = session_id

    # Add any extra fields from extra_body
    upstream_body.update(extra_body)

    # Return streaming or non-streaming response
    if stream:
        return StreamingResponse(
            _forward_stream(upstream_body, provider_slug, tier),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    else:
        return await _forward_non_stream(upstream_body, provider_slug, tier)


async def _forward_non_stream(
    body: dict,
    provider_slug: str,
    tier: str,
) -> dict:
    """Forward non-streaming request to OpenRouter."""
    global ERROR_COUNT
    start = time.monotonic()
    model_id = body.get("model", "")
    session_id = body.get("session_id")
    api_key = config.openrouter_api_key
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    url = "https://openrouter.ai/api/v1/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=body, headers=headers)

            if resp.status_code != 200:
                error_text = await resp.aread()
                logger.error(f"Upstream error {resp.status_code}: {error_text}")
                _router.record_error(provider_slug)
                ERROR_COUNT += 1
                latency_ms = int((time.monotonic() - start) * 1000)
                _write_proxy_log(
                    model_id, provider_slug, tier, session_id, False,
                    resp.status_code, None, latency_ms, error_text.decode(), None, None,
                )
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Upstream error: {error_text.decode()}",
                )
            _router.record_success(provider_slug)
            response_data = resp.json()
            usage = response_data.get("usage")
            provider_response = response_data.get("provider")
            model_response = response_data.get("model")
            latency_ms = int((time.monotonic() - start) * 1000)
            _write_proxy_log(
                model_id, provider_slug, tier, session_id, False,
                200, usage, latency_ms, None, provider_response, model_response,
            )
            return response_data

    except httpx.ConnectError as e:
        logger.error(f"Connect error to {provider_slug}: {e}")
        _router.record_error(provider_slug)
        ERROR_COUNT += 1
        latency_ms = int((time.monotonic() - start) * 1000)
        _write_proxy_log(
            model_id, provider_slug, tier, session_id, False,
            502, None, latency_ms, str(e), None, None,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Provider {provider_slug} unreachable: {e}",
        )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting to {provider_slug}: {e}")
        _router.record_error(provider_slug)
        ERROR_COUNT += 1
        latency_ms = int((time.monotonic() - start) * 1000)
        _write_proxy_log(
            model_id, provider_slug, tier, session_id, False,
            504, None, latency_ms, str(e), None, None,
        )
        raise HTTPException(
            status_code=504,
            detail=f"Provider {provider_slug} timeout: {e}",
        )


@router.post("/refresh")
async def trigger_refresh():
    """Manual trigger for endpoint cache refresh."""
    from main import get_scheduler

    scheduler = get_scheduler()
    if scheduler is None:
        raise HTTPException(status_code=500, detail="Scheduler not initialized")

    result = await scheduler.manual_refresh()
    return {
        "status": "ok",
        "refreshed_models": list(result.keys()) if result else [],
        "message": f"Refreshed {len(result) if result else 0} models",
    }


@router.get("/status")
async def status():
    """Get router status (for debugging)."""
    if _router is None:
        raise HTTPException(status_code=500, detail="Router not initialized")

    # Get scheduler to access migration info
    from main import get_scheduler
    scheduler = get_scheduler()

    # Build base status
    status_data = _router.get_status()

    # Add migration info if scheduler available
    if scheduler is not None:
        # Get migration log
        migration_log = scheduler.get_migration_log()

        # Get migration config
        migration_config = {
            "enabled": config.migration_enabled,
            "hysteresis_mult": config.hysteresis_mult,
            "est_turns_per_session": config.est_turns_per_session,
            "r_cache_estimate": config.r_cache_estimate,
            "out_per_turn_estimate": config.out_per_turn_estimate,
        }

        # Add to status
        status_data["migration"] = {
            "config": migration_config,
            "last_events": migration_log[-10:],  # Last 10 events
        }

    return status_data
