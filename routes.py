"""HTTP routes — FastAPI endpoints with SSE streaming support."""

import json
import logging
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
) -> AsyncGenerator[str, None]:
    """Forward streaming response from OpenRouter as SSE."""
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
                    # Pass through the line as-is (upstream already has "data: " prefix)
                    yield f"{line}\n\n"

                # End of stream
                yield "[DONE]\n\n"

    except httpx.ConnectError as e:
        logger.error(f"Connect error to {provider_slug}: {e}")
        _router.record_error(provider_slug)
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
            _forward_stream(upstream_body, provider_slug),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    else:
        return await _forward_non_stream(upstream_body, provider_slug)


async def _forward_non_stream(
    body: dict,
    provider_slug: str,
) -> dict:
    """Forward non-streaming request to OpenRouter."""
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
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"Upstream error: {error_text.decode()}",
                )
            _router.record_success(provider_slug)
            return resp.json()

    except httpx.ConnectError as e:
        logger.error(f"Connect error to {provider_slug}: {e}")
        _router.record_error(provider_slug)
        raise HTTPException(
            status_code=502,
            detail=f"Provider {provider_slug} unreachable: {e}",
        )
    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting to {provider_slug}: {e}")
        _router.record_error(provider_slug)
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
