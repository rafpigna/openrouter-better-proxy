"""HTTP routes — FastAPI endpoints with SSE streaming support."""

import asyncio
import json
import logging
import time
from datetime import datetime
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

# Per (session_id, provider) last cached tokens — to detect silent cache drops
# (same session+provider going from cached>0 to cached=0 across requests).
_last_cache: dict[tuple[str, str], int] = {}


def _is_transient_status(status: int) -> bool:
    """True for retryable pre-stream errors: 429 and 5xx.

    Definitive 4xx (400/404/422...) are NOT retried (no point retrying a
    bad request), and mid-stream failures are handled separately (never
    retryable)."""
    return status == 429 or status >= 500


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
    """Write a structured JSON line to proxy.jsonl.

    `ts` is the proxy's LOCAL time (LXC is Europe/Rome; uniform with app.log).
    Detects and flags silent cache loss: same session+provider had cached>0 on
    a prior request and now returns cached=0 (OpenRouter per-endpoint caching).
    """
    global _last_cache
    tokens_cached = usage.get("prompt_tokens_details", {}).get("cached_tokens") if usage else None

    # Silent cache-drop detection (only meaningful for successful requests)
    cache_drop = False
    if status_code == 200 and session_id and tokens_cached is not None:
        key = (session_id, provider_slug)
        prev = _last_cache.get(key)
        cur = int(tokens_cached or 0)
        if prev is not None and prev > 0 and cur == 0:
            cache_drop = True
            logger.warning(
                f"[CACHE-DROP] session {session_id} provider {provider_slug}: "
                f"cached {prev} -> 0 (same provider + same session)"
            )
        _last_cache[key] = cur

    entry = {
        "ts": datetime.now().isoformat(),
        "type": "error" if error_message or (status_code != 200 and status_code != 0) else "request",
        "model": model_id,
        "provider": provider_slug,
        "tier": tier,
        "session_id": session_id,
        "stream": stream,
        "status": status_code,
        "tokens_in": usage.get("prompt_tokens") if usage else None,
        "tokens_out": usage.get("completion_tokens") if usage else None,
        "tokens_cached": tokens_cached,
        "tokens_reasoning": usage.get("completion_tokens_details", {}).get("reasoning_tokens") if usage else None,
        "cost": usage.get("cost") if usage else None,
        "latency_ms": latency_ms,
        "error": error_message,
        "provider_response": provider_response,
        "model_response": model_response,
        "cache_drop": cache_drop,
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
    candidates: list[tuple[str, str]],
) -> AsyncGenerator[str, None]:
    """Forward streaming response from OpenRouter as SSE, with in-process
    failover across the AUTHORIZED candidates only.

    Each candidate is attempted at most once per request (pre-stream errors:
    429, 5xx, timeout, connect → next candidate). The session is bound to the
    provider that actually returns a 200. If every authorized candidate fails,
    an error SSE block is returned (never a provider outside the list).
    """
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

    last_status = None
    last_error_text = None

    max_attempts = max(1, config.retry_max_attempts)
    delay = max(0.0, config.retry_delay_seconds)

    for provider_slug, tier in candidates:
        # Pin this attempt to the authorized provider
        upstream = dict(body)
        upstream["provider"] = {"only": [provider_slug]}

        # Per-provider retry outcome (last failed attempt)
        provider_status = None
        provider_text = None

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                await asyncio.sleep(delay)

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream("POST", url, json=upstream, headers=headers) as resp:
                        if resp.status_code != 200:
                            err_body = await resp.aread()
                            provider_status = resp.status_code
                            provider_text = err_body.decode()
                            logger.warning(
                                f"Attempt {attempt}/{max_attempts} {provider_slug}: "
                                f"HTTP {resp.status_code} (treating as "
                                f"{'retryable' if _is_transient_status(resp.status_code) else 'definitive'})"
                            )
                            # Definitive 4xx: no retry, failover now
                            if not _is_transient_status(resp.status_code):
                                break
                            # Transient: try again (next attempt)
                            continue

                        # Success: bind session to the winning provider
                        _router.record_success(provider_slug)
                        _router.bind_session(session_id, provider_slug)

                        usage = None
                        provider_response = None
                        model_response = None

                        try:
                            # Stream SSE chunks byte-by-byte. Upstream already sends
                            # "data: {...}" format, pass through as-is.
                            async for line in resp.aiter_lines():
                                if not line:
                                    continue
                                if "OPENROUTER PROCESSING" in line:
                                    logger.debug(f"Skipping upstream status message: {line[:100]}")
                                    continue
                                if line.startswith("data: ") and line[6:] != "[DONE]":
                                    try:
                                        chunk = json.loads(line[6:])
                                        if "usage" in chunk:
                                            usage = chunk["usage"]
                                        if provider_response is None and chunk.get("provider"):
                                            provider_response = chunk.get("provider")
                                        if model_response is None and chunk.get("model"):
                                            model_response = chunk.get("model")
                                    except json.JSONDecodeError:
                                        pass
                                yield f"{line}\n\n"
                        except (httpx.ReadError, httpx.ProtocolError, httpx.HTTPError) as e:
                            # Mid-stream failure: not revocable (DESIGN §4). Log it
                            # as an error and cut the stream.
                            _router.record_error(provider_slug)
                            ERROR_COUNT += 1
                            _write_proxy_log(
                                model_id, provider_slug, tier, session_id, True,
                                502, None, int((time.monotonic() - start) * 1000),
                                f"mid-stream: {e}", provider_response, model_response,
                            )
                            error_data = {
                                "error": {
                                    "message": f"Provider {provider_slug} stream interrupted: {e}",
                                    "type": "stream_error",
                                }
                            }
                            yield f"data: {json.dumps(error_data)}\n\n"
                            yield "data: [DONE]\n\n"
                            return
                        except asyncio.CancelledError:
                            # Client disconnected / aborted the stream. The upstream
                            # request DID happen (and may be billed), so log it.
                            _write_proxy_log(
                                model_id, provider_slug, tier, session_id, True,
                                200, usage, int((time.monotonic() - start) * 1000),
                                None, provider_response, model_response,
                            )
                            raise

                        # End of stream — write proxy log
                        _write_proxy_log(
                            model_id, provider_slug, tier, session_id, True,
                            200, usage, int((time.monotonic() - start) * 1000),
                            None, provider_response, model_response,
                        )
                        yield "data: [DONE]\n\n"
                        return

            except httpx.ConnectError as e:
                provider_status = 502
                provider_text = str(e)
                logger.warning(f"Attempt {attempt}/{max_attempts} {provider_slug}: connect error (retryable)")
                # transient -> try again
                continue

            except httpx.TimeoutException as e:
                provider_status = 504
                provider_text = str(e)
                logger.warning(f"Attempt {attempt}/{max_attempts} {provider_slug}: timeout (retryable)")
                continue

            except httpx.HTTPError as e:
                provider_status = 502
                provider_text = str(e)
                logger.warning(f"Attempt {attempt}/{max_attempts} {provider_slug}: HTTP error (retryable)")
                continue

        # Out of the per-provider attempt loop (retries exhausted or definitive
        # error): this provider failed for the request — record ONE error and
        # fail over to the next authorized candidate.
        _router.record_error(provider_slug)
        ERROR_COUNT += 1
        _write_proxy_log(
            model_id, provider_slug, tier, session_id, True,
            provider_status or 503, None, int((time.monotonic() - start) * 1000),
            provider_text or "no response", None, None,
        )
        last_status = provider_status or 503
        last_error_text = provider_text or "no response"

    # Exhausted all authorized candidates with no success
    if last_status is not None:
        message = f"All authorized providers failed (last: {last_status} {last_error_text})"
        error_code = last_status
    else:
        message = "No authorized provider available"
        error_code = 503
    logger.error(message)
    error_data = {
        "error": {
            "message": message,
            "type": "upstream_error",
            "code": error_code,
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

    # Select the ordered list of AUTHORIZED candidates (allowlist gate applied
    # inside the router). Never a provider outside the configured list.
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    candidates = _router.select_candidates(
        model_id=model_id,
        session_id=session_id,
        tools=has_tools,
    )

    if not candidates:
        # No authorized, usable provider: honest error to the client. The proxy
        # NEVER falls back to a provider that isn't configured.
        raise HTTPException(
            status_code=400,
            detail=f"No authorized provider available for model {model_id}",
        )

    provider_slug, tier = candidates[0]
    logger.info(f"Routing {model_id} → {provider_slug} (tier={tier}) [candidates={len(candidates)}]")

    # Build upstream request (provider pinned per-attempt by the forward loop)
    upstream_body = {
        "model": model_id,
        "messages": messages,
        "stream": stream,
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

    # Return streaming or non-streaming response. Both run the in-process
    # failover loop over the authorized `candidates` only.
    if stream:
        return StreamingResponse(
            _forward_stream(upstream_body, candidates),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    else:
        return await _forward_non_stream(upstream_body, candidates)


async def _forward_non_stream(
    body: dict,
    candidates: list[tuple[str, str]],
) -> dict:
    """Forward non-streaming request to OpenRouter, with in-process failover
    across the AUTHORIZED candidates only. Returns the first successful
    response body; if none succeeds, raises an error to the calling client."""
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

    last_status = None
    last_error_text = None

    max_attempts = max(1, config.retry_max_attempts)
    delay = max(0.0, config.retry_delay_seconds)

    for provider_slug, tier in candidates:
        upstream = dict(body)
        upstream["provider"] = {"only": [provider_slug]}

        provider_status = None
        provider_text = None

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                await asyncio.sleep(delay)

            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(url, json=upstream, headers=headers)

                if resp.status_code != 200:
                    err_body = await resp.aread()
                    provider_status = resp.status_code
                    provider_text = err_body.decode()
                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} {provider_slug}: HTTP {resp.status_code} "
                        f"(treating as {'retryable' if _is_transient_status(resp.status_code) else 'definitive'})"
                    )
                    if not _is_transient_status(resp.status_code):
                        break
                    # transient -> retry
                    continue

                # Success: bind session to the winning provider
                _router.record_success(provider_slug)
                _router.bind_session(session_id, provider_slug)

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
                provider_status = 502
                provider_text = str(e)
                logger.warning(f"Attempt {attempt}/{max_attempts} {provider_slug}: connect error (retryable)")
                continue

            except httpx.TimeoutException as e:
                provider_status = 504
                provider_text = str(e)
                logger.warning(f"Attempt {attempt}/{max_attempts} {provider_slug}: timeout (retryable)")
                continue

            except httpx.HTTPError as e:
                provider_status = 502
                provider_text = str(e)
                logger.warning(f"Attempt {attempt}/{max_attempts} {provider_slug}: HTTP error (retryable)")
                continue

        # Out of the per-provider attempt loop: this provider failed.
        _router.record_error(provider_slug)
        ERROR_COUNT += 1
        _write_proxy_log(
            model_id, provider_slug, tier, session_id, False,
            provider_status or 503, None, int((time.monotonic() - start) * 1000),
            provider_text or "no response", None, None,
        )
        last_status = provider_status or 503
        last_error_text = provider_text or "no response"

    # Exhausted all authorized candidates with no success
    if last_status is not None:
        message = f"All authorized providers failed (last: {last_status} {last_error_text})"
        error_code = last_status
    else:
        message = f"No authorized provider available for model {model_id}"
        error_code = 503
    logger.error(message)
    raise HTTPException(status_code=error_code, detail=message)


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
