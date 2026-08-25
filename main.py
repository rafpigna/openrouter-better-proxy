"""Main entry point — uvicorn startup with scheduler."""

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from config import config
from backoff import BackoffManager
from session import SessionManager
from cache import EndpointCache
from router import Router
from fetcher import EndpointFetcher
from price_diff import PriceDiffDetector
from migration import PriceMigration
from scheduler import RefreshScheduler
from routes import router as routes_router, init_routes

# Configure logging
import os
os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/app.log"),
    ],
)
logger = logging.getLogger(__name__)

# Global scheduler reference
_scheduler = None

# SSELogHandler (initialised during lifespan if dashboard is enabled)
_log_handler = None


def get_scheduler():
    """Get global scheduler instance."""
    return _scheduler


def get_log_handler():
    """Get global SSE log handler instance."""
    return _log_handler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: start scheduler on init, stop on shutdown."""
    global _scheduler, _log_handler

    # Initialize components
    backoff = BackoffManager(
        initial_cooldown=config.initial_cooldown_seconds,
        consecutive_threshold=config.consecutive_threshold,
        escalation_seconds=config.escalation_seconds,
        max_cooldown=config.max_cooldown_seconds,
    )
    sessions = SessionManager()
    cache = EndpointCache(data_dir="data")
    router = Router(backoff, sessions, cache)

    # Initialize fetcher and scheduler
    fetcher = EndpointFetcher(api_key=config.openrouter_api_key, cache=cache)
    diff_detector = PriceDiffDetector(snapshot_dir="data")
    price_migration = PriceMigration(
        hysteresis_mult=config.hysteresis_mult,
        est_turns_per_session=config.est_turns_per_session,
        r_cache_estimate=config.r_cache_estimate,
        out_per_turn_estimate=config.out_per_turn_estimate,
    )
    _scheduler = RefreshScheduler(fetcher, cache, diff_detector, router, sessions, price_migration)

    # Wire routes
    init_routes(router, cache)

    # Initialise dashboard components if enabled
    if config.dashboard_enabled:
        from log_handler import SSELogHandler
        from web_routes import router as web_router, init_web_routes

        # SSELogHandler: capture all INFO+ log records
        _log_handler = SSELogHandler(level=logging.INFO)
        _log_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(_log_handler)

        # Inject dependencies into web_routes
        init_web_routes(_scheduler, router, _log_handler, fetcher)

        # Include web routes
        app.include_router(web_router)

        logger.info("Dashboard enabled — web routes active")
    else:
        logger.info("Dashboard disabled — web routes inactive")

    logger.info(f"Starting server on {config.host}:{config.port}")
    logger.info(f"Models configured: {list(config.models.keys())}")

    # Start scheduler
    await _scheduler.start()

    # Wait briefly for network to stabilize before first refresh
    await asyncio.sleep(2)

    # Initial refresh to populate endpoint cache immediately
    await _scheduler.manual_refresh()

    yield

    # Shutdown: disconnect all SSE clients FIRST — prima che uvicorn provi
    # a chiudere le connessioni in graceful shutdown (che hang perché le
    # SSE sono long-lived). Vedi DESIGN per analisi opzione A vs B.
    if _log_handler is not None:
        logger.info("Shutdown: disconnecting SSE clients")
        _log_handler.disconnect_all()

    # Stop scheduler on shutdown
    await _scheduler.stop()
    await fetcher.close()


# Create main app with lifespan
app = FastAPI(
    title="OpenRouter Router Proxy",
    description="Proxy with intelligent provider selection",
    version="0.1.0",
    lifespan=lifespan,
)

# Include core routes (always present)
app.include_router(routes_router)


def main():
    """Start uvicorn."""
    import argparse

    parser = argparse.ArgumentParser(description="OpenRouter Better Proxy")
    parser.add_argument("--noweb", action="store_true",
                        help="Disable dashboard (overrides config.yaml)")
    args = parser.parse_args()

    # --noweb overrides dashboard enablement for python main.py runs
    if args.noweb:
        import os as _os
        _os.environ["DASHBOARD_DISABLED"] = "1"
        print("Dashboard disabled via --noweb flag")

    import uvicorn

    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()