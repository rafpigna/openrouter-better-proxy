"""Main entry point — uvicorn startup with scheduler."""

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


def get_scheduler():
    """Get global scheduler instance."""
    return _scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: start scheduler on init, stop on shutdown."""
    global _scheduler

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

    logger.info(f"Starting server on {config.host}:{config.port}")
    logger.info(f"Models configured: {list(config.models.keys())}")

    # Start scheduler
    await _scheduler.start()

    yield

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

# Include routes
app.include_router(routes_router)


def main():
    """Start uvicorn."""
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
