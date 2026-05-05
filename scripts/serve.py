"""
Run the FastAPI app locally with uvicorn.

Usage:
    python -m scripts.serve
    python -m scripts.serve --port 8080      # alternate port
    python -m scripts.serve --reload         # auto-reload on code changes (dev)

What you'll see:
  - Server starts at http://localhost:8000
  - Swagger UI at http://localhost:8000/docs
  - Health check at http://localhost:8000/health

For production (Render), this script isn't used — Render runs uvicorn directly
via a start command (configured in render.yaml later).
"""

from __future__ import annotations

import argparse
import logging

# Load .env BEFORE importing app modules so LangSmith @traceable picks up
# tracing env vars at module-import time.
from dotenv import load_dotenv

load_dotenv()

import uvicorn

from app.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GraphRAG FastAPI server.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: localhost only)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-reload on code changes. Dev only — uses watchdog and reduces performance.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("serve")
    log.info("Starting GraphRAG API on http://%s:%d", args.host, args.port)
    log.info("  - Health:  http://%s:%d/health", args.host, args.port)
    log.info("  - Docs:    http://%s:%d/docs", args.host, args.port)
    log.info("  - Ask:     POST http://%s:%d/ask", args.host, args.port)

    uvicorn.run(
        "app.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
