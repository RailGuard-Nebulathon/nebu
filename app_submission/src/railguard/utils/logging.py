"""Consistent logging configuration."""

import logging
import os


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=getattr(logging, (level or os.getenv("RAILGUARD_LOG_LEVEL", "INFO")).upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

