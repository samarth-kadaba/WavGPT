"""structlog console configuration."""

import logging
import sys

import structlog


def configure_logging(use_json: bool = False, level: int = logging.INFO):
    """Configure structlog *and* the stdlib logging backend it routes through.

    Without a stdlib handler attached to the root logger, every structlog
    `logger.info(...)` is silently dropped. We attach a single stderr handler
    with a passthrough formatter so structlog's own renderer controls the
    final string."""
    renderer = structlog.processors.JSONRenderer() if use_json else structlog.dev.ConsoleRenderer()

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
