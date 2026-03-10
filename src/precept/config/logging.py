"""
Logging Configuration for PRECEPT Framework.

Provides production-ready logging with:
- Structured logging support
- Multiple output handlers (console, file)
- Log level configuration via environment
- Colored console output (optional)
- JSON formatting for production

Usage:
    from precept.config.logging import get_logger, setup_logging

    # Setup logging (call once at application start)
    setup_logging(level="INFO", log_file="precept.log")

    # Get a logger for your module
    logger = get_logger(__name__)
    logger.info("Starting PRECEPT agent")
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LogConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_file: Optional[str] = None
    json_format: bool = False
    colorize: bool = True
    # Module-specific log levels
    module_levels: Dict[str, str] = field(default_factory=dict)


# ANSI color codes for console output
class Colors:
    """ANSI color codes for log levels."""

    RESET = "\033[0m"
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[32m"  # Green
    WARNING = "\033[33m"  # Yellow
    ERROR = "\033[31m"  # Red
    CRITICAL = "\033[35m"  # Magenta
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Log level to color mapping
    LEVEL_COLORS = {
        "DEBUG": DEBUG,
        "INFO": INFO,
        "WARNING": WARNING,
        "ERROR": ERROR,
        "CRITICAL": CRITICAL,
    }


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""

    def __init__(self, fmt: str, datefmt: str, colorize: bool = True):
        super().__init__(fmt, datefmt)
        self.colorize = colorize

    def format(self, record: logging.LogRecord) -> str:
        if not self.colorize:
            return super().format(record)

        # Store original values
        original_levelname = record.levelname
        original_msg = record.msg

        # Apply colors
        color = Colors.LEVEL_COLORS.get(record.levelname, Colors.RESET)
        record.levelname = f"{color}{record.levelname}{Colors.RESET}"

        # Format the record
        result = super().format(record)

        # Restore original values
        record.levelname = original_levelname
        record.msg = original_msg

        return result


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_data"):
            log_data["data"] = record.extra_data

        return json.dumps(log_data)


class PRECEPTLogger(logging.Logger):
    """Extended logger with structured data support."""

    def _log_with_data(
        self,
        level: int,
        msg: str,
        data: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ) -> None:
        """Log with optional structured data."""
        if data:
            # Create a custom record with extra data
            extra = kwargs.get("extra", {})
            extra["extra_data"] = data
            kwargs["extra"] = extra
        super()._log(level, msg, args, **kwargs)

    def info_with_data(self, msg: str, data: Dict[str, Any], *args, **kwargs) -> None:
        """Log info with structured data."""
        self._log_with_data(logging.INFO, msg, data, *args, **kwargs)

    def debug_with_data(self, msg: str, data: Dict[str, Any], *args, **kwargs) -> None:
        """Log debug with structured data."""
        self._log_with_data(logging.DEBUG, msg, data, *args, **kwargs)


# Set custom logger class
logging.setLoggerClass(PRECEPTLogger)

# Global log config
_log_config: Optional[LogConfig] = None
_initialized: bool = False


def get_log_config() -> LogConfig:
    """Get or create log configuration from environment."""
    global _log_config
    if _log_config is None:
        _log_config = LogConfig(
            level=os.environ.get("PRECEPT_LOG_LEVEL", "INFO"),
            json_format=os.environ.get("PRECEPT_LOG_JSON", "false").lower() == "true",
            colorize=os.environ.get("PRECEPT_LOG_COLOR", "true").lower() == "true",
            log_file=os.environ.get("PRECEPT_LOG_FILE"),
        )
    return _log_config


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    json_format: Optional[bool] = None,
    colorize: Optional[bool] = None,
    use_stderr: bool = False,
) -> None:
    """
    Setup logging for the PRECEPT framework.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for logging
        json_format: Use JSON format for logs (production)
        colorize: Use colored console output
        use_stderr: Use stderr instead of stdout (required for MCP servers)
    """
    global _initialized

    config = get_log_config()

    # Apply overrides
    if level is not None:
        config.level = level
    if log_file is not None:
        config.log_file = log_file
    if json_format is not None:
        config.json_format = json_format
    if colorize is not None:
        config.colorize = colorize

    # Get root logger for precept
    root_logger = logging.getLogger("precept")
    root_logger.setLevel(getattr(logging, config.level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler - use stderr for MCP servers to avoid JSONRPC conflicts
    stream = sys.stderr if use_stderr else sys.stdout
    console_handler = logging.StreamHandler(stream)
    console_handler.setLevel(getattr(logging, config.level.upper()))

    if config.json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            ColoredFormatter(config.format, config.date_format, config.colorize)
        )

    root_logger.addHandler(console_handler)

    # File handler (if specified)
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(config.log_file)
        file_handler.setLevel(getattr(logging, config.level.upper()))

        # Always use JSON format for file logs
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)

    # Set module-specific levels
    for module, mod_level in config.module_levels.items():
        logging.getLogger(module).setLevel(getattr(logging, mod_level.upper()))

    _initialized = True


def get_logger(name: str) -> PRECEPTLogger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        PRECEPTLogger instance
    """
    # Auto-initialize if not done
    if not _initialized:
        setup_logging()

    # Ensure the logger is under the precept namespace
    if not name.startswith("precept"):
        name = f"precept.{name}"

    return logging.getLogger(name)


# Convenience function for experiment scripts
def setup_experiment_logging(
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> PRECEPTLogger:
    """
    Setup logging for experiment scripts.

    Args:
        verbose: If True, use DEBUG level; otherwise INFO
        log_file: Optional log file path

    Returns:
        Logger for the experiment
    """
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level, log_file=log_file)
    return get_logger("precept.experiment")


# Pre-defined loggers for common modules
def get_agent_logger() -> PRECEPTLogger:
    """Get logger for agent module."""
    return get_logger("precept.agent")


def get_baseline_logger() -> PRECEPTLogger:
    """Get logger for baseline agents."""
    return get_logger("precept.baseline")


def get_mcp_logger() -> PRECEPTLogger:
    """Get logger for MCP server/client."""
    return get_logger("precept.mcp")


def get_experiment_logger() -> PRECEPTLogger:
    """Get logger for experiments."""
    return get_logger("precept.experiment")


def setup_mcp_server_logging(level: str = "INFO") -> PRECEPTLogger:
    """
    Setup logging for MCP server processes.

    MCP servers MUST NOT output to stdout as it interferes with JSONRPC.
    This function configures logging to use stderr only.

    Args:
        level: Log level (default: INFO)

    Returns:
        Logger for the MCP server
    """
    setup_logging(level=level, use_stderr=True, colorize=False)
    return get_logger("precept.mcp.server")


__all__ = [
    "LogConfig",
    "setup_logging",
    "get_logger",
    "setup_experiment_logging",
    "setup_mcp_server_logging",
    "get_agent_logger",
    "get_baseline_logger",
    "get_mcp_logger",
    "get_experiment_logger",
    "PRECEPTLogger",
]
