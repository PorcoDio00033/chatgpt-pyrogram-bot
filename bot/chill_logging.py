import logging
import sys
import re
import traceback
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from datetime import datetime


# Shared console handler
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
_console_handler.setLevel(logging.INFO)

# Cache for created loggers
_loggers = {}


# to not import the whole logging module when changing log level
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

LOGS_FILE_PATH = "data/logs"
MAX_CHR_LOGGED_RES = 2000

class Logger:
    """
    Unified logger class for module-specific and global logs,
    with weekly rotation and optional Telegram handler.
    """

    def __init__(self, name: str, in_subfolder: bool = True,
                 level: int = INFO, rotating_when: str = "W0",
                 rotation_interval: int = 1, backup_count: int = 4,
                 utc: bool = True, propagate: bool = False, log_file_name: str | None = None):
        """
        :param name: Logger name (and filename).
        :param in_subfolder: Store in logs/{name}/{name}.log if True, else logs/{name}.log in root.
        :param level: Logging level.
        :param rotating_when: Rotation schedule ('W0' = Monday, 'midnight', 'D', etc.).
        :param rotation_interval: Interval between rotations.
        :param backup_count: Number of rotated logs to keep.
        :param utc: Whether to use UTC for rotation.
        """
        if name in _loggers:
            existing = _loggers[name]
            self.logger = existing.logger
            self.log_file_path = existing.log_file_path
            return

        file_name = log_file_name or sanitize_name(name.replace(".", "_")) or "unnamed"
        # Log directory & file path
        if in_subfolder:
            log_dir = Path(LOGS_FILE_PATH) / file_name
            log_file = log_dir / f"{file_name}.log"
        else:
            log_dir = Path(LOGS_FILE_PATH)
            log_file = log_dir / f"{file_name}.log"

        log_dir.mkdir(parents=True, exist_ok=True)

        # Rotating file handler
        file_handler = TimedRotatingFileHandler(
            log_file,
            when=rotating_when,
            interval=rotation_interval,
            backupCount=backup_count,
            encoding="utf-8",
            utc=utc
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        file_handler.setLevel(level)

        # Logger instance
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = propagate
        self.logger.addHandler(file_handler)
        self.logger.addHandler(_console_handler)
        
        self.log_file_path = log_file
        _loggers[name] = self


# ------------------------------
# Helper Functions
# ------------------------------

def configure_third_party_loggers():
    """Silence noisy third-party libraries."""
    logging.getLogger("pyrogram").setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    # logging.getLogger('openai._base_client').setLevel(logging.DEBUG)

def get_logger_instance(name: str, **kargs) -> Logger:
    """
    Get a cached logger instance or create a new one.
    """
    return _loggers.get(name) or Logger(name, **kargs)

def list_logger_objects():
    return list(_loggers.values())

def list_logger_names() -> list[str]:
    """Return all logger names currently in cache."""
    return sorted(_loggers.keys())

def list_logger_paths() -> list[str]:
    """Return all logger file paths currently in cache."""
    return sorted(str(logger.log_file_path) for logger in _loggers.values())

def log_exception(logger: logging.Logger, context: str, error: Exception):
    """
    Log an exception for the given service.
    Skips traceback for classes ending with 'AuthError'.
    """
    logger.error(f"[{context}] {type(error).__name__}: {error}")

    if not type(error).__name__.endswith("AuthError"):
        logger.error(traceback.format_exc())


def handle_long_response_log(sub_status: str, full_response: str, service_name: str, max_length: int = MAX_CHR_LOGGED_RES, save_to_file: bool = True) -> str:
    """
    Truncate long responses, save full content to file if too long.
    """
    logger = get_logger_instance(service_name).logger
    short_response = truncate(full_response, max_length=max_length)

    if save_to_file and full_response and len(full_response) > max_length:
        file_path = save_long_response(sub_status, full_response, service_name)
        if file_path:
            logger.info(f"{service_name}[{sub_status}] Full error saved to: {file_path}")

    return short_response


def save_long_response(sub_status: str, content: str, service_name: str) -> str:
    """
    Save long HTTP response to timestamped text file under logs/<service_name>/.
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"{sub_status}_{service_name}_{timestamp}.txt"

    service_dir = Path(LOGS_FILE_PATH) / service_name
    service_dir.mkdir(parents=True, exist_ok=True)

    file_path = service_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return str(file_path)


def sanitize_name(name: str) -> str:
    if not name:
        return ''
    name = str(name).strip()
    name = re.sub(r'[\x00-\x1F\x7F]', '', name)
    name = re.sub(r'[\\/*?"<>|$]', '', name)
    name = re.sub(r'[:]', ' - ', name)
    return name


def truncate(text: str, max_length: int = MAX_CHR_LOGGED_RES) -> str:
    """
    Truncate `text` to `max_length` characters, appending '...' if truncated.
    """
    return text if len(text) <= max_length else text[:max_length] + "..."
