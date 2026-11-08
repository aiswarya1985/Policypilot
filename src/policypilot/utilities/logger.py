"""Central loguru setup. Call setup_logger() ONCE at each entry point
(pipeline __main__, streamlit_app, scripts). Other modules just do:
    from loguru import logger
"""

import sys

from loguru import logger

from policypilot.config import REPO_ROOT

LOG_DIR = REPO_ROOT / "logs"
LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <8} | "
    "{name}:{function}:{line} - {message}"
)


def setup_logger(level: str = "INFO") -> None:
    LOG_DIR.mkdir(exist_ok=True)
    logger.remove()  # clear default + any earlier handlers (safe on Streamlit reruns)

    # Console: level is configurable (use "DEBUG" while learning)
    logger.add(sys.stderr, format=LOG_FORMAT, level=level)

    # File: always DEBUG, so the full detail is saved even if the console is quieter
    logger.add(
        LOG_DIR / "logs_policypilot.txt",
        format=LOG_FORMAT,
        level="DEBUG",
        colorize=False,
        rotation="10 MB",
        retention="30 days",
        encoding="utf-8",
    )
