import os
from rich.logging import RichHandler
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("httpx").setLevel(logging.WARN)

TARGET_HF_REPO = os.getenv("TARGET_HF_REPO", "Moenupa/verl")
DRY_RUN = os.getenv("DRY_RUN", "1").lower() in ("true", "1", "yes")
