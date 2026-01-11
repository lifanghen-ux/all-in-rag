import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


LOGGER = logging.getLogger(__name__)


def _logs_root() -> Path:
    return Path(__file__).resolve().parents[2] / "logs"


def log_render_event(request_payload: Dict[str, Any], markdown: str) -> None:
    _logs_root().mkdir(parents=True, exist_ok=True)
    job_id = request_payload.get("job_id", "unknown")
    log_path = _logs_root() / f"{job_id}.jsonl"
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request": request_payload,
        "markdown": markdown,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    LOGGER.info("Logged render event for job_id=%s", job_id)
