import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional


LOGGER = logging.getLogger(__name__)


def _data_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "result"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {path}") from exc


def load_analysis_result(job_id: str) -> Dict[str, Any]:
    path = _data_root() / job_id / "analysis_result.json"
    data = _load_json(path)
    LOGGER.info("Loaded analysis_result.json for job_id=%s", job_id)
    return data


def load_metrics(job_id: str) -> Optional[Dict[str, Any]]:
    path = _data_root() / job_id / "metrics.json"
    if not path.exists():
        LOGGER.info("metrics.json not found for job_id=%s", job_id)
        return None
    data = _load_json(path)
    LOGGER.info("Loaded metrics.json for job_id=%s", job_id)
    return data
