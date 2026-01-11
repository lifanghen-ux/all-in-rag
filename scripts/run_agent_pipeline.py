#!/usr/bin/env python3
import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT_FALLBACK = Path("/root/autodl-tmp")


def _resolve_root() -> Path:
    if ROOT_FALLBACK.exists():
        return ROOT_FALLBACK
    return Path(__file__).resolve().parents[1]


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"agent_pipeline_{log_path}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _load_topic_words(
    topic_path: Path, num_topics: int, logger: logging.Logger
) -> List[List[str]]:
    if not topic_path.exists():
        logger.error("Topic words file not found: %s", topic_path)
        raise FileNotFoundError(f"Missing topic words file: {topic_path}")

    with topic_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    topic_words: List[List[str]] = []
    if isinstance(raw, list):
        topic_words = [_normalize_topic_words(item) for item in raw]
    elif isinstance(raw, dict):

        def _key_sort(key: str) -> Tuple[int, str]:
            try:
                return int(key), key
            except ValueError:
                return 10**9, key

        for key in sorted(raw.keys(), key=_key_sort):
            topic_words.append(_normalize_topic_words(raw[key]))
    else:
        logger.warning("Unexpected topic words structure: %s", type(raw))

    if len(topic_words) < num_topics:
        logger.warning(
            "Topic words count (%s) smaller than num_topics (%s)",
            len(topic_words),
            num_topics,
        )
        topic_words.extend([[] for _ in range(num_topics - len(topic_words))])
    return topic_words[:num_topics]


def _normalize_topic_words(item: Any) -> List[str]:
    if isinstance(item, list):
        return [str(word) for word in item]
    if isinstance(item, dict):
        words = item.get("words") or item.get("top_words") or item.get("tokens") or []
        if isinstance(words, list):
            return [str(word) for word in words]
    return [str(item)]


def _load_theta(theta_path: Path, logger: logging.Logger) -> np.ndarray:
    if not theta_path.exists():
        logger.error("Theta file not found: %s", theta_path)
        raise FileNotFoundError(f"Missing theta file: {theta_path}")
    theta = np.load(theta_path)
    logger.info("Loaded theta with shape %s", theta.shape)
    return theta


def _build_topic_name(topic_id: int, keywords: List[str]) -> str:
    if keywords:
        short = " ".join(keywords[:2])
        if short.strip():
            return f"Topic {topic_id}: {short}"
    return f"Topic_{topic_id}"


def _load_metrics(metrics_path: Path, logger: logging.Logger) -> Tuple[Any, Any]:
    if not metrics_path.exists():
        logger.warning("Metrics file not found: %s", metrics_path)
        return None, None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse metrics.json: %s", exc)
        return None, None

    topic_quality = metrics.get("topic_quality", {})
    return topic_quality.get("coherence_npmi"), topic_quality.get("diversity")


def _export_theta_csv(theta: np.ndarray, output_path: Path, logger: logging.Logger) -> None:
    if output_path.exists():
        logger.info("theta.csv already exists: %s", output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [f"topic_{idx}" for idx in range(theta.shape[1])]
    df = pd.DataFrame(theta, columns=columns)
    df.to_csv(output_path, index=False)
    logger.info("Exported theta.csv to %s", output_path)


def _export_beta_csv(beta_path: Path, output_path: Path, logger: logging.Logger) -> None:
    if not beta_path.exists():
        logger.warning("Beta file not found: %s", beta_path)
        return
    if output_path.exists():
        logger.info("beta.csv already exists: %s", output_path)
        return
    beta = np.load(beta_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(beta)
    df.to_csv(output_path, index=False)
    logger.info("Exported beta.csv to %s", output_path)


def _format_completed_at() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _trim_keywords(keywords: List[str], topic_id: int, logger: logging.Logger) -> List[str]:
    trimmed = keywords[:5]
    if not trimmed:
        logger.warning("Topic %s has no keywords", topic_id)
    return trimmed


def _check_wordcloud(job_id: str, topic_id: int, root: Path, logger: logging.Logger) -> None:
    path = (
        root
        / "visualization"
        / "outputs"
        / job_id
        / f"wordcloud_topic_{topic_id}.png"
    )
    if not path.exists():
        logger.warning("Wordcloud image missing: %s", path)


def main() -> None:
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Run agent pipeline analysis")
    parser.add_argument("job_id", help="Job identifier")
    args = parser.parse_args()

    root = _resolve_root()
    job_id = str(args.job_id)

    theta_path = root / "ETM" / "outputs" / "theta" / f"{job_id}_theta.npy"
    topic_path = root / "ETM" / "outputs" / "topic_words" / f"{job_id}_topics.json"
    result_dir = root / "result" / job_id
    analysis_path = result_dir / "analysis_result.json"
    log_path = result_dir / "log.txt"
    metrics_path = result_dir / "metrics.json"
    theta_csv_path = result_dir / "theta.csv"
    beta_csv_path = result_dir / "beta.csv"
    beta_path = root / "ETM" / "outputs" / "beta" / f"{job_id}_beta.npy"

    logger = _setup_logger(log_path)
    logger.info("Starting pipeline for job_id=%s", job_id)
    logger.info("Using root path: %s", root)

    status = "success"
    metrics: Dict[str, Any] = {
        "coherence_score": None,
        "diversity_score": None,
        "optimal_k": None,
    }
    topics: List[Dict[str, Any]] = []

    try:
        theta = _load_theta(theta_path, logger)
        num_docs, num_topics = theta.shape
        topic_words = _load_topic_words(topic_path, num_topics, logger)

        topic_prevalence = theta.mean(axis=0)

        for topic_id in range(num_topics):
            keywords = _trim_keywords(topic_words[topic_id], topic_id, logger)
            _check_wordcloud(job_id, topic_id, root, logger)
            topics.append(
                {
                    "id": int(topic_id),
                    "name": _build_topic_name(int(topic_id), keywords),
                    "keywords": keywords,
                    "proportion": float(topic_prevalence[topic_id]),
                    "wordcloud_url": f"/api/download/{job_id}/wordcloud_topic_{topic_id}.png",
                }
            )

        _export_theta_csv(theta, theta_csv_path, logger)
        _export_beta_csv(beta_path, beta_csv_path, logger)
        coherence_score, diversity_score = _load_metrics(metrics_path, logger)
        metrics = {
            "coherence_score": coherence_score,
            "diversity_score": diversity_score,
            "optimal_k": int(num_topics),
        }
    except Exception as exc:  # pragma: no cover - defensive
        status = "failed"
        logger.exception("Pipeline failed: %s", exc)

    duration_seconds = int(time.time() - start_time)
    analysis_result = {
        "job_id": job_id,
        "status": status,
        "completed_at": _format_completed_at(),
        "duration_seconds": duration_seconds,
        "metrics": metrics,
        "topics": topics,
        "charts": {
            "topic_distribution": f"/api/download/{job_id}/topic_distribution.png",
            "heatmap": f"/api/download/{job_id}/heatmap_doc_topic.png",
            "coherence_curve": f"/api/download/{job_id}/coherence_curve.png",
            "topic_similarity": f"/api/download/{job_id}/topic_similarity.png",
        },
        "downloads": {
            "report": f"/api/download/{job_id}/report.docx",
            "theta_csv": f"/api/download/{job_id}/theta.csv",
            "beta_csv": f"/api/download/{job_id}/beta.csv",
        },
    }

    result_dir.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(
        json.dumps(analysis_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved analysis_result.json to %s", analysis_path)
    logger.info("Completed in %s seconds", duration_seconds)


if __name__ == "__main__":
    main()
