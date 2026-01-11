#!/usr/bin/env python3
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("agent_pipeline")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _load_topic_words(topic_path: Path, num_topics: int, logger: logging.Logger) -> List[List[str]]:
    if not topic_path.exists():
        logger.error("Topic words file not found: %s", topic_path)
        raise FileNotFoundError(f"Missing topic words file: {topic_path}")

    with topic_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    topic_words: List[List[str]] = []
    if isinstance(raw, list):
        topic_words = [
            _normalize_topic_words(item) for item in raw
        ]
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


def _prepare_time_facts(
    df: pd.DataFrame,
    theta: np.ndarray,
    dominant_topics: np.ndarray,
    logger: logging.Logger,
) -> Dict[str, Any]:
    time_series = df["time"]
    parsed = pd.to_datetime(time_series, errors="coerce")
    if parsed.notna().any():
        time_values = parsed.dt.year.astype("Int64").astype(str)
    else:
        time_values = time_series.astype(str)

    n_docs = min(len(time_values), theta.shape[0])
    if len(time_values) != theta.shape[0]:
        logger.warning(
            "Data rows (%s) and theta rows (%s) mismatch; using first %s rows",
            len(time_values),
            theta.shape[0],
            n_docs,
        )
    time_values = time_values.iloc[:n_docs]
    theta = theta[:n_docs]
    dominant_topics = dominant_topics[:n_docs]

    doc_counts = time_values.value_counts(dropna=False).sort_index()
    doc_count_by_time = {str(idx): int(count) for idx, count in doc_counts.items()}

    topic_strength_by_time: Dict[str, List[float]] = {}
    for time_value, group_indices in time_values.groupby(time_values).groups.items():
        group_theta = theta[list(group_indices)]
        topic_strength_by_time[str(time_value)] = [
            float(value) for value in group_theta.mean(axis=0)
        ]

    peak_year = None
    if doc_count_by_time:
        peak_year = max(doc_count_by_time.items(), key=lambda item: item[1])[0]

    return {
        "enabled": True,
        "doc_count_by_time": doc_count_by_time,
        "topic_strength_by_time": topic_strength_by_time,
        "peak_year": peak_year,
    }


def _prepare_category_facts(
    df: pd.DataFrame,
    dominant_topics: np.ndarray,
    num_topics: int,
    logger: logging.Logger,
) -> Dict[str, Any]:
    categories = df["category"].fillna("Unknown").astype(str)
    n_docs = min(len(categories), len(dominant_topics))
    if len(categories) != len(dominant_topics):
        logger.warning(
            "Category rows (%s) and theta rows (%s) mismatch; using first %s rows",
            len(categories),
            len(dominant_topics),
            n_docs,
        )
    categories = categories.iloc[:n_docs]
    dominant_topics = dominant_topics[:n_docs]

    counts = pd.crosstab(categories, dominant_topics)
    for topic_id in range(num_topics):
        if topic_id not in counts.columns:
            counts[topic_id] = 0
    counts = counts.sort_index(axis=1)
    categories_sorted = counts.index.tolist()

    matrix = counts.values.tolist()
    top_category_by_topic: Dict[str, str] = {}
    for topic_id in range(num_topics):
        column = counts[topic_id]
        if column.sum() == 0:
            top_category_by_topic[str(topic_id)] = "Unknown"
        else:
            top_category_by_topic[str(topic_id)] = column.idxmax()

    return {
        "enabled": True,
        "category_heatmap": {
            "categories": categories_sorted,
            "topics": [int(topic) for topic in counts.columns.tolist()],
            "matrix": matrix,
        },
        "top_category_by_topic": top_category_by_topic,
    }


def _build_summary(num_docs: int, topics_overview: List[Dict[str, Any]]) -> str:
    top_topics = sorted(topics_overview, key=lambda t: t["prevalence"], reverse=True)[:3]
    topic_lines = ", ".join(
        f"主题{item['topic_id']}" for item in top_topics
    )
    return f"分析了 {num_docs} 篇文档，共 {len(topics_overview)} 个主题，主要关注 {topic_lines}。"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent pipeline analysis")
    parser.add_argument("job_id", help="Job identifier")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    job_id = str(args.job_id)

    theta_path = root / "ETM" / "outputs" / "theta" / f"{job_id}_theta.npy"
    topic_path = root / "ETM" / "outputs" / "topic_words" / f"{job_id}_topics.json"
    data_path = root / "data" / job_id / "data.csv"
    result_dir = root / "result" / job_id
    analysis_path = result_dir / "analysis_result.json"
    log_path = result_dir / "log.txt"

    logger = _setup_logger(log_path)
    logger.info("Starting pipeline for job_id=%s", job_id)

    theta = _load_theta(theta_path, logger)
    num_docs, num_topics = theta.shape
    topic_words = _load_topic_words(topic_path, num_topics, logger)

    topic_prevalence = theta.mean(axis=0)
    dominant_topics = theta.argmax(axis=1)

    topics_overview = []
    for topic_id in range(num_topics):
        topics_overview.append(
            {
                "topic_id": int(topic_id),
                "top_words": topic_words[topic_id],
                "prevalence": float(topic_prevalence[topic_id]),
            }
        )

    data_summary = {
        "doc_count": int(num_docs),
        "topic_count": int(num_topics),
        "dominant_topic_distribution": {
            str(topic_id): int(count)
            for topic_id, count in enumerate(
                np.bincount(dominant_topics, minlength=num_topics)
            )
        },
        "data_path": str(data_path.relative_to(root)),
    }

    computed_facts: Dict[str, Any] = {
        "time": {"enabled": False},
        "category": {"enabled": False},
    }

    if data_path.exists():
        try:
            df = pd.read_csv(data_path)
            if "time" in df.columns:
                computed_facts["time"] = _prepare_time_facts(
                    df, theta, dominant_topics, logger
                )
            if "category" in df.columns:
                computed_facts["category"] = _prepare_category_facts(
                    df, dominant_topics, num_topics, logger
                )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to process data.csv: %s", exc)
    else:
        logger.info("Optional data file not found: %s", data_path)

    analysis_result = {
        "job_id": job_id,
        "model_type": "ETM",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_summary": data_summary,
        "topics_overview": topics_overview,
        "figures": [],
        "computed_facts": computed_facts,
        "agent": {
            "default_summary": _build_summary(num_docs, topics_overview),
            "suggested_questions": [
                "有哪些主题占比最高？",
                "主题随时间的变化趋势是什么？",
                "不同类别与主题的关系如何？",
            ],
        },
    }

    result_dir.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(
        json.dumps(analysis_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Saved analysis_result.json to %s", analysis_path)


if __name__ == "__main__":
    main()
