#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


ROOT_FALLBACK = Path("/root/autodl-tmp")


def _resolve_root() -> Path:
    if ROOT_FALLBACK.exists():
        return ROOT_FALLBACK
    return Path(__file__).resolve().parents[1]


def _load_analysis(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _format_topic_line(topic: Dict[str, Any]) -> str:
    words = ", ".join(topic.get("keywords", [])[:5])
    proportion = topic.get("proportion", 0.0)
    return f"{topic.get('name')} (占比 {proportion:.4f}) 关键词: {words}"


def _format_top_topics(topics: List[Dict[str, Any]], top_n: int = 3) -> str:
    if not topics:
        return "暂无可用主题信息。"
    sorted_topics = sorted(topics, key=lambda t: t.get("proportion", 0), reverse=True)
    lines = [_format_topic_line(topic) for topic in sorted_topics[:top_n]]
    return "\n".join(lines)


def _format_all_topics(topics: List[Dict[str, Any]]) -> str:
    if not topics:
        return "暂无可用主题信息。"
    lines = [_format_topic_line(topic) for topic in topics]
    return "\n".join(lines)


def _format_downloads(charts: Dict[str, Any], downloads: Dict[str, Any]) -> str:
    chart_lines = [
        f"topic_distribution: {charts.get('topic_distribution', '')}",
        f"heatmap: {charts.get('heatmap', '')}",
        f"coherence_curve: {charts.get('coherence_curve', '')}",
        f"topic_similarity: {charts.get('topic_similarity', '')}",
    ]
    download_lines = [
        f"report: {downloads.get('report', '')}",
        f"theta_csv: {downloads.get('theta_csv', '')}",
        f"beta_csv: {downloads.get('beta_csv', '')}",
    ]
    return "\n".join(["图表:"] + chart_lines + ["下载:"] + download_lines)


def _answer_question(question: str, analysis: Dict[str, Any]) -> str:
    if not analysis:
        return "未找到分析结果，请先运行 run_agent_pipeline.py 生成 analysis_result.json。"

    topics = analysis.get("topics", [])
    charts = analysis.get("charts", {})
    downloads = analysis.get("downloads", {})
    query = question.lower()

    top_keywords = ["占比最高", "top", "最高"]
    list_keywords = ["主题列表", "主题概览", "主题概况", "topic list"]
    download_keywords = ["下载链接", "图表", "charts", "downloads", "链接"]

    if any(keyword in query for keyword in top_keywords):
        return _format_top_topics(topics)

    if any(keyword in query for keyword in list_keywords):
        return _format_all_topics(topics)

    if any(keyword in query for keyword in download_keywords):
        return _format_downloads(charts, downloads)

    return "可以询问：有哪些主题占比最高？ / 给我主题列表 / 下载链接有哪些？"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent chat")
    parser.add_argument("job_id", help="Job identifier")
    parser.add_argument("question", help="User question")
    args = parser.parse_args()

    root = _resolve_root()
    job_id = str(args.job_id)

    result_dir = root / "result" / job_id
    analysis_path = result_dir / "analysis_result.json"
    chat_history_path = result_dir / "chat_history.jsonl"

    analysis = _load_analysis(analysis_path)
    answer = _answer_question(args.question, analysis)

    chat_entry = {
        "job_id": job_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": args.question,
        "answer": answer,
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    with chat_history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(chat_entry, ensure_ascii=False) + "\n")

    print(answer)


if __name__ == "__main__":
    main()
