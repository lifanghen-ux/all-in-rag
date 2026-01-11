#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _load_analysis(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _format_top_topics(topics: List[Dict[str, Any]], top_n: int = 3) -> str:
    sorted_topics = sorted(topics, key=lambda t: t.get("prevalence", 0), reverse=True)
    lines = []
    for topic in sorted_topics[:top_n]:
        words = ", ".join(topic.get("top_words", [])[:5])
        lines.append(
            f"主题{topic.get('topic_id')} 占比 {topic.get('prevalence'):.4f}，关键词: {words}"
        )
    return "\n".join(lines)


def _answer_question(question: str, analysis: Dict[str, Any]) -> str:
    if not analysis:
        return "未找到分析结果，请先运行 run_agent_pipeline.py 生成 analysis_result.json。"

    computed = analysis.get("computed_facts", {})
    topics = analysis.get("topics_overview", [])
    agent = analysis.get("agent", {})

    query = question.lower()
    time_keywords = ["time", "trend", "year", "年份", "时间", "趋势"]
    category_keywords = ["category", "类别", "分类", "行业"]
    topic_keywords = ["topic", "主题", "话题"]

    if any(keyword in query for keyword in time_keywords):
        time_facts = computed.get("time", {})
        if not time_facts.get("enabled"):
            return "当前数据未包含 time 列，无法计算时序相关信息。"
        peak_year = time_facts.get("peak_year")
        doc_count_by_time = time_facts.get("doc_count_by_time", {})
        lines = [
            f"峰值年份: {peak_year}",
            "时间分布:",
            json.dumps(doc_count_by_time, ensure_ascii=False),
        ]
        return "\n".join(lines)

    if any(keyword in query for keyword in category_keywords):
        category_facts = computed.get("category", {})
        if not category_facts.get("enabled"):
            return "当前数据未包含 category 列，无法计算类别热力信息。"
        top_category_by_topic = category_facts.get("top_category_by_topic", {})
        lines = ["各主题最强类别:", json.dumps(top_category_by_topic, ensure_ascii=False)]
        return "\n".join(lines)

    if any(keyword in query for keyword in topic_keywords):
        return _format_top_topics(topics)

    summary = agent.get("default_summary", "暂无摘要。")
    suggestions = agent.get("suggested_questions", [])
    suggestion_text = "\n".join(f"- {item}" for item in suggestions)
    return f"{summary}\n\n可继续询问:\n{suggestion_text}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run agent chat")
    parser.add_argument("job_id", help="Job identifier")
    parser.add_argument("question", help="User question")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
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
