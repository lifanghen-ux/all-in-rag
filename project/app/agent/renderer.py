from typing import Any, Dict, List, Optional


def _format_percent(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def _join_keywords(keywords: List[str], limit: int = 8) -> str:
    if not keywords:
        return "-"
    return "、".join(keywords[:limit])


def _safe_value(value: Optional[Any]) -> str:
    if value is None:
        return "-"
    return str(value)


def _build_tldr(topics: List[Dict[str, Any]], metrics: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    if topics:
        top_topic = topics[0]
        lines.append(
            f"主题占比最高的是「{top_topic.get('name', '主题')}」（{_format_percent(top_topic.get('proportion'))}）。"
        )
    coherence = metrics.get("coherence")
    diversity = metrics.get("diversity")
    if coherence is not None or diversity is not None:
        lines.append(
            f"一致性 { _safe_value(coherence) }，多样性 { _safe_value(diversity) }。"
        )
    lines.append("报告与图表已生成，可直接下载并用于复盘。")
    return lines


def _extract_metrics(
    analysis_result: Dict[str, Any],
    metrics_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if metrics_payload:
        topic_quality = metrics_payload.get("topic_quality", {})
        model_perf = metrics_payload.get("model_performance", {})
        training_info = metrics_payload.get("training_info", {})
        metrics["coherence"] = topic_quality.get("coherence_npmi")
        metrics["diversity"] = topic_quality.get("diversity")
        metrics["perplexity"] = model_perf.get("perplexity")
        metrics["reconstruction_loss"] = model_perf.get("reconstruction_loss")
        metrics["epochs"] = training_info.get("total_epochs")
        metrics["training_time"] = training_info.get("training_time_seconds")
    else:
        summary = analysis_result.get("metrics", {})
        metrics["coherence"] = summary.get("coherence_score")
        metrics["diversity"] = summary.get("diversity_score")
        metrics["optimal_k"] = summary.get("optimal_k")
    return metrics


def _build_topic_table(topics: List[Dict[str, Any]]) -> List[str]:
    rows = ["| Topic | 名称 | 占比 | 关键词 |", "|------|------|------|--------|"]
    for topic in topics:
        rows.append(
            "| {id} | {name} | {ratio} | {keywords} |".format(
                id=_safe_value(topic.get("id")),
                name=_safe_value(topic.get("name")),
                ratio=_format_percent(topic.get("proportion")),
                keywords=_join_keywords(topic.get("keywords", [])),
            )
        )
    return rows


def _build_charts_section(charts: Dict[str, Any]) -> List[str]:
    sections: List[str] = []
    mapping = [
        ("topic_distribution", "主题分布", "主题分布"),
        ("heatmap", "文档-主题热力图", "热力图"),
        ("topic_similarity", "主题相似度", "主题相似度"),
        ("coherence_curve", "一致性曲线", "一致性曲线"),
    ]
    for key, title, alt in mapping:
        url = charts.get(key)
        if url:
            sections.append(f"### {title}")
            sections.append(f"![{alt}]({url})")
            sections.append("")
    if sections:
        sections.insert(0, "## 关键图表")
    return sections


def _build_quality_section(metrics: Dict[str, Any]) -> List[str]:
    lines = ["## 质量指标"]
    lines.append(f"- coherence: {_safe_value(metrics.get('coherence'))}")
    lines.append(f"- diversity: {_safe_value(metrics.get('diversity'))}")
    if metrics.get("perplexity") is not None:
        lines.append(f"- perplexity: {_safe_value(metrics.get('perplexity'))}")
    if metrics.get("reconstruction_loss") is not None:
        lines.append(f"- reconstruction_loss: {_safe_value(metrics.get('reconstruction_loss'))}")
    if metrics.get("epochs") is not None:
        lines.append(f"- epochs: {_safe_value(metrics.get('epochs'))}")
    if metrics.get("training_time") is not None:
        lines.append(f"- training_time: {_safe_value(metrics.get('training_time'))}s")
    if metrics.get("optimal_k") is not None:
        lines.append(f"- optimal_k: {_safe_value(metrics.get('optimal_k'))}")
    return lines


def _build_downloads_section(downloads: Dict[str, Any]) -> List[str]:
    lines = ["## 下载"]
    report = downloads.get("report")
    if report:
        lines.append(f"- [Word 报告]({report})")
    theta_csv = downloads.get("theta_csv")
    if theta_csv:
        lines.append(f"- [theta.csv]({theta_csv})")
    beta_csv = downloads.get("beta_csv")
    if beta_csv:
        lines.append(f"- [beta.csv]({beta_csv})")
    return lines


def render_markdown(
    analysis_result: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    view: str,
    topic_id: Optional[int],
    top_n: Optional[int],
) -> str:
    completed_at = analysis_result.get("completed_at", "-")
    duration = analysis_result.get("duration_seconds", "-")

    topics = analysis_result.get("topics", [])
    sorted_topics = sorted(topics, key=lambda item: item.get("proportion", 0), reverse=True)
    limit = top_n or 10
    overview_topics = sorted_topics[:limit]

    charts = analysis_result.get("charts", {})
    downloads = analysis_result.get("downloads", {})

    metric_summary = _extract_metrics(analysis_result, metrics)

    lines: List[str] = [f"# 主题分析解读（{analysis_result.get('job_id', '-')})"]
    lines.append("")
    lines.append(f"> 生成时间：{completed_at} ｜耗时：{duration}s")
    lines.append("")
    lines.append("## TL;DR（关键结论）")
    for line in _build_tldr(sorted_topics, metric_summary):
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## 主题概览")
    lines.extend(_build_topic_table(overview_topics))
    lines.append("")

    if view == "topic" and topic_id is not None:
        topic_detail = next((topic for topic in topics if topic.get("id") == topic_id), None)
        if topic_detail:
            lines.append(f"## Topic {topic_id} 深入解读")
            lines.append(f"**关键词：** {_join_keywords(topic_detail.get('keywords', []))}")
            lines.append(f"**占比：** {_format_percent(topic_detail.get('proportion'))}")
            wordcloud_url = topic_detail.get("wordcloud_url")
            if wordcloud_url:
                lines.append(f"![Topic {topic_id} 词云]({wordcloud_url})")
            lines.append("")

    chart_section = _build_charts_section(charts)
    if chart_section:
        lines.extend(chart_section)

    lines.append("## 质量指标")
    lines.extend(_build_quality_section(metric_summary)[1:])
    lines.append("")

    lines.extend(_build_downloads_section(downloads))

    return "\n".join(lines).strip() + "\n"
