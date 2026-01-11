# ETM 主题分析平台 Agent 模块（本地文件模式）

本项目提供一个最小可运行的 Agent 服务，读取本地 JSON 结果并渲染为 Markdown 字符串，供前端直接渲染。

## 目录结构

```
project/
  app/
    main.py
    agent/
      loaders.py
      renderer.py
      schemas.py
      storage.py
  data/
    result/{job_id}/analysis_result.json
    result/{job_id}/metrics.json
  logs/
  requirements.txt
  README.md
```

## 本地数据准备

将作业结果放到如下路径（示例已提供）：

```
project/data/result/job_demo_001/analysis_result.json
project/data/result/job_demo_001/metrics.json
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 健康检查

```bash
curl http://localhost:8000/agent/health
```

## 渲染 Markdown

```bash
curl -X POST http://localhost:8000/agent/render_markdown \
  -H 'Content-Type: application/json' \
  -d '{
    "job_id": "job_demo_001",
    "view": "overview",
    "top_n": 10
  }'
```

返回示例：

```json
{
  "job_id": "job_demo_001",
  "markdown": "# 主题分析解读（job_demo_001）\n\n> 生成时间：2025-01-06 10:30:00 ｜耗时：42.5s\n..."
}
```

## 日志记录

每次调用 `/agent/render_markdown` 会在 `project/logs/{job_id}.jsonl` 追加一条记录，包含请求、生成结果与时间戳，便于后续接入对象存储或数据库。
