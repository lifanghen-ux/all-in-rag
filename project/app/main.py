from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from app.agent.loaders import load_analysis_result, load_metrics
from app.agent.renderer import render_markdown
from app.agent.schemas import RenderMarkdownRequest, RenderMarkdownResponse
from app.agent.storage import log_render_event

app = FastAPI(title="ETM Topic Analysis Agent")


@app.get("/agent/health", response_class=PlainTextResponse)
def health_check() -> str:
    return "OK"


@app.post("/agent/render_markdown", response_model=RenderMarkdownResponse)
def render_markdown_endpoint(payload: RenderMarkdownRequest) -> RenderMarkdownResponse:
    try:
        analysis_result = load_analysis_result(payload.job_id)
        metrics = load_metrics(payload.job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    markdown = render_markdown(
        analysis_result=analysis_result,
        metrics=metrics,
        view=payload.view,
        topic_id=payload.topic_id,
        top_n=payload.top_n,
    )

    log_render_event(payload.dict(), markdown)

    return RenderMarkdownResponse(job_id=payload.job_id, markdown=markdown)
