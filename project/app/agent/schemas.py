from typing import Literal, Optional

from pydantic import BaseModel, Field


class TimeRange(BaseModel):
    start: str
    end: str


class CategoryFilter(BaseModel):
    field: str
    value: str


class RenderMarkdownRequest(BaseModel):
    job_id: str
    view: Literal["overview", "topic", "quality"]
    topic_id: Optional[int] = None
    top_n: Optional[int] = Field(default=10, ge=1)
    time_range: Optional[TimeRange] = None
    category_filter: Optional[CategoryFilter] = None


class RenderMarkdownResponse(BaseModel):
    job_id: str
    markdown: str
