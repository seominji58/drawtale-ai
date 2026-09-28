from pydantic import BaseModel, Field
from typing import List


class BoundingBox(BaseModel):
    left: int
    top: int
    right: int
    bottom: int


class Joint(BaseModel):
    name: str
    x: int
    y: int


class MaskInfo(BaseModel):
    width: int
    height: int


class AnalyzeResponse(BaseModel):
    success: bool
    bbox: BoundingBox | None = None
    joints: List[Joint] = Field(default_factory=list)
    mask: MaskInfo | None = None
    message: str | None = None