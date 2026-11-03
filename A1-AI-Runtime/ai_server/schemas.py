from pydantic import BaseModel, Field
from typing import List, Literal


CharacterType = Literal["human", "quadruped", "nonhuman"]


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


class RuntimeMetadata(BaseModel):
    model_version: str
    pipeline_version: str
    coordinate_space: str
    processing_time_ms: float


class AnalyzeResponse(BaseModel):
    success: bool

    character_type: CharacterType | None = None
    skeleton_type: str | None = None

    bbox: BoundingBox | None = None
    joints: List[Joint] = Field(default_factory=list)
    mask: MaskInfo | None = None
    metadata: RuntimeMetadata | None = None
    message: str | None = None


class RenderRequest(BaseModel):
    character_type: CharacterType = "human"
    skeleton_type: str | None = None
    joints: List[Joint] = Field(default_factory=list)