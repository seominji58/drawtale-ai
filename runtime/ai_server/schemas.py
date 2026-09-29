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
    saved: bool = False


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    runtime: str
    model_version: str
    pipeline_version: str


class AnalyzeResponse(BaseModel):
    success: bool

    # render 호출 시 이 값을 넘기면 저장된 원본/마스크를 재사용한다
    request_id: str | None = None
    bbox: BoundingBox | None = None
    joints: List[Joint] = Field(default_factory=list)
    mask: MaskInfo | None = None
    message: str | None = None

    # Backend 계약 필수 메타데이터 (실패 시에는 없을 수 있다)
    model_version: str | None = None
    pipeline_version: str | None = None
    coordinate_space: str | None = None
    processing_time_ms: int | None = None

    # model_ 로 시작하는 필드명 경고 방지
    model_config = {"protected_namespaces": ()}