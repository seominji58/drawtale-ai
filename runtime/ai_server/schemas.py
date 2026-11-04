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

    # 모델이 이 관절을 얼마나 확신하는지 (0~1). 렌더 요청에는 없어도 된다
    score: float | None = None


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

    # 캐릭터 검출 점수 (0~1). 낮으면 백엔드·프론트가 어른 확인을 권한다
    confidence: float | None = None

    # 형태 (pt_docs 19, forms/*.json). 관절 이름이 형태마다 다르다
    form: str = "biped"
    # 관절을 실제로 찾았나. False 면 형태의 기본 뼈대를 그림 크기에 맞춰 놓은 것 — 화면이 관절 고치기로 보낸다
    joints_found: bool = True
    mask: MaskInfo | None = None
    message: str | None = None

    # Backend 계약 필수 메타데이터 (실패 시에는 없을 수 있다)
    model_version: str | None = None
    pipeline_version: str | None = None
    coordinate_space: str | None = None
    processing_time_ms: int | None = None

    # model_ 로 시작하는 필드명 경고 방지
    model_config = {"protected_namespaces": ()}