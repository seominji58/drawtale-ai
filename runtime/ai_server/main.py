import json
import logging
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ai_server.analyzer import (
    MODEL_VERSION,
    PIPELINE_VERSION,
    analyze_image,
    check_torchserve,
)
from ai_server.renderer import MOTIONS, RenderError, render_animation
from ai_server.schemas import AnalyzeResponse, HealthResponse

log = logging.getLogger(__name__)

app = FastAPI(title="DrawTale AI Runtime", version=PIPELINE_VERSION)

# 분석 결과(마스크)를 렌더링까지 넘기기 위한 임시 저장소.
# 컨테이너 재시작 시 사라지므로 영구 보관용이 아니다.
WORK_DIR = Path(os.getenv("AI_WORK_DIR", "/tmp/drawtale"))


@app.get("/internal/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=check_torchserve(),
        runtime="animated-drawings",
        model_version=MODEL_VERSION,
        pipeline_version=PIPELINE_VERSION,
    )


# 분석·렌더링은 CPU를 오래 쓰므로 def로 둔다.
# async def로 두면 처리하는 동안 /health 등 다른 요청까지 멈춘다.
@app.post("/internal/v1/analyze", response_model=AnalyzeResponse)
def analyze(file: UploadFile = File(...)):
    """그림을 분석해 bbox, 관절 15개를 반환한다.

    렌더링에 필요한 원본 그림과 마스크는 request_id 폴더에 저장해 둔다.
    """

    request_id = str(uuid.uuid4())
    session_dir = WORK_DIR / request_id
    temp_path = None

    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".png"
        session_dir.mkdir(parents=True, exist_ok=True)

        temp_path = str(session_dir / f"source{suffix}")
        with open(temp_path, "wb") as f:
            f.write(file.file.read())

        result = analyze_image(temp_path, mask_path=str(session_dir / "mask.png"))

        if not result.get("success"):
            shutil.rmtree(session_dir, ignore_errors=True)
            return AnalyzeResponse(**result)

        # 렌더링에서 재사용할 정보를 남긴다
        with open(session_dir / "analysis.json", "w") as f:
            json.dump(
                {"bbox": result["bbox"], "source": os.path.basename(temp_path)}, f
            )

        return AnalyzeResponse(request_id=request_id, **result)

    except FileNotFoundError as e:
        shutil.rmtree(session_dir, ignore_errors=True)
        return AnalyzeResponse(success=False, message=str(e))

    except Exception as e:
        log.exception("analyze failed")
        shutil.rmtree(session_dir, ignore_errors=True)
        return AnalyzeResponse(success=False, message=str(e))


@app.post("/internal/v1/render")
def render(
    request_id: str = Form(..., description="analyze 응답의 request_id"),
    joints: str = Form(..., description='관절 15개 JSON 배열 [{"name","x","y"}, ...]'),
    motion: str = Form("wave_hello", description=f"사용 가능: {', '.join(MOTIONS)}"),
    max_frames: int | None = Form(
        None, ge=1, description="프레임 상한. 동작 전체에서 일정 간격으로 고른다 (S-05 미리보기용)"
    ),
):
    """Backend가 보낸 관절(사용자 보정 반영)로 애니메이션 MP4를 만든다.

    analyze가 남긴 원본 그림과 마스크를 재사용하므로 request_id가 필요하다.
    성공하면 MP4 파일을 그대로 반환한다.
    """

    session_dir = WORK_DIR / request_id

    if not session_dir.is_dir():
        return _error("SESSION_NOT_FOUND", f"request_id not found: {request_id}")

    try:
        joint_list = json.loads(joints)
    except json.JSONDecodeError as e:
        return _error("INVALID_JOINTS", f"joints must be a JSON array: {e}")

    try:
        with open(session_dir / "analysis.json") as f:
            analysis = json.load(f)

        # 미리보기와 이야기 렌더가 같은 세션에서 겹칠 수 있어 요청마다 다른 파일에 쓴다
        output_path = session_dir / f"animation-{uuid.uuid4().hex}.mp4"
        result = render_animation(
            image_path=str(session_dir / analysis["source"]),
            mask_path=str(session_dir / "mask.png"),
            joints=joint_list,
            bbox=analysis["bbox"],
            motion=motion,
            output_path=str(output_path),
            max_frames=max_frames,
        )

        log.info(
            "render %s: %s, %s bytes, %s ms",
            request_id,
            result["motion"],
            result["size_bytes"],
            result["processing_time_ms"],
        )

        return FileResponse(
            output_path,
            # 보낸 뒤 지운다 — 렌더할 때마다 파일이 쌓이지 않게
            background=BackgroundTask(output_path.unlink, missing_ok=True),
            media_type="video/mp4",
            filename="animation.mp4",
            headers={
                "X-Processing-Time-Ms": str(result["processing_time_ms"]),
                "X-Model-Version": MODEL_VERSION,
                "X-Pipeline-Version": PIPELINE_VERSION,
            },
        )

    except RenderError as e:
        code, _, detail = str(e).partition(": ")
        return _error(code, detail or str(e))

    except Exception as e:
        log.exception("render failed")
        return _error("RENDER_FAILED", str(e))


@app.delete("/internal/v1/sessions/{request_id}")
def delete_session(request_id: str):
    """Backend가 결과를 받아간 뒤 임시 파일을 정리한다."""

    shutil.rmtree(WORK_DIR / request_id, ignore_errors=True)
    return {"success": True}


def _error(code: str, message: str) -> dict:
    """analyze와 같은 형식으로 오류를 돌려준다. Backend가 code를 읽는다."""

    return {"success": False, "message": f"{code}: {message}"}
