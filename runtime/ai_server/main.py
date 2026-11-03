import json
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

from ai_server.analyzer import (
    MODEL_VERSION,
    PIPELINE_VERSION,
    analyze_image,
    make_cutout,
    check_torchserve,
)
from ai_server.renderer import MOTIONS, RenderError, render_animation
from ai_server.schemas import AnalyzeResponse, HealthResponse

log = logging.getLogger(__name__)

app = FastAPI(title="DrawTale AI Runtime", version=PIPELINE_VERSION)

# 분석 결과(마스크)를 렌더링까지 넘기기 위한 임시 저장소.
# 컨테이너 재시작 시 사라지므로 영구 보관용이 아니다.
WORK_DIR = Path(os.getenv("AI_WORK_DIR", "/tmp/drawtale"))

# 아이 그림 사본이 들어 있으므로 늦어도 하루 뒤 지운다 (02 시스템 아키텍처 7-3)
SESSION_MAX_AGE_SECONDS = 24 * 60 * 60

# 렌더는 한 번에 하나만 (2026-11-03). 한 번에 메모리를 1GB 가까이 더 쓰는데 컨테이너 한도가 있어
# 둘이 겹치면 서버가 죽는다. Meta 렌더는 작업 폴더(os.chdir)를 바꿔 가며 써서 겹치면 서로 꼬이기도 한다.
# 렌더 하나가 1~8초라 뒤 요청은 그만큼 기다린다 (Backend 호출 제한 120초 안)
RENDER_LOCK = threading.Lock()


def purge_old_sessions() -> None:
    """기한이 지난 분석 세션을 지운다. 분석 요청마다 도는 가벼운 조회다."""

    if not WORK_DIR.is_dir():
        return

    cutoff = time.time() - SESSION_MAX_AGE_SECONDS

    for session_dir in WORK_DIR.iterdir():
        try:
            if session_dir.is_dir() and session_dir.stat().st_mtime < cutoff:
                shutil.rmtree(session_dir, ignore_errors=True)
        except OSError:
            log.warning("could not check session %s", session_dir.name)


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

    purge_old_sessions()

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

        # 배경을 잘라 낸 정지 그림. 실패해도 분석은 성공이다 (화면이 원본으로 대신한다)
        make_cutout(temp_path, str(session_dir / "mask.png"), str(session_dir / "cutout.png"))

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

        output_path = session_dir / "animation.mp4"
        waited = time.perf_counter()
        with RENDER_LOCK:
            waited_ms = round((time.perf_counter() - waited) * 1000)
            result = render_animation(
                image_path=str(session_dir / analysis["source"]),
                mask_path=str(session_dir / "mask.png"),
                joints=joint_list,
                bbox=analysis["bbox"],
                motion=motion,
                output_path=str(output_path),
            )

        log.info(
            "render %s: %s, %s bytes, %s ms (waited %s ms)",
            request_id,
            result["motion"],
            result["size_bytes"],
            result["processing_time_ms"],
            waited_ms,
        )

        return FileResponse(
            output_path,
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


@app.get("/internal/v1/sessions/{request_id}/cutout")
def cutout(request_id: str):
    """배경을 투명하게 잘라 낸 캐릭터 PNG. 분석할 때 만들어 둔 것이다."""

    path = WORK_DIR / request_id / "cutout.png"
    # request_id 로 폴더 밖을 가리키지 못하게 한다
    if not path.resolve().is_relative_to(WORK_DIR.resolve()) or not path.is_file():
        return _error("CUTOUT_NOT_FOUND", f"no cutout for {request_id}")
    return FileResponse(path, media_type="image/png")


@app.delete("/internal/v1/sessions/{request_id}")
def delete_session(request_id: str):
    """Backend가 결과를 받아간 뒤 임시 파일을 정리한다."""

    shutil.rmtree(WORK_DIR / request_id, ignore_errors=True)
    return {"success": True}


def _error(code: str, message: str) -> dict:
    """analyze와 같은 형식으로 오류를 돌려준다. Backend가 code를 읽는다."""

    return {"success": False, "message": f"{code}: {message}"}
