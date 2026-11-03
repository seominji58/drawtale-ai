from ai_server.agent import run_agent_tools

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)

from ai_server.schemas import (
    AnalyzeResponse,
    RenderRequest,
)

from ai_server.analyzer import (
    analyze_image,
    check_torchserve,
)

from ai_server.renderer import (
    create_animation_annotation,
    apply_joint_corrections,
)

from ai_server.story_steps import StorySteps
from ai_server.tts import generate_story_audio

from ai_server.runtime_config import (
    ANNOTATION_DIR,
    ANIMATED_DRAWINGS_DIR,
    MOTION_CONFIG_DIR,
    RETARGET_CONFIG_DEFAULT,
    AUDIO_OUTPUT_DIR,
)

import tempfile
import os
import subprocess
import time


app = FastAPI()


# ---------------------------------------------------------
# 공통 AI Runtime Error 처리
# ---------------------------------------------------------

def raise_ai_http_error(message: str):

    if message.startswith("INVALID_IMAGE"):
        raise HTTPException(
            status_code=400,
            detail=message
        )

    if (
        message.startswith("NO_CHARACTER_DETECTED")
        or message.startswith("INVALID_BBOX")
    ):
        raise HTTPException(
            status_code=422,
            detail=message
        )

    if message.startswith("MODEL_UNAVAILABLE"):
        raise HTTPException(
            status_code=503,
            detail=message
        )

    if message.startswith("MODEL_TIMEOUT"):
        raise HTTPException(
            status_code=504,
            detail=message
        )

    raise HTTPException(
        status_code=500,
        detail=message
    )


# ---------------------------------------------------------
# Health
# ---------------------------------------------------------

@app.get("/internal/v1/health")
def health():

    model_loaded = check_torchserve()

    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "runtime": "animated-drawings"
    }


# ---------------------------------------------------------
# Analyze
# ---------------------------------------------------------

@app.post(
    "/internal/v1/analyze",
    response_model=AnalyzeResponse
)
async def analyze(
    file: UploadFile = File(...),
    character_type: str = Form("human")
):

    temp_path = None
    start_time = time.perf_counter()

    try:

        # -------------------------------------------------
        # character_type 검사
        # -------------------------------------------------

        allowed_character_types = {
            "human",
            "quadruped",
            "nonhuman",
        }

        if character_type not in allowed_character_types:
            raise HTTPException(
                status_code=400,
                detail=(
                    "INVALID_CHARACTER_TYPE: "
                    "character_type must be "
                    "human, quadruped, or nonhuman"
                )
            )

        # -------------------------------------------------
        # 업로드 이미지 임시 저장
        # -------------------------------------------------

        suffix = (
            os.path.splitext(
                file.filename or ""
            )[1]
            or ".tmp"
        )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_path = temp_file.name

            content = await file.read()

            temp_file.write(content)

        # -------------------------------------------------
        # 캐릭터 타입별 Analyze Pipeline
        # -------------------------------------------------

        if character_type == "human":

            # 기존 Meta Animated Drawings 기반
            # 사람형 분석 Pipeline
            result = analyze_image(
                 temp_path,
                 character_type=character_type
             )

            result["character_type"] = "human"
            result["skeleton_type"] = "humanoid_15"

        elif character_type == "quadruped":

            # 이후 quadruped analyzer 연결 예정
            raise HTTPException(
                status_code=503,
                detail=(
                    "MODEL_UNAVAILABLE: "
                    "quadruped_16 analyzer "
                    "is not implemented yet"
                )
            )

        elif character_type == "nonhuman":

            # 이후 nonhuman analyzer 연결 예정
            raise HTTPException(
                status_code=503,
                detail=(
                    "MODEL_UNAVAILABLE: "
                    "nonhuman analyzer "
                    "is not implemented yet"
                )
            )

        # -------------------------------------------------
        # Analyze 실패 처리
        # -------------------------------------------------

        if not result.get("success"):
            raise_ai_http_error(
                result.get("message")
                or "ANALYZE_FAILED"
            )

        # -------------------------------------------------
        # AnimatedDrawings Annotation 생성
        # -------------------------------------------------

        annotation_dir = (
            create_animation_annotation(
                result,
                temp_path
            )
        )

        result["message"] = (
            "Analysis complete. "
            f"Annotation created: "
            f"{annotation_dir}"
        )

        # -------------------------------------------------
        # Runtime Metadata
        # -------------------------------------------------

        result["metadata"] = {
            "model_version":
                "meta-animated-drawings-v1",

            "pipeline_version":
                "a1-runtime-v2",

            "coordinate_space":
                "original_image_pixels",

            "processing_time_ms":
                round(
                    (
                        time.perf_counter()
                        - start_time
                    )
                    * 1000,
                    2
                )
        }

        return AnalyzeResponse(
            **result
        )

    # -----------------------------------------------------
    # FastAPI에서 이미 만든 HTTP 오류
    # -----------------------------------------------------

    except HTTPException:
        raise

    # -----------------------------------------------------
    # AI Runtime 오류
    # -----------------------------------------------------

    except RuntimeError as e:
        raise_ai_http_error(
            str(e)
        )

    # -----------------------------------------------------
    # 파일 누락
    # -----------------------------------------------------

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "ANALYZE_RESOURCE_MISSING: "
                f"{e}"
            )
        )

    # -----------------------------------------------------
    # 기타 오류
    # -----------------------------------------------------

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "ANALYZE_FAILED: "
                f"{e}"
            )
        )

    # -----------------------------------------------------
    # 임시 이미지 삭제
    # -----------------------------------------------------

    finally:

        if (
            temp_path
            and os.path.exists(temp_path)
        ):
            os.remove(
                temp_path
            )


# ---------------------------------------------------------
# Render
# ---------------------------------------------------------

@app.post("/internal/v1/render")
def render(
    request: RenderRequest | None = None
):

    annotation_dir = ANNOTATION_DIR

    animated_drawings_dir = (
        ANIMATED_DRAWINGS_DIR
    )

    # -----------------------------------------------------
    # 기본값
    # -----------------------------------------------------

    character_type = "human"
    skeleton_type = "humanoid_15"

    if request:

        character_type = (
            request.character_type
            or "human"
        )

        skeleton_type = (
            request.skeleton_type
            or (
                "humanoid_15"
                if character_type == "human"
                else None
            )
        )

    # -----------------------------------------------------
    # 현재 Render 지원 범위
    # -----------------------------------------------------

    if character_type == "quadruped":
        raise HTTPException(
            status_code=503,
            detail=(
                "MODEL_UNAVAILABLE: "
                "quadruped render pipeline "
                "is not implemented yet"
            )
        )

    if character_type == "nonhuman":
        raise HTTPException(
            status_code=503,
            detail=(
                "MODEL_UNAVAILABLE: "
                "nonhuman render pipeline "
                "is not implemented yet"
            )
        )

    # -----------------------------------------------------
    # 기존 Human Motion
    # -----------------------------------------------------

    motion_config = os.path.join(
        MOTION_CONFIG_DIR,
        "wave_hello.yaml"
    )

    retarget_config = (
        RETARGET_CONFIG_DEFAULT
    )

    command = [
        "python",

        os.path.join(
            ANIMATED_DRAWINGS_DIR,
            "examples",
            "annotations_to_animation.py"
        ),

        annotation_dir,
        motion_config,
        retarget_config,
    ]

    try:

        # -------------------------------------------------
        # 수정 Joint가 있을 경우 Re-render
        # -------------------------------------------------

        if (
            request
            and request.joints
        ):

            corrected_joints = [
                {
                    "name": joint.name,
                    "x": joint.x,
                    "y": joint.y,
                }
                for joint
                in request.joints
            ]

            apply_joint_corrections(
                annotation_dir,
                corrected_joints
            )

        # -------------------------------------------------
        # AnimatedDrawings 실행 환경
        # -------------------------------------------------

        env = os.environ.copy()

        env["PYTHONPATH"] = (
            animated_drawings_dir
        )

        # -------------------------------------------------
        # Render 실행
        # -------------------------------------------------

        result = subprocess.run(
            command,
            cwd=animated_drawings_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

        # -------------------------------------------------
        # Render 실패
        # -------------------------------------------------

        if result.returncode != 0:

            raise HTTPException(
                status_code=500,
                detail=(
                    "RENDER_FAILED: "
                    + (
                        result.stderr.strip()
                        or (
                            "Renderer returned "
                            "a non-zero exit code"
                        )
                    )
                )
            )

        # -------------------------------------------------
        # Render 성공
        # -------------------------------------------------

        return {
            "success": True,

            "character_type":
                character_type,

            "skeleton_type":
                skeleton_type,

            "video": os.path.join(
                annotation_dir,
                "video.gif"
            ),

            "message":
                "Rendering complete",
        }

    except HTTPException:
        raise

    except subprocess.TimeoutExpired as e:
        raise HTTPException(
            status_code=504,
            detail=(
                "RENDER_TIMEOUT: "
                "Rendering exceeded "
                f"{e.timeout} seconds"
            )
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                "INVALID_JOINTS: "
                f"{e}"
            )
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "RENDER_RESOURCE_MISSING: "
                f"{e}"
            )
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "RENDER_FAILED: "
                f"{e}"
            )
        )


# ---------------------------------------------------------
# TTS
# ---------------------------------------------------------

@app.post("/internal/v1/tts")
def generate_tts(
    story_steps: StorySteps
):

    output_dir = (
        AUDIO_OUTPUT_DIR
    )

    try:

        results = (
            generate_story_audio(
                story_steps,
                output_dir
            )
        )

        return {
            "success": True,
            "audio": results,
            "message":
                "TTS generation complete"
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


# ---------------------------------------------------------
# Agent
# ---------------------------------------------------------

@app.post("/internal/v1/agent/run")
def run_agent(
    story_steps: StorySteps
):

    try:

        result = (
            run_agent_tools(
                story_steps
            )
        )

        return result

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }