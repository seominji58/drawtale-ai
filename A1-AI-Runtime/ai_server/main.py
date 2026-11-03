import os
import time
import tempfile
import subprocess
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)

from ai_server.analyzer import (
    analyze_image,
    check_torchserve,
)

from ai_server.schemas import (
    AnalyzeResponse,
    RenderRequest,
)

from ai_server.renderer import (
    create_animation_annotation,
    apply_joint_corrections,
)


# ============================================================
# Path Settings
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

CAPSTON_ROOT = PROJECT_ROOT.parent

ANIMATED_DRAWINGS_DIR = (
    CAPSTON_ROOT
    / "third_party"
    / "AnimatedDrawings"
)

ANNOTATION_DIR = (
    CAPSTON_ROOT
    / "render_results"
    / "current"
)

MOTION_CONFIG_DIR = (
    ANIMATED_DRAWINGS_DIR
    / "examples"
    / "config"
    / "motion"
)

RETARGET_CONFIG_DEFAULT = (
    ANIMATED_DRAWINGS_DIR
    / "examples"
    / "config"
    / "retarget"
    / "fair1_ppf.yaml"
)


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="DrawTale AI Runtime",
    version="2.0.0",
)


# ============================================================
# Health
# ============================================================

@app.get("/internal/v1/health")
def health():
    model_loaded = check_torchserve()

    return {
        "success": True,
        "status": "ok",
        "model_loaded": model_loaded,
        "runtime_version": "a1-runtime-v2",
    }


# ============================================================
# Analyze
# ============================================================

@app.post(
    "/internal/v1/analyze",
    response_model=AnalyzeResponse,
)
async def analyze(
    file: UploadFile = File(...),
    character_type: str = Form("human"),
    skeleton_type: str | None = Form(None),
):
    start_time = time.perf_counter()

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
                "human, quadruped, or nonhuman"
            ),
        )

    suffix = Path(
        file.filename or "upload.png"
    ).suffix

    if not suffix:
        suffix = ".png"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_path = temp_file.name

            file_bytes = await file.read()

            if not file_bytes:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "INVALID_IMAGE: "
                        "empty file"
                    ),
                )

            temp_file.write(
                file_bytes
            )

        # --------------------------------------------
        # Character type router
        # --------------------------------------------

        result = analyze_image(
            temp_path,
            character_type=character_type,
        )

        if not result.get(
            "success"
        ):
            message = (
                result.get(
                    "message"
                )
                or "ANALYZE_FAILED"
            )

            if (
                "MODEL_TIMEOUT"
                in message
            ):
                raise HTTPException(
                    status_code=504,
                    detail=message,
                )

            if (
                "MODEL_UNAVAILABLE"
                in message
            ):
                raise HTTPException(
                    status_code=503,
                    detail=message,
                )

            if (
                "INVALID_IMAGE"
                in message
            ):
                raise HTTPException(
                    status_code=400,
                    detail=message,
                )

            raise HTTPException(
                status_code=422,
                detail=message,
            )

        # --------------------------------------------
        # Character / Skeleton metadata
        # --------------------------------------------

        result["character_type"] = (
            result.get(
                "character_type",
                character_type,
            )
        )

        if (
            result.get(
                "skeleton_type"
            )
            is None
        ):
            if skeleton_type:
                result[
                    "skeleton_type"
                ] = skeleton_type

            elif (
                character_type
                == "human"
            ):
                result[
                    "skeleton_type"
                ] = "humanoid_15"

            elif (
                character_type
                == "quadruped"
            ):
                result[
                    "skeleton_type"
                ] = "quadruped_16"

            else:
                result[
                    "skeleton_type"
                ] = (
                    "nonhuman_generic"
                )

        # --------------------------------------------
        # Human only:
        # Animated Drawings annotation 생성
        # --------------------------------------------

        if (
            character_type
            == "human"
        ):
            annotation_dir = (
                create_animation_annotation(
                    result,
                    temp_path,
                )
            )

            result["message"] = (
                "Analysis complete. "
                "Animation annotation "
                f"created: {annotation_dir}"
            )

        # quadruped / nonhuman은 아직
        # 기존 humanoid AnimatedDrawings
        # render pipeline에 강제로 넣지 않는다.

        # --------------------------------------------
        # Metadata
        # --------------------------------------------

        result["metadata"] = {
            "model_version": (
                "meta-animated-drawings-v1"
            ),
            "pipeline_version": (
                "a1-runtime-v2"
            ),
            "coordinate_space": (
                "original_image_pixels"
            ),
            "processing_time_ms": round(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000,
                2,
            ),
        }

        return result

    except HTTPException:
        raise

    except Exception as e:
        message = str(e)

        if (
            "MODEL_TIMEOUT"
            in message
        ):
            raise HTTPException(
                status_code=504,
                detail=message,
            )

        if (
            "MODEL_UNAVAILABLE"
            in message
        ):
            raise HTTPException(
                status_code=503,
                detail=message,
            )

        if (
            "INVALID_IMAGE"
            in message
        ):
            raise HTTPException(
                status_code=400,
                detail=message,
            )

        if (
            "NO_CHARACTER_DETECTED"
            in message
        ):
            raise HTTPException(
                status_code=422,
                detail=message,
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "INTERNAL_ERROR: "
                f"{message}"
            ),
        )

    finally:
        if (
            temp_path is not None
            and os.path.exists(
                temp_path
            )
        ):
            os.remove(
                temp_path
            )


# ============================================================
# Render
# ============================================================

@app.post(
    "/internal/v1/render"
)
def render(
    request: RenderRequest | None = None,
):
    """
    현재 Animated Drawings Render는
    humanoid_15 skeleton 기준으로 동작한다.

    quadruped / nonhuman은
    skeleton 규격과 render adapter를
    별도로 연결한 뒤 활성화한다.
    """

    character_type = "human"

    if request is not None:
        character_type = (
            request.character_type
            or "human"
        )

    # --------------------------------------------
    # 현재 Render 지원 범위
    # --------------------------------------------

    if (
        character_type
        == "quadruped"
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "RENDER_UNAVAILABLE: "
                "quadruped_16 render "
                "pipeline is not "
                "connected yet"
            ),
        )

    if (
        character_type
        == "nonhuman"
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "RENDER_UNAVAILABLE: "
                "nonhuman render "
                "pipeline is not "
                "connected yet"
            ),
        )

    if (
        character_type
        != "human"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "INVALID_CHARACTER_TYPE: "
                "human, quadruped, "
                "or nonhuman"
            ),
        )

    annotation_dir = str(
        ANNOTATION_DIR
    )

    animated_drawings_dir = str(
        ANIMATED_DRAWINGS_DIR
    )

    motion_config = str(
        MOTION_CONFIG_DIR
        / "wave_hello.yaml"
    )

    retarget_config = str(
        RETARGET_CONFIG_DEFAULT
    )

    animation_script = str(
        ANIMATED_DRAWINGS_DIR
        / "examples"
        / "annotations_to_animation.py"
    )

    # --------------------------------------------
    # Resource Check
    # --------------------------------------------

    if not ANIMATED_DRAWINGS_DIR.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "RENDER_RESOURCE_MISSING: "
                f"{ANIMATED_DRAWINGS_DIR}"
            ),
        )

    if not Path(
        animation_script
    ).exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "RENDER_RESOURCE_MISSING: "
                f"{animation_script}"
            ),
        )

    if not Path(
        motion_config
    ).exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "RENDER_RESOURCE_MISSING: "
                f"{motion_config}"
            ),
        )

    if not Path(
        retarget_config
    ).exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "RENDER_RESOURCE_MISSING: "
                f"{retarget_config}"
            ),
        )

    if not Path(
        annotation_dir
    ).exists():
        raise HTTPException(
            status_code=400,
            detail=(
                "ANNOTATION_NOT_FOUND: "
                "Run /internal/v1/analyze "
                "with character_type=human "
                "before rendering"
            ),
        )

    command = [
        "python",
        animation_script,
        annotation_dir,
        motion_config,
        retarget_config,
    ]

    try:
        # ----------------------------------------
        # 사용자가 수정한 joint가 있으면 반영
        # ----------------------------------------

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
                corrected_joints,
            )

        # ----------------------------------------
        # Animated Drawings 실행
        # ----------------------------------------

        env = os.environ.copy()

        env["PYTHONPATH"] = (
            animated_drawings_dir
        )

        process = subprocess.run(
            command,
            cwd=animated_drawings_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )

        if (
            process.returncode
            != 0
        ):
            error_message = (
                process.stderr.strip()
                or process.stdout.strip()
                or (
                    "Renderer returned "
                    "a non-zero exit code"
                )
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "RENDER_FAILED: "
                    + error_message
                ),
            )

        video_path = os.path.join(
            annotation_dir,
            "video.gif",
        )

        return {
            "success": True,
            "character_type": (
                "human"
            ),
            "skeleton_type": (
                "humanoid_15"
            ),
            "video": video_path,
            "message": (
                "Rendering complete"
            ),
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
            ),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=(
                "INVALID_JOINTS: "
                f"{e}"
            ),
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "RENDER_RESOURCE_MISSING: "
                f"{e}"
            ),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "RENDER_FAILED: "
                f"{e}"
            ),
        )