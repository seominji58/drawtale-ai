from ai_server.agent import run_agent_tools
from fastapi import FastAPI, UploadFile, File, HTTPException
from ai_server.schemas import AnalyzeResponse, RenderRequest
from ai_server.analyzer import analyze_image, check_torchserve, check_torchserve
from ai_server.renderer import create_animation_annotation, apply_joint_corrections
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


def raise_ai_http_error(message: str):
    if message.startswith("INVALID_IMAGE"):
        raise HTTPException(status_code=400, detail=message)

    if (
        message.startswith("NO_CHARACTER_DETECTED")
        or message.startswith("INVALID_BBOX")
    ):
        raise HTTPException(status_code=422, detail=message)

    if message.startswith("MODEL_UNAVAILABLE"):
        raise HTTPException(status_code=503, detail=message)

    if message.startswith("MODEL_TIMEOUT"):
        raise HTTPException(status_code=504, detail=message)

    raise HTTPException(status_code=500, detail=message)


@app.get("/internal/v1/health")
def health():
    model_loaded = check_torchserve()

    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "runtime": "animated-drawings"
    }


@app.post("/internal/v1/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    temp_path = None
    start_time = time.perf_counter()

    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".tmp"

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        # ?대?吏 遺꾩꽍
        result = analyze_image(temp_path)

        if not result.get("success"):
            raise_ai_http_error(
                result.get("message") or "ANALYZE_FAILED"
            )

        # 遺꾩꽍 ?깃났 ??AnimatedDrawings annotation ?앹꽦
        if result.get("success"):
            annotation_dir = create_animation_annotation(
                result,
                temp_path
            )

            result["message"] = (
                f"Analysis complete. "
                f"Annotation created: {annotation_dir}"
            )

        result["metadata"] = {
            "model_version": "meta-animated-drawings-v1",
            "pipeline_version": "a1-runtime-v1",
            "coordinate_space": "original_image_pixels",
            "processing_time_ms": round(
                (time.perf_counter() - start_time) * 1000,
                2
            )
        }

        return AnalyzeResponse(**result)

    except HTTPException:
        raise

    except RuntimeError as e:
        raise_ai_http_error(str(e))

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"ANALYZE_RESOURCE_MISSING: {e}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ANALYZE_FAILED: {e}"
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/internal/v1/render")
def render(request: RenderRequest | None = None):
    """
    ?꾩옱 ?앹꽦??AnimatedDrawings annotation??    video.gif濡??뚮뜑留곹븳??
    """

    annotation_dir = ANNOTATION_DIR
    animated_drawings_dir = ANIMATED_DRAWINGS_DIR
    motion_config = os.path.join(
        MOTION_CONFIG_DIR,
        "wave_hello.yaml"
    )
    retarget_config = RETARGET_CONFIG_DEFAULT

    command = [
        "python",
        os.path.join(ANIMATED_DRAWINGS_DIR, "examples", "annotations_to_animation.py"),
        annotation_dir,
        motion_config,
        retarget_config
    ]

    try:
        if request and request.joints:
            corrected_joints = [
                {
                    "name": joint.name,
                    "x": joint.x,
                    "y": joint.y
                }
                for joint in request.joints
            ]

            apply_joint_corrections(
                annotation_dir,
                corrected_joints
            )

        env = os.environ.copy()
        env["PYTHONPATH"] = animated_drawings_dir

        result = subprocess.run(
            command,
            cwd=animated_drawings_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=180
        )

        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=(
                    "RENDER_FAILED: "
                    + (result.stderr.strip() or "Renderer returned a non-zero exit code")
                )
            )

        return {
            "success": True,
            "video": os.path.join(
                annotation_dir,
                "video.gif"
            ),
            "message": "Rendering complete"
        }

    except HTTPException:
        raise

    except subprocess.TimeoutExpired as e:
        raise HTTPException(
            status_code=504,
            detail=f"RENDER_TIMEOUT: Rendering exceeded {e.timeout} seconds"
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"INVALID_JOINTS: {e}"
        )

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=f"RENDER_RESOURCE_MISSING: {e}"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RENDER_FAILED: {e}"
        )

@app.post("/internal/v1/tts")
def generate_tts(story_steps: StorySteps):
    """
    Story Steps??媛?臾몄옣??ElevenLabs TTS濡?蹂?섑븳??
    """

    output_dir = AUDIO_OUTPUT_DIR

    try:
        results = generate_story_audio(
            story_steps,
            output_dir
        )

        return {
            "success": True,
            "audio": results,
            "message": "TTS generation complete"
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }

@app.post("/internal/v1/agent/run")
def run_agent(story_steps: StorySteps):
    """
    Agent Tool ?몄텧 ?먮쫫???ㅽ뻾?쒕떎.
    """

    try:
        result = run_agent_tools(story_steps)

        return result

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }
















