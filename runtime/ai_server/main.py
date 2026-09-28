import os
import tempfile

from fastapi import FastAPI, File, UploadFile

from ai_server.analyzer import (
    MODEL_VERSION,
    PIPELINE_VERSION,
    analyze_image,
    check_torchserve,
)
from ai_server.schemas import AnalyzeResponse, HealthResponse

app = FastAPI(title="DrawTale AI Runtime", version=PIPELINE_VERSION)


@app.get("/internal/v1/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=check_torchserve(),
        runtime="animated-drawings",
        model_version=MODEL_VERSION,
        pipeline_version=PIPELINE_VERSION,
    )


# 분석은 CPU를 오래 쓰므로 def로 둔다.
# async def로 두면 분석하는 동안 /health 등 다른 요청까지 멈춘다.
@app.post("/internal/v1/analyze", response_model=AnalyzeResponse)
def analyze(file: UploadFile = File(...)):
    temp_path = None

    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".tmp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            temp_file.write(file.file.read())

        result = analyze_image(temp_path)

        return AnalyzeResponse(**result)

    except FileNotFoundError as e:
        return AnalyzeResponse(
            success=False,
            message=str(e)
        )

    except Exception as e:
        return AnalyzeResponse(
            success=False,
            message=str(e)
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
