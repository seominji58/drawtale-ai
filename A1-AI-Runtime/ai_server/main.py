from fastapi import FastAPI, UploadFile, File
from ai_server.schemas import AnalyzeResponse
from ai_server.analyzer import analyze_image
import tempfile
import os

app = FastAPI()


@app.get("/internal/v1/health")
def health():
    return {
        "status": "ok",
        "model_loaded": False,
        "runtime": "animated-drawings"
    }


@app.post("/internal/v1/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...)):
    temp_path = None

    try:
        suffix = os.path.splitext(file.filename or "")[1] or ".tmp"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

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
