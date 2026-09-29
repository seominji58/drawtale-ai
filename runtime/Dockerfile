FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ANIMATED_DRAWINGS_DIR=/opt/AnimatedDrawings

# libosmesa6: headless OpenGL rendering (no display on the target VM)
# ffmpeg: OpenCV has no H.264 encoder, so GIF output is converted to MP4
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libosmesa6 libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY runtime/requirements.txt .
RUN pip install -r requirements.txt

# Meta rendering code, imported by ai_server/renderer.py
COPY third_party/AnimatedDrawings /opt/AnimatedDrawings
ENV PYTHONPATH=/opt/AnimatedDrawings

COPY runtime/ai_server ./ai_server

RUN useradd --create-home appuser && mkdir -p /tmp/drawtale && chown appuser /tmp/drawtale
USER appuser

EXPOSE 8001
CMD ["uvicorn", "ai_server.main:app", "--host", "0.0.0.0", "--port", "8001"]
