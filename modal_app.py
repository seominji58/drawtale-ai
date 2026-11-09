"""drawtale-tts 를 Modal(GPU)에 띄운다.

    modal serve modal_app.py     # 개발용 임시 주소 (코드 바꾸면 바로 반영)
    modal deploy modal_app.py    # 고정 주소로 배포

필요한 Secret (한 번만):
    modal secret create drawtale-tts TTS_API_TOKEN=<긴 임의 문자열>
"""

import os
from pathlib import Path

import modal

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
# 앞에서부터 남는 GPU 를 쓴다. L4 가 모자라면 오래 기다리지 않고 다음 것으로 (2026-11-09 L4 대기 확인)
GPU = ["L4", "A10G", "T4"]
# auto: GPU 가 bf16 을 지원하면(L4 · A10G) bfloat16, 아니면(T4) float16
DTYPE = "auto"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "sox")  # sox: qwen-tts 가 찾는다
    # 실제로 돌려 본 뒤 버전을 고정한다 (예: qwen-tts==x.y.z)
    .pip_install("qwen-tts", "soundfile", "numpy", "fastapi", "python-multipart")
    .env({"HF_HOME": "/cache/huggingface"})
    .add_local_python_source("api", "audio", "engine", "store")
)

hf_cache = modal.Volume.from_name("drawtale-tts-hf-cache", create_if_missing=True)
voices = modal.Volume.from_name("drawtale-tts-voices", create_if_missing=True)

app = modal.App("drawtale-tts")


@app.cls(
    gpu=GPU,
    image=image,
    volumes={"/cache": hf_cache, "/voices": voices},
    secrets=[modal.Secret.from_name("drawtale-tts")],
    timeout=600,
    # 이야기 한 편은 쪽마다 차례로 부른다. 그동안 컨테이너를 살려 둬 콜드 스타트를 한 번만 겪게
    scaledown_window=300,
    # 비용 상한. 동시에 이야기를 만드는 사람이 많아지면 올린다
    max_containers=2,
)
class TTSService:
    @modal.enter()
    def load(self) -> None:
        from engine import QwenEngine

        self.engine = QwenEngine(MODEL_ID, dtype=DTYPE)
        try:
            hf_cache.commit()
        except Exception:
            pass

    @modal.asgi_app()
    def web(self):
        from api import create_app
        from store import VoiceStore

        store = VoiceStore(Path("/voices"), commit=voices.commit, reload=voices.reload)
        return create_app(self.engine, store, token=os.environ["TTS_API_TOKEN"])
