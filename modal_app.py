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
# flash-attn (2026-11-11 시험): 생성이 빨라질 수 있다. 끄려면 False 로 바꿔 다시 배포
FLASH_ATTN = True
# 앞에서부터 남는 GPU 를 쓴다. L4 가 모자라면 오래 기다리지 않고 다음 것으로 (2026-11-09 L4 대기 확인)
# T4 는 flash-attn 을 쓸 수 없어(Ampere 이후만) flash-attn 을 켜면 뺀다
GPU = ["L4", "A10G"] if FLASH_ATTN else ["L4", "A10G", "T4"]
# auto: GPU 가 bf16 을 지원하면(L4 · A10G) bfloat16, 아니면(T4) float16
DTYPE = "auto"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "sox")  # sox: qwen-tts 가 찾는다
    # 실제로 돌려 본 뒤 버전을 고정한다 (예: qwen-tts==x.y.z)
    .pip_install("qwen-tts", "soundfile", "numpy", "fastapi", "python-multipart")
)
if FLASH_ATTN:
    # 설치된 torch 에 맞는 미리 빌드된 판을 받는다 (여기엔 CUDA 컴파일러가 없어 직접 빌드는 못 한다).
    # 맞는 판이 없으면 설치에 실패하고, 그래도 이미지는 만들어진다 — 서버는 flash-attn 없이 지금처럼 뜬다
    image = image.pip_install("packaging", "ninja", "psutil", "wheel").run_commands(
        "pip install flash-attn --no-build-isolation"
        " || echo '[image] flash-attn 설치 실패: 없이 진행합니다'"
    )
image = (
    image.env(
        {
            "HF_HOME": "/cache/huggingface",
            # GPU 코드 컴파일 결과를 Volume 에 남겨, 새로 켜지는 컨테이너가 다시 컴파일하지 않게
            "TRITON_CACHE_DIR": "/cache/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
            "TRITON_DEFAULT_FP_FUSION": "1",  # 지금 동작 그대로 (경고만 끈다)
        }
    )
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
    # 동시에 켤 수 있는 최대 대수 = 비용 상한. 백엔드가 쪽 음성을 5개씩 동시에 요청한다(PAGE_AUDIO_WORKERS)
    # 비용은 GPU 를 쓴 시간이라, 같은 일을 2대가 오래 하든 5대가 짧게 하든 거의 같다
    max_containers=5,
)
class TTSService:
    @modal.enter()
    def load(self) -> None:
        from engine import QwenEngine

        self.engine = QwenEngine(MODEL_ID, dtype=DTYPE)
        self.engine.warmup()
        try:
            # 모델 파일 · 컴파일 결과를 남긴다. 여러 컨테이너가 동시에 남겨도 실패는 무시한다
            hf_cache.commit()
        except Exception:
            pass

    @modal.asgi_app()
    def web(self):
        from api import create_app
        from store import VoiceStore

        store = VoiceStore(Path("/voices"), commit=voices.commit, reload=voices.reload)
        return create_app(self.engine, store, token=os.environ["TTS_API_TOKEN"])
