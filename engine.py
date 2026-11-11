"""Qwen3-TTS 를 감싼다. GPU 가 있는 곳(Modal)에서만 만든다.

시험에서는 같은 모양(prepare · synthesize · forget · model_id)의 가짜 엔진을 쓴다.
"""

import logging
import threading
from collections import OrderedDict

import numpy as np

from store import Voice

log = logging.getLogger(__name__)


def _flash_attention(torch) -> str | None:  # noqa: ANN001
    """flash-attn 이 깔려 있고 GPU 가 받쳐 주면(Ampere 이후: A10G · L4) "flash_attention_2"."""

    try:
        import flash_attn  # noqa: F401
    except ImportError:
        return None
    major, _ = torch.cuda.get_device_capability(0)
    if major < 8:
        print("[engine] 이 GPU 는 flash-attn 을 쓸 수 없어 기본 방식으로", flush=True)
        return None
    return "flash_attention_2"


class QwenEngine:
    def __init__(self, model_id: str, dtype: str = "bfloat16", device: str = "cuda:0", cache_size: int = 64) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        if dtype == "auto":
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        self.model_id = model_id
        self.dtype = dtype
        kwargs = {"device_map": device, "dtype": getattr(torch, dtype)}
        # flash-attn 이 있으면 그것, 없으면 PyTorch 내장 가속(SDPA). 둘 다 안 되면 기본 방식
        attn = _flash_attention(torch) or "sdpa"
        try:
            self.model = Qwen3TTSModel.from_pretrained(model_id, attn_implementation=attn, **kwargs)
        except (TypeError, ValueError, ImportError) as e:
            print(f"[engine] attention={attn} 으로 못 불러 기본 방식으로: {e}", flush=True)
            attn = None
            self.model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
        # Modal 로그에 남긴다 (검증 자료: 어떤 GPU · 어떤 방식으로 돌았는지)
        print(
            f"[engine] 모델 로드: {model_id} · GPU {torch.cuda.get_device_name(0)} · {dtype}"
            f" · attention {attn or '기본(PyTorch)'}",
            flush=True,
        )
        self._lock = threading.Lock()  # GPU 하나에 한 번에 하나씩
        self._prompts: OrderedDict[str, dict] = OrderedDict()
        self._cache_size = cache_size

    # 참조 음성 특징은 목소리마다 한 번만 뽑아 메모리에 둔다 (컨테이너가 바뀌면 다시 뽑는다)
    def _voice_kwargs(self, voice: Voice) -> dict:
        if voice.id in self._prompts:
            self._prompts.move_to_end(voice.id)
            return self._prompts[voice.id]

        ref = str(voice.ref_path)
        if hasattr(self.model, "create_voice_clone_prompt"):
            if voice.ref_text:
                prompt = self.model.create_voice_clone_prompt(ref_audio=ref, ref_text=voice.ref_text)
            else:
                prompt = self.model.create_voice_clone_prompt(
                    ref_audio=ref, ref_text=None, x_vector_only_mode=True
                )
            kwargs = {"voice_clone_prompt": prompt}
        elif voice.ref_text:
            kwargs = {"ref_audio": ref, "ref_text": voice.ref_text}
        else:
            raise RuntimeError("이 qwen-tts 판은 글 없는 참조 음성(x-vector)을 지원하지 않습니다")

        self._prompts[voice.id] = kwargs
        if len(self._prompts) > self._cache_size:
            self._prompts.popitem(last=False)
        return kwargs

    def prepare(self, voice: Voice) -> None:
        """등록할 때 한 번 불러, 이 녹음으로 목소리를 만들 수 있는지 미리 확인한다."""

        with self._lock:
            self._voice_kwargs(voice)

    def synthesize(self, voice: Voice, text: str, language: str) -> tuple[np.ndarray, int]:
        with self._lock:
            wavs, sr = self.model.generate_voice_clone(
                text=text, language=language, **self._voice_kwargs(voice)
            )
        return np.asarray(wavs[0], dtype=np.float32), int(sr)

    def warmup(self, text: str = "안녕하세요. 오늘도 재미있는 이야기를 들려줄게요.") -> float | None:
        """예열: 한 번 만들어 봐서 첫 생성 때의 준비 작업(GPU 코드 컴파일)을 미리 끝낸다.

        컴파일 결과는 캐시 폴더(Modal Volume)에 남아 다음에 켜지는 컨테이너가 다시 쓴다.
        실패해도 서버는 그대로 뜬다 (첫 요청이 준비 작업을 할 뿐). 걸린 초를 돌려준다.
        """

        import os
        import tempfile
        import time

        from audio import warmup_reference

        start = time.perf_counter()
        fd, path = tempfile.mkstemp(suffix=".wav")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(warmup_reference())
            with self._lock:
                if hasattr(self.model, "create_voice_clone_prompt"):
                    prompt = self.model.create_voice_clone_prompt(
                        ref_audio=path, ref_text=None, x_vector_only_mode=True
                    )
                    kwargs = {"voice_clone_prompt": prompt}
                else:
                    kwargs = {"ref_audio": path, "ref_text": "아"}
                self.model.generate_voice_clone(text=text, language="Korean", **kwargs)
            elapsed = time.perf_counter() - start
            print(f"[engine] 예열 완료 {elapsed:.1f}s", flush=True)
            return elapsed
        except Exception as e:  # noqa: BLE001
            print(f"[engine] 예열 실패 (서버는 그대로 뜸): {e}", flush=True)
            return None
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def forget(self, voice_id: str) -> None:
        with self._lock:
            self._prompts.pop(voice_id, None)
