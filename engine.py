"""Qwen3-TTS 를 감싼다. GPU 가 있는 곳(Modal)에서만 만든다.

시험에서는 같은 모양(prepare · synthesize · forget · model_id)의 가짜 엔진을 쓴다.
"""

import logging
import threading
from collections import OrderedDict

import numpy as np

from tts_service.store import Voice

log = logging.getLogger(__name__)


class QwenEngine:
    def __init__(self, model_id: str, dtype: str = "bfloat16", device: str = "cuda:0", cache_size: int = 64) -> None:
        import torch
        from qwen_tts import Qwen3TTSModel

        self.model_id = model_id
        self.model = Qwen3TTSModel.from_pretrained(
            model_id, device_map=device, dtype=getattr(torch, dtype)
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

    def forget(self, voice_id: str) -> None:
        with self._lock:
            self._prompts.pop(voice_id, None)
