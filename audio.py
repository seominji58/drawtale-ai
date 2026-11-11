"""오디오 처리 — 참조 음성 다듬기, MP3 만들기, 긴 글 나누기. GPU 와 상관없어 로컬에서 시험한다."""

import io
import os
import re
import subprocess
import tempfile

import numpy as np
import soundfile as sf

REF_SAMPLE_RATE = 24000
# 앞뒤 무음을 걷어 낸다 (집에서 녹음하면 앞뒤에 무음이 길다). 가운데 쉼은 그대로 둔다.
# 작게 녹음된 말도 남도록 기준을 낮게 잡는다 (-55dB). 크기는 아래 _LOUDNESS 가 맞춘다
_TRIM_SILENCE = (
    "silenceremove=start_periods=1:start_threshold=-55dB:start_silence=0.1,"
    "areverse,"
    "silenceremove=start_periods=1:start_threshold=-55dB:start_silence=0.1,"
    "areverse"
)
# 소리 크기를 맞춘다 (EBU R128). 작게 녹음돼도 참조 음성 · 결과 음성이 늘 비슷한 크기로.
# 참조 음성이 작으면 만든 목소리도 작게 나오기 쉬워 참조 음성부터 맞춘다
REF_LOUDNESS = "loudnorm=I=-20:TP=-2:LRA=11"
OUT_LOUDNESS = "loudnorm=I=-16:TP=-1.5:LRA=11"  # 휴대폰 · 태블릿에서 듣기 좋은 크기


class AudioError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def _ffmpeg(data: bytes, args: list[str], timeout: float = 60) -> bytes:
    # mp4 · webm 은 파이프로 읽으면 실패할 수 있어 파일로 넘긴다.
    # Windows 에서는 열려 있는 임시 파일을 다른 프로그램(ffmpeg)이 못 읽어서, 닫은 뒤 넘기고 직접 지운다
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "wb") as src:
            src.write(data)
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", path, *args, "pipe:1"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        # 녹음 탓이 아니라 서버 설치 문제다. INVALID_AUDIO 로 감추지 않는다
        raise RuntimeError("ffmpeg 가 설치되어 있지 않습니다 (apt-get install ffmpeg)") from e
    except (OSError, subprocess.TimeoutExpired) as e:
        raise AudioError("INVALID_AUDIO", "녹음 파일을 읽을 수 없어요.") from e
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    if proc.returncode != 0 or not proc.stdout:
        raise AudioError("INVALID_AUDIO", "녹음 파일을 읽을 수 없어요.")
    return proc.stdout


def prepare_ref(data: bytes, min_seconds: float, max_seconds: float, cut_to_max: bool) -> tuple[bytes, float]:
    """참조 음성을 24kHz 모노 WAV 로 바꾸고 앞뒤 무음을 걷어 낸다.

    cut_to_max: 너무 길면 앞부분만 쓴다. ref_text 가 없을 때만 (글과 맞출 필요가 없다).
    """

    # 무음을 먼저 걷고(잡음까지 키우지 않게) 크기를 맞춘다.
    # 두 번에 나눈다: areverse 뒤에 loudnorm 을 한 번에 이으면 ffmpeg 가 멈춘다
    args = ["-ac", "1", "-ar", str(REF_SAMPLE_RATE), "-af", _TRIM_SILENCE]
    if cut_to_max:
        args += ["-t", str(max_seconds)]
    trimmed = _ffmpeg(data, args + ["-f", "wav"])
    wav = _ffmpeg(trimmed, ["-af", REF_LOUDNESS, "-ar", str(REF_SAMPLE_RATE), "-f", "wav"])
    try:
        info = sf.info(io.BytesIO(wav))
    except Exception as e:
        raise AudioError("INVALID_AUDIO", "녹음 파일을 읽을 수 없어요.") from e
    seconds = info.frames / info.samplerate if info.samplerate else 0.0
    if seconds < min_seconds:
        raise AudioError("REF_TOO_SHORT", f"녹음이 너무 짧아요. {min_seconds:g}초 넘게 읽어 주세요.")
    if seconds > max_seconds + 0.05:
        raise AudioError("REF_TOO_LONG", f"녹음이 너무 길어요. {max_seconds:g}초 안으로 읽어 주세요.")
    return wav, round(seconds, 2)


def warmup_reference(seconds: float = 4.0, sample_rate: int = REF_SAMPLE_RATE) -> bytes:
    """서버 예열용 참조 소리 (WAV). 사람 목소리가 아니라 말소리 비슷하게 만든 소리다.

    사용자 목소리를 예열에 쓰지 않으려고 만든다. 음절처럼 끊기는 배음 소리라
    음성 복제가 그대로 받아들인다. 결과 음성은 쓰지 않고 버린다.
    """

    t = np.arange(int(sample_rate * seconds)) / sample_rate
    f0 = 170 + 25 * np.sin(2 * np.pi * 0.7 * t)  # 억양처럼 오르내리는 음높이
    phase = 2 * np.pi * np.cumsum(f0) / sample_rate
    voice = sum((0.5 / k) * np.sin(k * phase) for k in range(1, 12))
    syllables = np.clip(np.sin(2 * np.pi * 3.5 * t), 0, None) ** 0.6  # 초당 3~4 음절
    samples = (0.3 * voice * syllables / np.abs(voice).max()).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


def decode(data: bytes, sample_rate: int) -> np.ndarray:
    """다른 곳에서 만든 음성(MP3 등)을 모노 float 로 풀어 sample_rate 에 맞춘다 (대사 잇기용)."""

    wav = _ffmpeg(data, ["-ac", "1", "-ar", str(sample_rate), "-f", "wav"])
    samples, _ = sf.read(io.BytesIO(wav), dtype="float32")
    return np.asarray(samples, dtype=np.float32).reshape(-1)


def encode(samples: np.ndarray, sample_rate: int, fmt: str = "mp3", speed: float = 1.0) -> bytes:
    """생성된 음성을 MP3(또는 WAV)로. 소리 크기를 맞추고(OUT_LOUDNESS),
    speed 는 음높이를 바꾸지 않고 빠르기만 바꾼다 (ffmpeg atempo).
    대사 잇기에서는 이은 뒤 한 번에 맞춰 줄마다 크기가 덜 들쭉날쭉하다.
    """

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    filters = ([f"atempo={speed}"] if speed != 1.0 else []) + [OUT_LOUDNESS]
    # loudnorm 은 내부에서 표본율을 올리므로 원래 표본율로 되돌린다
    args = ["-ac", "1", "-af", ",".join(filters), "-ar", str(sample_rate)]
    if fmt == "mp3":
        args += ["-codec:a", "libmp3lame", "-b:a", "96k", "-f", "mp3"]
    else:
        args += ["-f", "wav"]
    return _ffmpeg(buf.getvalue(), args)


def join(chunks: list[np.ndarray], sample_rate: int, gap_seconds: float = 0.3) -> np.ndarray:
    """나눠 만든 음성을 짧은 쉼을 두고 잇는다."""

    gap = np.zeros(int(sample_rate * gap_seconds), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        if i:
            parts.append(gap)
        parts.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
    return np.concatenate(parts) if parts else gap


_SENTENCE = re.compile(r"[^.!?…]*[.!?…]+[”\"’」』)]*|[^.!?…]+$")


def split_text(text: str, max_chars: int = 120) -> list[str]:
    """긴 글을 문장 단위로 묶어 나눈다. 한 번에 너무 길게 읽히면 뒤로 갈수록 흐트러지기 쉽다.

    한 문장이 max_chars 보다 길면 그 문장 하나를 그대로 한 덩어리로 둔다.
    """

    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text] if text else []
    groups: list[str] = []
    current = ""
    for piece in _SENTENCE.findall(text):
        if not piece.strip():
            continue
        if current and len(current) + len(piece) > max_chars:
            groups.append(current.strip())
            current = piece
        else:
            current += piece
    if current.strip():
        groups.append(current.strip())
    return groups
