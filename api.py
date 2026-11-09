"""drawtale-tts HTTP API. 백엔드만 부른다 (Bearer 토큰).

  GET    /internal/v1/health               상태 (토큰 없이)
  POST   /internal/v1/voices               보호자 목소리 등록 → voice_id
  POST   /internal/v1/tts                  voice_id + 글 → MP3
  POST   /internal/v1/dialogue             여러 목소리의 대사 → MP3 하나 (캐릭터 목소리 판)
                                           줄마다 여기 목소리(voice_id + text) 또는 이미 만든 음성(audio)
  DELETE /internal/v1/voices/{voice_id}    목소리 지우기 (없어도 204)

오류는 {"error": {"code": "...", "message": "..."}} 모양으로 돌려준다.
"""

import base64
import binascii
import hmac
import logging
import time
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, Request, Response, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

import audio
from store import VoiceStore

log = logging.getLogger(__name__)

MAX_UPLOAD_MB = 20


class ApiError(Exception):
    def __init__(self, code: str, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.code, self.message, self.http_status = code, message, http_status


class TTSRequest(BaseModel):
    voice_id: str
    text: str = Field(min_length=1)
    language: str = "Korean"
    # 아이가 따라올 수 있게 천천히 (백엔드 TTS_SPEED 0.9 와 맞춘다). 음높이는 그대로
    speed: float = Field(default=1.0, ge=0.5, le=1.5)
    format: Literal["mp3", "wav"] = "mp3"


MAX_LINE_AUDIO_MB = 5


class DialogueLine(BaseModel):
    """한 줄. 여기서 읽을 줄은 voice_id + text, 다른 곳(ElevenLabs 등)에서 이미 만든 줄은 audio(base64)."""

    voice_id: str | None = None
    text: str | None = None
    audio: str | None = None

    @model_validator(mode="after")
    def _one_kind(self):
        speak = bool(self.voice_id and (self.text or "").strip())
        if speak == bool(self.audio):
            raise ValueError("줄마다 voice_id + text 또는 audio 중 하나만")
        return self


class DialogueRequest(BaseModel):
    lines: list[DialogueLine] = Field(min_length=1, max_length=40)
    language: str = "Korean"
    speed: float = Field(default=1.0, ge=0.5, le=1.5)
    format: Literal["mp3", "wav"] = "mp3"


def create_app(
    engine,
    store: VoiceStore,
    token: str,
    *,
    max_text_chars: int = 1000,
    ref_min_seconds: float = 3.0,
    ref_max_seconds: float = 30.0,
    chunk_chars: int = 120,
) -> FastAPI:
    if not token:
        # 토큰 없이 열리면 누구나 GPU 를 쓴다. 아예 뜨지 않게 한다
        raise RuntimeError("TTS_API_TOKEN 이 비어 있습니다")

    api = FastAPI(title="drawtale-tts", docs_url=None, redoc_url=None, openapi_url=None)

    @api.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError) -> JSONResponse:
        return JSONResponse({"error": {"code": e.code, "message": e.message}}, status_code=e.http_status)

    @api.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "요청 형식이 올바르지 않아요.",
                    # e.errors() 에는 JSON 으로 못 바꾸는 예외 객체가 들어 있을 수 있다
                    "detail": [{"loc": list(err["loc"]), "msg": err["msg"]} for err in e.errors()],
                }
            },
            status_code=422,
        )

    def auth(authorization: str | None = Header(default=None)) -> None:
        given = (authorization or "").removeprefix("Bearer ").strip()
        if not given or not hmac.compare_digest(given.encode(), token.encode()):
            raise ApiError("UNAUTHORIZED", "인증이 필요해요.", 401)

    @api.get("/internal/v1/health")
    def health() -> dict:
        return {"status": "ok", "model": engine.model_id}

    @api.post("/internal/v1/voices", status_code=status.HTTP_201_CREATED, dependencies=[Depends(auth)])
    def register(
        audio_file: UploadFile = File(alias="audio"),
        ref_text: str | None = Form(default=None, max_length=500),
    ) -> dict:
        data = audio_file.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
        if not data or len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise ApiError("INVALID_AUDIO", f"녹음은 {MAX_UPLOAD_MB}MB 이하만 올릴 수 있어요.")
        ref_text = (ref_text or "").strip() or None
        try:
            wav, seconds = audio.prepare_ref(
                data, ref_min_seconds, ref_max_seconds, cut_to_max=ref_text is None
            )
        except audio.AudioError as e:
            raise ApiError(e.code, e.message) from e
        finally:
            del data

        voice = store.create(wav, ref_text, seconds)
        try:
            engine.prepare(voice)
        except Exception as e:
            log.exception("목소리 준비 실패 %s", voice.id)
            store.delete(voice.id)
            raise ApiError("VOICE_FAILED", "이 녹음으로 목소리를 만들지 못했어요.", 502) from e
        log.info("목소리 등록 %s (%s, %.1fs)", voice.id, voice.mode, seconds)
        return {"voice_id": voice.id, "mode": voice.mode, "seconds": seconds}

    @api.post("/internal/v1/tts", dependencies=[Depends(auth)])
    def tts(req: TTSRequest) -> Response:
        text = req.text.strip()
        if len(text) > max_text_chars:
            raise ApiError("TEXT_TOO_LONG", f"글은 {max_text_chars}자까지 읽을 수 있어요.")
        voice = store.get(req.voice_id)
        if voice is None:
            raise ApiError("VOICE_NOT_FOUND", "목소리를 찾을 수 없어요.", 404)

        start = time.perf_counter()
        try:
            chunks, sample_rate = [], 24000
            for part in audio.split_text(text, chunk_chars):
                samples, sample_rate = engine.synthesize(voice, part, req.language)
                chunks.append(samples)
            body = audio.encode(audio.join(chunks, sample_rate), sample_rate, req.format, req.speed)
        except Exception as e:
            log.exception("음성 생성 실패 %s", voice.id)
            raise ApiError("TTS_FAILED", "음성을 만들지 못했어요.", 502) from e
        elapsed = time.perf_counter() - start
        log.info("tts %s %d자 %d조각 %.1fs", voice.id, len(text), len(chunks), elapsed)
        return Response(
            body,
            media_type="audio/mpeg" if req.format == "mp3" else "audio/wav",
            headers={"X-Processing-Seconds": f"{elapsed:.2f}"},
        )

    @api.post("/internal/v1/dialogue", dependencies=[Depends(auth)])
    def dialogue(req: DialogueRequest) -> Response:
        """대사마다 그 목소리로 읽어 차례로 잇는다. 한 줄이라도 목소리가 없으면 404."""

        total = sum(len((line.text or "").strip()) for line in req.lines if line.voice_id)
        if total > max_text_chars:
            raise ApiError("TEXT_TOO_LONG", f"글은 {max_text_chars}자까지 읽을 수 있어요.")
        clips: dict[int, bytes] = {}
        for i, line in enumerate(req.lines):
            if line.audio:
                try:
                    clips[i] = base64.b64decode(line.audio, validate=True)
                except (binascii.Error, ValueError) as e:
                    raise ApiError("INVALID_AUDIO", "대사 음성을 읽을 수 없어요.") from e
                if not clips[i] or len(clips[i]) > MAX_LINE_AUDIO_MB * 1024 * 1024:
                    raise ApiError("INVALID_AUDIO", f"대사 음성은 {MAX_LINE_AUDIO_MB}MB 이하만 받아요.")
        voices = {}
        for line in req.lines:
            if line.voice_id and line.voice_id not in voices:
                voice = store.get(line.voice_id)
                if voice is None:
                    raise ApiError("VOICE_NOT_FOUND", "목소리를 찾을 수 없어요.", 404)
                voices[line.voice_id] = voice

        start = time.perf_counter()
        try:
            spoken: dict[int, list] = {}
            sample_rate = 24000
            for i, line in enumerate(req.lines):
                if line.voice_id:
                    spoken[i] = []
                    for part in audio.split_text(line.text.strip(), chunk_chars):
                        samples, sample_rate = engine.synthesize(voices[line.voice_id], part, req.language)
                        spoken[i].append(samples)
            chunks = []
            for i, line in enumerate(req.lines):
                if line.voice_id:
                    chunks.extend(spoken[i])
                else:
                    try:
                        chunks.append(audio.decode(clips[i], sample_rate))
                    except audio.AudioError as e:
                        raise ApiError("INVALID_AUDIO", "대사 음성을 읽을 수 없어요.") from e
            # 빠르기는 여기서 읽은 줄에만 맞추고 싶지만 한 파일이라 전체에 건다 — 보내는 쪽이 정한다
            body = audio.encode(audio.join(chunks, sample_rate), sample_rate, req.format, req.speed)
        except ApiError:
            raise
        except Exception as e:
            log.exception("대사 음성 생성 실패")
            raise ApiError("TTS_FAILED", "음성을 만들지 못했어요.", 502) from e
        elapsed = time.perf_counter() - start
        log.info(
            "dialogue %d줄 (만든 음성 %d) %d목소리 %d자 %.1fs",
            len(req.lines), len(clips), len(voices), total, elapsed,
        )
        return Response(
            body,
            media_type="audio/mpeg" if req.format == "mp3" else "audio/wav",
            headers={"X-Processing-Seconds": f"{elapsed:.2f}"},
        )

    @api.delete("/internal/v1/voices/{voice_id}", status_code=204, dependencies=[Depends(auth)])
    def delete(voice_id: str) -> Response:
        engine.forget(voice_id)
        if store.delete(voice_id):
            log.info("목소리 삭제 %s", voice_id)
        return Response(status_code=204)

    return api
