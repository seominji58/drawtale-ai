"""GPU 없이 API 전체를 시험한다 — 가짜 엔진 + 임시 폴더 저장소 + 실제 ffmpeg."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tts_service import audio
from tts_service.api import create_app
from tts_service.store import VoiceStore

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class FakeEngine:
    model_id = "fake"

    def __init__(self) -> None:
        self.prepared, self.calls, self.forgotten = [], [], []
        self.fail_prepare = False

    def prepare(self, voice) -> None:
        if self.fail_prepare:
            raise RuntimeError("boom")
        self.prepared.append(voice.id)

    def synthesize(self, voice, text, language):
        self.calls.append((voice.id, text, language))
        return np.full(24000, 0.1, dtype=np.float32), 24000  # 1초

    def forget(self, voice_id) -> None:
        self.forgotten.append(voice_id)


def tone(seconds: float, sr: int = 16000, lead_silence: float = 0.0) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    wave = 0.3 * np.sin(2 * np.pi * 220 * t)
    wave = np.concatenate([np.zeros(int(sr * lead_silence)), wave, np.zeros(int(sr * lead_silence))])
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def engine():
    return FakeEngine()


@pytest.fixture
def client(tmp_path, engine):
    return TestClient(create_app(engine, VoiceStore(tmp_path), TOKEN))


def register(client, data: bytes, ref_text: str | None = "오늘은 친구와 함께 놀이터에 갔어요."):
    form = {"ref_text": ref_text} if ref_text is not None else {}
    return client.post(
        "/internal/v1/voices", headers=AUTH, files={"audio": ("ref.wav", data, "audio/wav")}, data=form
    )


def test_health_without_token(client):
    assert client.get("/internal/v1/health").json()["status"] == "ok"


def test_token_required(client):
    assert client.post("/internal/v1/tts", json={"voice_id": "x", "text": "안녕"}).status_code == 401
    bad = {"Authorization": "Bearer wrong"}
    res = client.delete("/internal/v1/voices/qv_" + "0" * 32, headers=bad)
    assert res.status_code == 401
    assert res.json()["error"]["code"] == "UNAUTHORIZED"


def test_empty_token_refuses_to_start(tmp_path, engine):
    with pytest.raises(RuntimeError):
        create_app(engine, VoiceStore(tmp_path), "")


def test_register_and_speak(client, engine):
    res = register(client, tone(5, lead_silence=1.0))
    assert res.status_code == 201, res.text
    body = res.json()
    assert body["voice_id"].startswith("qv_") and body["mode"] == "text"
    assert 4.5 < body["seconds"] < 5.5  # 앞뒤 무음이 걷혔다
    assert engine.prepared == [body["voice_id"]]

    res = client.post("/internal/v1/tts", headers=AUTH, json={"voice_id": body["voice_id"], "text": "친구와 함께 놀았어요.", "speed": 0.9})
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/mpeg"
    assert res.content[:3] == b"ID3" or res.content[0] == 0xFF
    assert engine.calls == [(body["voice_id"], "친구와 함께 놀았어요.", "Korean")]


def test_wav_output(client):
    vid = register(client, tone(5)).json()["voice_id"]
    res = client.post("/internal/v1/tts", headers=AUTH, json={"voice_id": vid, "text": "안녕", "format": "wav"})
    data, sr = sf.read(io.BytesIO(res.content))
    assert sr == 24000 and len(data) == 24000


def test_long_text_is_split(client, engine):
    vid = register(client, tone(5)).json()["voice_id"]
    text = " ".join(["햇살이 반짝이는 날이었어요, 나는 친구와 함께 숲길을 따라 천천히 걸어갔어요."] * 6)
    res = client.post("/internal/v1/tts", headers=AUTH, json={"voice_id": vid, "text": text})
    assert res.status_code == 200
    assert len(engine.calls) > 1
    assert all(len(t) <= 120 for _, t, _ in engine.calls)


def test_ref_too_short(client):
    res = register(client, tone(1.5))
    assert res.status_code == 400 and res.json()["error"]["code"] == "REF_TOO_SHORT"


def test_silence_only_is_rejected(client):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000 * 5), 16000, format="WAV")
    res = register(client, buf.getvalue())
    assert res.status_code == 400 and res.json()["error"]["code"] in {"REF_TOO_SHORT", "INVALID_AUDIO"}


def test_long_ref_with_text_is_rejected(client):
    res = register(client, tone(40))
    assert res.json()["error"]["code"] == "REF_TOO_LONG"


def test_long_ref_without_text_is_cut(client):
    res = register(client, tone(40), ref_text=None)
    assert res.status_code == 201
    assert res.json()["mode"] == "x_vector" and res.json()["seconds"] <= 30.05


def test_garbage_audio(client):
    res = register(client, b"not audio at all")
    assert res.status_code == 400 and res.json()["error"]["code"] == "INVALID_AUDIO"


def test_prepare_failure_leaves_nothing(client, engine, tmp_path):
    engine.fail_prepare = True
    res = register(client, tone(5))
    assert res.status_code == 502 and res.json()["error"]["code"] == "VOICE_FAILED"
    assert list(tmp_path.iterdir()) == []


def test_unknown_and_malformed_voice(client):
    for vid in ["qv_" + "a" * 32, "../../etc/passwd", "x"]:
        res = client.post("/internal/v1/tts", headers=AUTH, json={"voice_id": vid, "text": "안녕"})
        assert res.status_code == 404 and res.json()["error"]["code"] == "VOICE_NOT_FOUND"


def test_text_too_long(client):
    vid = register(client, tone(5)).json()["voice_id"]
    res = client.post("/internal/v1/tts", headers=AUTH, json={"voice_id": vid, "text": "가" * 1001})
    assert res.json()["error"]["code"] == "TEXT_TOO_LONG"


def test_validation_error_shape(client):
    res = client.post("/internal/v1/tts", headers=AUTH, json={"text": "안녕"})
    assert res.status_code == 422 and res.json()["error"]["code"] == "VALIDATION_ERROR"


def test_delete(client, engine, tmp_path):
    vid = register(client, tone(5)).json()["voice_id"]
    assert client.delete(f"/internal/v1/voices/{vid}", headers=AUTH).status_code == 204
    assert not (tmp_path / vid).exists()
    assert engine.forgotten == [vid]
    res = client.post("/internal/v1/tts", headers=AUTH, json={"voice_id": vid, "text": "안녕"})
    assert res.status_code == 404
    # 이미 없어도 204 (백엔드가 다시 지워도 괜찮다)
    assert client.delete(f"/internal/v1/voices/{vid}", headers=AUTH).status_code == 204


def test_store_reloads_on_miss(tmp_path):
    calls = []
    store = VoiceStore(tmp_path, reload=lambda: calls.append(1))
    assert store.get("qv_" + "b" * 32) is None
    assert calls == [1]


def test_split_text_keeps_short_text():
    assert audio.split_text("  친구와   놀았어요. ") == ["친구와 놀았어요."]
    assert audio.split_text("") == []


def test_split_text_keeps_dialogue_together():
    text = "하루는 친구에게 “미안해.” 하고 말했어요. " * 10
    parts = audio.split_text(text, 60)
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")
    assert all(len(p) <= 60 for p in parts)
