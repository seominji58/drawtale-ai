# drawtale-tts

보호자 목소리로 이야기를 읽어 주는 **음성 복제 TTS 서버**입니다 (Qwen3-TTS, Modal GPU).
ElevenLabs Instant Voice Cloning 자리를 대신합니다. 백엔드(`drawtale-backend`)만 HTTP 로 부릅니다.

- 그림 AI(`drawtale-ai`)와 **저장소 · 배포 · GPU 를 따로** 둡니다. 서로 배포가 묶이지 않습니다.
- 백엔드는 이 서버 코드를 import 하지 않고 HTTP 로만 부릅니다 (`ai_client.py` 와 같은 원칙).

## 구조

```
modal_app.py      Modal 배포 (GPU · Volume · Secret)
api.py            HTTP API (FastAPI) — 인증 · 검사 · 오류 모양
engine.py         Qwen3-TTS 감싸기 (GPU 에서만 만든다)
store.py          보호자 목소리 보관 (Modal Volume 위 폴더)
audio.py          참조 음성 다듬기 · MP3 만들기 · 긴 글 나누기
test_api.py       GPU 없이 API 전체 시험 (가짜 엔진 + 실제 ffmpeg)
```

## 지금 구현된 API

> 계약서가 정해지면 그에 맞춰 바꿉니다. 아래는 지금 코드의 동작입니다.

모든 요청에 `Authorization: Bearer <TTS_API_TOKEN>` (health 만 빼고).
오류는 `{"error": {"code": "...", "message": "..."}}` — `message` 는 사용자에게 보여 줘도 되는 말입니다.

| 요청 | 하는 일 | 응답 |
|---|---|---|
| `GET /internal/v1/health` | 상태 | `{"status": "ok", "model": "..."}` |
| `POST /internal/v1/voices` (multipart: `audio`, `ref_text` 선택) | 목소리 등록 | `201 {"voice_id": "qv_…", "mode": "text" \| "x_vector", "seconds": 8.4}` |
| `POST /internal/v1/tts` (JSON: `voice_id`, `text`, `language`=Korean, `speed`=1.0, `format`=mp3) | 읽기 | `200 audio/mpeg` 바이트 |
| `DELETE /internal/v1/voices/{voice_id}` | 지우기 | `204` (원래 없어도 204) |

오류 code: `UNAUTHORIZED`(401) · `INVALID_AUDIO` · `REF_TOO_SHORT` · `REF_TOO_LONG` · `TEXT_TOO_LONG`(400) ·
`VOICE_NOT_FOUND`(404) · `VALIDATION_ERROR`(422) · `VOICE_FAILED` · `TTS_FAILED`(502)

### 참조 음성 규칙

- 앞뒤 무음은 서버가 걷어 냅니다. 무음을 뺀 길이가 **3~30초**여야 합니다.
- **`ref_text` 를 주면 (권장, `mode: text`)** 녹음에서 실제로 읽은 글과 **정확히** 같아야 합니다.
  녹음 화면에서 정해진 글을 읽게 하고 그 글을 함께 보내면 됩니다. 30초를 넘으면 `REF_TOO_LONG`.
- **`ref_text` 가 없으면 (`mode: x_vector`)** 글 없이 목소리 특징만 씁니다. 길면 앞 30초만 씁니다.
  편하지만 닮은 정도가 떨어질 수 있습니다 — 두 방식을 들어 보고 고르세요.

### 읽기 규칙

- `text` 는 1,000자까지. 120자가 넘으면 문장 단위로 나눠 만들고 0.3초 쉼을 두고 잇습니다.
- `speed` 는 음높이를 바꾸지 않고 빠르기만 바꿉니다 (0.5~1.5). 백엔드 `TTS_SPEED = 0.9` 와 맞추려면 `0.9`.

## 로컬 시험 (GPU 필요 없음)

```bash
pip install -r requirements-dev.txt      # ffmpeg 도 깔려 있어야 합니다
pytest -q
```

## Modal 배포

```bash
modal setup                                                  # 처음 한 번 로그인
modal secret create drawtale-tts TTS_API_TOKEN=$(openssl rand -hex 32)
modal serve modal_app.py      # 개발: 임시 주소, 코드 바꾸면 바로 반영
modal deploy modal_app.py     # 배포: 고정 주소 (https://<workspace>--drawtale-tts-ttsservice-web.modal.run 모양)
```

확인:

```bash
URL=https://...modal.run ; TOKEN=...
curl $URL/internal/v1/health
curl -X POST $URL/internal/v1/voices -H "Authorization: Bearer $TOKEN" \
  -F audio=@ref_audio.wav -F ref_text="오늘은 친구와 함께 놀이터에 갔어요."
curl -X POST $URL/internal/v1/tts -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"voice_id":"qv_...","text":"친구와 함께 재미있게 놀았어요.","speed":0.9}' -o out.mp3
```

첫 요청은 이미지 빌드 · 모델 다운로드로 오래 걸립니다. 이후에도 컨테이너가 꺼져 있다가 켜질 때(콜드 스타트)는
수십 초가 걸리니 **백엔드 쪽 타임아웃을 넉넉히** (예: 120초) 잡아 주세요.

### 조정할 곳 (`modal_app.py`)

| 값 | 기본 | 뜻 |
|---|---|---|
| `GPU` · `DTYPE` | `L4` · `bfloat16` | T4 로 바꾸면 `float16` |
| `MODEL_ID` | 0.6B | 1.7B 로 바꿀 때 여기만 |
| `scaledown_window` | 300초 | 이야기 한 편의 쪽들을 차례로 읽는 동안 컨테이너를 살려 둠. 길수록 유휴 비용 |
| `max_containers` | 2 | **비용 상한.** 동시에 만드는 사람이 많아지면 올림 |

## 데이터 · 개인정보

- 참조 음성은 **Modal Volume `drawtale-tts-voices`** 에 목소리마다 폴더 하나로 둡니다 (`ref.wav`, `meta.json`).
  Modal 은 미국 회사라 **국외 이전 대상이 ElevenLabs → Modal 로 바뀝니다.** 백엔드 동의 안내문(`consent_version`)을 바꿔야 합니다.
- `DELETE` 하면 폴더를 지우고 메모리 캐시도 비웁니다. 백엔드의 탈퇴 · 동의 철회 흐름(`_forget`)에서 부르면 됩니다.
- 녹음을 받은 원본 바이트는 다듬은 WAV 만 남기고 버립니다.

## 아직 확인하지 않은 것

- **실제 GPU 에서 돌려 보지 않았습니다.** API · 오디오 처리 · 저장은 `test_api.py` 시험(19개)으로 확인했고, Qwen 호출부(`engine.py`)는 Colab 노트북과 같은 호출을 씁니다.
- `create_voice_clone_prompt` · `x_vector_only_mode` 는 qwen-tts 판에 따라 이름이 다를 수 있습니다. 처음 배포할 때 로그를 확인하세요. `create_voice_clone_prompt` 가 없는 판이면 `mode: text` 만 동작합니다.
- 돌아가는 것을 확인하면 `modal_app.py` 의 `pip_install` 버전을 고정하세요.
