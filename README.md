# drawtale-ai

DrawTale AI Runtime — 손그림에서 캐릭터 · 관절을 찾고 애니메이션을 렌더링하는 **내부 전용 AI 서버**입니다.
Backend만 호출하며 외부(Frontend, 인터넷)에는 공개하지 않습니다.

- API 계약서: `drawtale-backend/docs/api-contract.md` 3장 (Backend ↔ AI)
- 담당: A1 (Runtime) · A2 (Pose 모델 학습)
- 진행 상황 (항목별 상태): [docs/진행상황.md](docs/진행상황.md)

## 구조

```text
drawtale-ai/
├── runtime/                     # A1 · AI 서버 (FastAPI, Python 3.10)
│   ├── ai_server/
│   │   ├── main.py              #   엔드포인트
│   │   ├── analyzer.py          #   검출 → crop → 관절 추정 → 마스크
│   │   └── schemas.py           #   요청/응답 형식
│   ├── Dockerfile
│   └── requirements.txt
├── torchserve/                  # Meta 모델 서버 (TorchServe, Python 3.8) Docker
├── third_party/AnimatedDrawings/  # Meta 원본 코드 (MIT) — 수정하지 않음, render에서 사용 예정
├── training/                    # A2 · Pose 모델 학습
└── docker-compose.yml           # torchserve + ai 함께 실행
```

```text
Backend(8000) ──HTTP──▶ ai(8001) ──HTTP──▶ torchserve(8080, Meta 검출 · 관절 모델)
```

## 실행 (Docker, 권장)

Docker Desktop만 있으면 됩니다. 첫 빌드는 PyTorch 설치와 Meta 모델 다운로드 때문에 오래 걸립니다(10~30분).

```bash
docker compose up -d --build
docker compose ps            # torchserve가 healthy가 되면 ai가 시작됨
```

- AI 서버 문서: http://127.0.0.1:8001/docs
- 헬스체크: http://127.0.0.1:8001/internal/v1/health
- TorchServe 상태: http://127.0.0.1:8080/ping
- 로그: `docker compose logs -f ai` · 종료: `docker compose down`

### 분석 테스트

```bash
curl -F "file=@third_party/AnimatedDrawings/test_annotations/image.png" \
     http://127.0.0.1:8001/internal/v1/analyze
```

## 로컬 실행 (Docker 없이, 개발용)

TorchServe는 Docker로 띄우고, AI 서버만 로컬에서 실행할 수 있습니다.

```bash
docker compose up -d torchserve

cd runtime
python -m venv .venv
# Windows: .venv\Scripts\activate   /   macOS·Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn ai_server.main:app --reload --port 8001
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `TORCHSERVE_URL` | `http://127.0.0.1:8080` | Meta 모델 서버 주소 (Docker 안에서는 `http://torchserve:8080`) |

## 모델 파일

Meta 사전학습 모델(`drawn_humanoid_detector.mar`, `drawn_humanoid_pose_estimator.mar`)은
`torchserve/Dockerfile`이 빌드할 때 Meta GitHub Release에서 자동으로 받습니다. Git에 올리지 않습니다.

- https://github.com/facebookresearch/AnimatedDrawings/releases/tag/v0.0.1

## 현재 상태

| 엔드포인트 | 상태 |
|---|---|
| `GET /internal/v1/health` | ✅ 동작 — 실제 TorchServe 상태 반영 |
| `POST /internal/v1/analyze` | ✅ 동작 — 파일 업로드(`file`) → bbox, 관절 15개(관절마다 `score`), 검출 `confidence`, 마스크 크기, 메타데이터 |
| `POST /internal/v1/render` | ✅ 동작 — 보정 관절 + motion → MP4 |
| `DELETE /internal/v1/sessions/{id}` | ✅ 동작 — 임시 파일 정리 |

Backend 연동 테스트 완료 (2026-09-28): 업로드 → 분석 → 관절 15개 반환, 에러 처리까지 확인.

**측정값** (20코어 PC, CPU only)

| 항목 | 값 |
|---|---|
| 분석 시간 | 약 1.5초 (2000x2000 기준) |
| 렌더링 시간 | 약 35~40초 (839프레임) |
| 결과 MP4 크기 | 0.6~1.7MB |
| TorchServe 메모리 | 약 1.4GB |
| AI 서버 메모리 | 약 100MB |
| TorchServe 이미지 | 5.84GB (CPU 전용) |
| AI 서버 이미지 | 1.15GB |

> **작업자 수 제한**: TorchServe는 기본적으로 CPU 코어 수만큼 작업자를 만들어 모델을 그만큼 복제한다.
> 20코어에서 17GB를 써서 16GB VM에 올릴 수 없었기 때문에,
> `torchserve/config.properties`에 `default_workers_per_model=1`을 설정했다.
>
> **CPU 전용 PyTorch**: Meta 원본 Dockerfile은 GPU(CUDA) 버전을 설치해 이미지가 32.8GB였다.
> 배포 대상 Azure VM에 GPU가 없으므로 CPU 전용 wheel로 교체해 5.84GB로 줄였다.
> 분석 결과는 동일하며 오히려 더 빠르다 (3.2초 → 1.4초).

## 사용 흐름

```text
1. POST /internal/v1/analyze  (그림 파일)
   → request_id, bbox, 관절 15개  … 원본과 마스크는 서버에 보관됨

2. (사용자가 Frontend에서 관절 보정)

3. POST /internal/v1/render  (request_id, 보정 관절, motion)
   → MP4 파일

4. DELETE /internal/v1/sessions/{request_id}  (정리)
```

**사용 가능한 motion**: `wave_hello`, `jumping`, `jumping_jacks`, `dab`, `zombie`,
**`wave_hello_gentle`, `jumping_gentle`** (순화 동작, 아래)

### 순화 동작 (`*_gentle`)

아이 그림은 팔을 몸통에 붙여 그리는 경우가 많다. 그런 그림은 **팔이 몸 쪽으로 내려오거나 팔꿈치가
접힐 때** 겨드랑이 메시가 몸통 위로 눌려 팔이 덩어리처럼 뭉개진다 (팔이 벌어져 있을 때는 멀쩡하다).

`*_gentle` 은 Meta BVH 에서 만든 판이다.

- 팔(쇄골·위팔) 회전을 50% 로 줄인다 — fair1 기본 자세가 T자라서 팔이 벌어진 쪽에 머문다
- 팔꿈치(아래팔) 회전은 20% 만 남긴다 — 거의 편 채로
- 몸 전체의 수평 이동을 고정한다 — 제자리에서 움직여 벌린 손이 화면 밖으로 잘리지 않는다

`runtime/tools/make_gentle_motions.py` 가 만들고 결과(`runtime/ai_server/motions/`)를 커밋해 둔다.
`third_party/` 는 고치지 않는다. cmu1 뼈대(`jumping_jacks`)는 기본 자세가 팔 내림이라 같은 방법이
맞지 않아 만들지 않았다. 원본 동작은 그대로 남아 있다.

## 완료된 작업

- [x] 관절 · bbox 좌표를 원본 이미지 기준으로 변환
- [x] 응답 메타데이터: `model_version`, `pipeline_version`, `coordinate_space`, `processing_time_ms`
- [x] health의 `model_loaded`를 실제 TorchServe 상태로 표시
- [x] `async def` → `def` (처리 중 서버 전체가 멈추지 않게)
- [x] TorchServe 작업자 수 제한 (17GB → 1.5GB)
- [x] TorchServe 주소를 `TORCHSERVE_URL` 환경변수로 분리
- [x] CPU 전용 PyTorch (32.8GB → 5.84GB)
- [x] 마스크 이미지 저장 (렌더링에 사용)
- [x] `POST /internal/v1/render` 구현
- [x] 관절 `score` · 검출 `confidence` 응답, 여러 명(`MULTIPLE_CHARACTERS`) · 사람이 아닌 그림(`LOW_CONFIDENCE`) 걸러내기 — 아래 「점수와 걸러내기」

## 처리 가능한 그림

| 그림 종류 | 지원 |
|---|---|
| 선만 그린 그림 (색칠 없음) | ✅ |
| 색칠한 그림 | ✅ |
| 뭉뚱그린 살 형태 | ✅ |
| 얇은 선 | ✅ |
| 선이 일부 끊긴 그림 | ✅ |
| 종이에 그린 사진 (얼룩·음영) | ✅ |
| 배경이 캐릭터와 밝기가 비슷한 그림 | ⚠️ 실패할 수 있음 |

색칠은 필수가 아니다. 사람 형태로 보이면 대부분 처리된다.

> **관절 위치 정확도**: Meta 모델은 사람 사진으로 학습되어 4등신 캐릭터 그림에서
> 관절이 안쪽으로 몰리는 경향이 있다. 애니메이션에서 팔다리가 다소 휘어 보인다.
> 개선 실험 결과는 [docs/관절인식-실험-기록.md](docs/관절인식-실험-기록.md) 참고.
> 근본 해결은 A2의 손그림 전용 Pose 모델 교체로 기대한다.

## 점수와 걸러내기

analyze 응답의 `joints[].score`(0~1)는 Pose 모델이 그 관절을 얼마나 확신하는지다.
`hip` · `torso` 처럼 두 점의 중간점은 낮은 쪽 점수를 쓴다. `confidence`는 캐릭터 검출 점수다.
백엔드가 그대로 프론트에 넘기고, 프론트는 낮은 관절이 있으면 어른 확인(S-05 「어른에게 도움 받기」, S-06 표시)을 권한다.

| 실패 code | 언제 | 기준 |
|---|---|---|
| `LOW_CONFIDENCE` | 사람 모양으로 보기 어려운 그림 (낙서, 도형) | 검출 점수 < 0.3, 또는 관절 점수 평균 < 0.35 |
| `MULTIPLE_CHARACTERS` | 따로 떨어진 사람이 둘 이상 | 두 번째 검출 점수 ≥ 0.8 이고 첫 번째 상자와 50% 미만 겹침 |
| `NO_CHARACTER_DETECTED` | 너무 흐리거나 아무것도 못 찾음 | 검출 결과 없음 (기존) |

기준은 2026-10-02 실제 그림으로 잰 값에서 정했다 (CPU, Meta 예제 그림).

| 그림 | 검출 | 관절 평균 | 관절 최저 | 결과 |
|---|---|---|---|---|
| Meta 예제 char1·2·3·5 | 0.99 이상 | 0.81~0.91 | 0.65~0.88 | 통과 |
| char4 (잘라낸 그림) | 0.99 | 0.80 | 0.51 | 통과 |
| char4 원본 사진 (막대 사람, 옆에 라디오·조명) | 0.71 | 0.44 | 0.15 | 통과, 낮은 관절 8개 → 어른 확인 |
| char6 | 0.95 | 0.56 | 0.32 | 통과, 낮은 관절 4개 |
| 마늘 캐릭터 | 0.99 | 0.66 | 0.30 | 통과, 팔 2개 낮음 |
| char1 살짝 흐림 (blur 12) | 0.81 | 0.81 | 0.71 | 통과 |
| char1 · char5 많이 흐림 (blur 20~25) | 없음 | - | - | `NO_CHARACTER_DETECTED` |
| 동그라미 하나 | 0.997 | 0.23 | 0.10 | `LOW_CONFIDENCE` |
| 낙서 | 0.97 | 0.10 | 0.07 | `LOW_CONFIDENCE` |
| char1 · char2 나란히 | 0.998 / 0.99 | - | - | `MULTIPLE_CHARACTERS` |
| char5 · char6 나란히 | 0.99 (하나만 검출) | 0.81 | 0.66 | 통과 (두 번째를 못 찾음) |

- **검출 점수로는 사람인지 가리지 못한다.** 낙서와 동그라미도 0.97 이상이다. 그래서 관절 점수 평균을 쓴다.
- 처음에 두 번째 검출 기준을 0.5로 두었더니 char4 원본(한 명)이 라디오·조명(0.54) 때문에 여러 명으로 걸렸다. 0.8로 올렸다.
- 두 사람을 나란히 그려도 검출기가 하나만 찾으면 걸러지지 않는다. 그 경우 점수가 높은 한 명만 쓴다.

## 테스트 도구

`output/` 폴더에 애니메이션 생성·관절 확인 스크립트가 있다.
사용법은 [output/README.md](output/README.md) 참고.

```bash
python output/render.py <이미지경로> <motion>
```

## 남은 작업

- [ ] Peak RAM 측정 (렌더링 중)
- [ ] 임시 파일 자동 정리 (오래된 세션)
- [ ] 동시 요청 처리 정책 (현재는 순차 처리)
- [ ] 배경 밝기가 캐릭터와 비슷할 때 실패 처리 및 안내

## Git 규칙

- `dev` 브랜치에서 작업 → 팀장 확인 후 `dev → main` PR
- 가상환경, 모델 가중치(`*.pth`, `*.mar`), 결과 영상(`*.mp4`, `*.gif`), 데이터셋은 올리지 않음 (`.gitignore`)
- 백업 파일(`*_backup.py`)을 만들지 않고 Git 커밋으로 이력 관리
- `third_party/AnimatedDrawings`는 Meta 원본이므로 수정하지 않음
