# drawtale-ai

DrawTale AI Runtime — 손그림에서 캐릭터 · 관절을 찾고 애니메이션을 렌더링하는 **내부 전용 AI 서버**입니다.
Backend만 호출하며 외부(Frontend, 인터넷)에는 공개하지 않습니다.

- API 계약서: `drawtale-backend/docs/api-contract.md` 3장 (Backend ↔ AI)
- 담당: A1 (Runtime) · A2 (Pose 모델 학습)

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
| `POST /internal/v1/analyze` | ✅ 동작 — 파일 업로드(`file`) → bbox, 관절 15개, 마스크 크기, 메타데이터 |
| `POST /internal/v1/render` | ⬜ **미구현 — A1 다음 작업** |

Backend 연동 테스트 완료 (2026-09-28): 업로드 → 분석 → 관절 15개 반환, 에러 처리까지 확인.

**측정값** (20코어 PC, CPU only)

| 항목 | 값 |
|---|---|
| 분석 시간 | 약 3초 (2000x2000 기준) |
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

## A1 다음 작업

- [ ] **`POST /internal/v1/render`** — Backend가 보낸 **사용자 보정 관절**로 렌더링해서 MP4 생성 (가장 중요)
- [ ] **마스크 이미지 저장** — 지금은 크기만 반환하고 버림 (render에 필요)
- [ ] 렌더링 시간 · Peak RAM 측정 (`docker stats`)

### 완료된 항목 (팀장 수정)

- [x] 관절 · bbox 좌표를 원본 이미지 기준으로 변환 (1000px 축소본 기준 → 원본 기준)
- [x] 응답 메타데이터 추가: `model_version`, `pipeline_version`, `coordinate_space`, `processing_time_ms`
- [x] health의 `model_loaded`를 실제 TorchServe 상태로 표시
- [x] `async def` → `def` (분석 중 서버 전체가 멈추지 않게)
- [x] TorchServe 작업자 수 제한 (17GB → 1.5GB)
- [x] TorchServe 주소를 `TORCHSERVE_URL` 환경변수로 분리

## Git 규칙

- `dev` 브랜치에서 작업 → 팀장 확인 후 `dev → main` PR
- 가상환경, 모델 가중치(`*.pth`, `*.mar`), 결과 영상(`*.mp4`, `*.gif`), 데이터셋은 올리지 않음 (`.gitignore`)
- 백업 파일(`*_backup.py`)을 만들지 않고 Git 커밋으로 이력 관리
- `third_party/AnimatedDrawings`는 Meta 원본이므로 수정하지 않음
