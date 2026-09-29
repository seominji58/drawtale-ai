# output

테스트로 만든 애니메이션 MP4와 확인용 이미지를 두는 곳입니다.
**결과물은 Git에 올라가지 않고, 아래 실행 스크립트만 올라갑니다.**

## 준비

AI 서버가 떠 있어야 합니다.

```bash
cd drawtale-ai
docker compose up -d
```

`docker compose ps`에서 `torchserve`가 `healthy`가 되면 준비 완료입니다.

Python은 `drawtale-backend`의 가상환경을 그대로 쓰면 됩니다.
(`Pillow`, `numpy`, `scipy`가 필요합니다.)

## 애니메이션 만들기

```bash
python output/render.py <이미지경로> <motion>
```

예시:

```bash
python output/render.py output/source.png wave_hello
```

- `motion`: `wave_hello`, `jumping`, `jumping_jacks`, `dab`, `zombie`
- 결과: `output/<motion>.mp4`
- 분석 1~2초 + 렌더링 20~40초

관절을 직접 지정해서 렌더링하려면 세 번째 인자로 관절 JSON을 줍니다.

```bash
python output/render.py output/source.png wave_hello output/joints_fitted.json
```

## 관절 확인하기

분석한 관절을 그림 위에 그려서 눈으로 확인합니다.

```bash
python output/draw_joints.py output/source.png output/joints.json output/before.png
```

- 파란 점: 관절, 빨간 점: 손·발, 파란 선: 뼈대
- `output/joints.json`은 `render.py`를 실행하면 자동으로 저장됩니다.

## 관절 자동 보정 실험

마스크 모양을 보고 손·발 관절을 팔다리 끝으로 옮깁니다.

```bash
python output/fit_joints.py output/joints.json output/mask.png output/joints_fitted.json
```

> **실험 결과 (2026-09-29):** 관절이 20~30px 이동하지만
> 애니메이션 결과에는 눈에 띄는 차이가 없었습니다.
> 자세한 내용은 `docs/관절인식-실험-기록.md` 참고.
