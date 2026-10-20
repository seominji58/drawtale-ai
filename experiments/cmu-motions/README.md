# CMU 동작 데이터 쓰기

Meta 가 주는 동작은 6종뿐이고 **걷기와 뛰기가 없습니다.**
Mixamo 는 BVH 를 주지 않아서(FBX · Collada 뿐) **CMU Mocap** 을 받아 씁니다.

2026-10-20 에 걷기와 뛰기를 **사람과 네 발 양쪽에서** 확인했습니다.
결과와 영상은 설계 문서 [14 확장 설계](https://github.com/seominji58/drawtale-pt_docs/blob/main/14-확장-설계.md) 14 · 15장에 있습니다.

> **아직 서비스에 들어가지 않았습니다.** `runtime/ai_server/renderer.py` 의 `MOTIONS` 에는
> 여전히 Meta 동작만 등록돼 있습니다. 여기는 **받아서 다듬는 방법**을 남겨 둔 곳입니다.

## 1. 쓰는 동작은 이미 들어 있습니다

`bvh/walk.bvh` · `bvh/run.bvh` 가 **다듬어 놓은 것**입니다 (합쳐 116KB).
받을 것도 돌릴 것도 없이 바로 5번으로 가면 됩니다.
출처와 손본 내용은 [`bvh/NOTICE.md`](bvh/NOTICE.md) 에 있습니다.

**다른 동작을 더하고 싶을 때만** 아래를 따릅니다.

[`una-dinosauria/cmu-mocap`](https://github.com/una-dinosauria/cmu-mocap) 에 BVH 가 그대로 있습니다.
전체를 받을 필요 없이 파일 하나씩 받으면 됩니다.

```powershell
# PowerShell 의 curl 은 다른 명령이라 curl.exe 로 불러야 합니다
curl.exe -L -o bvh/jump.bvh https://raw.githubusercontent.com/una-dinosauria/cmu-mocap/master/data/009/09_01.bvh
```

어느 번호에 무엇이 있는지는 그 저장소의 색인 파일(`cmu-mocap-index-*`)에서 찾습니다.
09 번이 달리기, 07 · 08 번이 걷기입니다. **대기와 좌절은 아직 못 찾았습니다.**

> **같은 CMU 라도 변환본마다 다릅니다.** 뼈 이름도 기본 자세도 다릅니다.
> 다른 곳에서 받으면 아래 2 · 3번 설정이 맞지 않을 수 있으니 먼저 확인하세요.
> (처음에 `cmuconvert-daz` 판을 받았다가 `hip` · `rThigh` 같은 이름과 없는 관절 때문에 쓰지 못했습니다.)

## 2. 프레임 솎기 — 안 하면 4배 느립니다

CMU 는 **초당 120프레임**으로 기록돼 있는데 저희 렌더는 30프레임쯤으로 뽑습니다.

```powershell
docker run --rm -v "${PWD}:/w" -w /w drawtale-ai-ai:latest `
  python resample.py bvh/run.bvh bvh/run30.bvh 4
```

## 3. 팔 순화 — 사람 그림에만 필요합니다

아이 그림은 팔을 몸통에 붙여 그리는 경우가 많아, 팔을 크게 휘두르면 **몸통에 눌려 뭉개집니다.**
`runtime/tools/make_gentle_motions.py` 가 Meta 동작에 하는 일과 같은데,
그 도구는 **기본 자세로 당기는** 방식이라 CMU 에는 쓸 수 없습니다
(받은 CMU 변환본은 기본 자세가 **T 자**라, 당기면 팔이 수평으로 올라가 만세하며 달립니다).

```powershell
# 팔 움직임 30% 유지 · 어깨를 아래로 25도 · 수평 이동 30% 유지
docker run --rm -v "${PWD}:/w" -w /w drawtale-ai-ai:latest `
  python gentle_cmu.py bvh/run30.bvh bvh/run_gentle.bvh 0.3 -25 0.3
```

| 값 | 뜻 |
|---|---|
| `0.3` | 팔 움직임을 30% 만 남깁니다 |
| `-25` | 어깨를 **아래로** 25도. 양수면 위로 올라가 만세가 됩니다 |
| `0.3` | 수평 이동을 30% 만 남깁니다. 0 이면 제자리, 1 이면 화면 밖으로 달려 나갑니다 |

**25도는 시험한 그림에 맞춘 값입니다.** 0 · 25 · 40 · 55 를 돌려 보고 골랐습니다 —
0 은 만세, 55 는 다시 몸에 붙습니다. 팔을 더 벌려 그린 그림이면 덜 내려야 합니다.
나중에는 그림의 팔 각도를 재서 맞출 수 있습니다(어깨와 손목 좌표로 계산되고, AI 가 이미 줍니다).

**네 발은 순화가 필요 없습니다.** 네 발 대응표가 사람 팔을 아예 쓰지 않기 때문입니다.
다만 수평 이동은 네 발에도 줄여야 합니다.

## 4. 대응표

Meta 가 주는 대응표는 **자기 BVH 이름 기준**이라 그대로 맞지 않습니다. 둘을 만들어 두었습니다.

| 파일 | 쓰는 곳 | 무엇을 고쳤나 |
|---|---|---|
| `retarget/cmu1_human.yaml` | **사람** | Meta `cmu1_pfp` 의 뼈 이름을 받은 변환본 이름으로 바꿈 |
| `retarget/four_legs_cmu.yaml` | **네 발** | Meta `four_legs` 에서 **없는 뼈 네 줄 제거** (`RightHandEnd` · `LeftHandEnd` · `Spine2` · `Spine3`) |

## 5. 돌려 보기

### 네 발 그림

관절을 사람이 찍어야 합니다. 만드는 방법은 [`../quadruped/README.md`](../quadruped/README.md) 에 있고,
캐릭터 폴더를 만든 뒤 **CMU 용 템플릿**을 쓰면 됩니다.

```powershell
cd ..\quadruped
(Get-Content mvc_template_cmu.yaml) -replace '__NAME__','dog' -replace '__MOTION__','walk' | Set-Content mvc_dog.yaml
docker run --rm -v "${PWD}:/w" -v "${PWD}\..\cmu-motions:/cmu" -w /opt/AnimatedDrawings drawtale-ai-ai:latest `
  python -c "import animated_drawings.render as r; r.start('/w/mvc_dog.yaml')"
```

폴더 둘을 물립니다 — 캐릭터는 `/w`, 동작과 대응표는 `/cmu` 입니다.

### 사람 그림

**AI 가 관절을 찾아 주므로 손으로 찍지 않습니다.** 대신 **모델 서버가 떠 있어야** 합니다.

```powershell
# 저장소 뿌리에서. 처음 한 번은 오래 걸립니다 (모델 내려받기)
docker compose up -d ai
```

그 다음 분석을 돌립니다. 작업 폴더가 **그 컨테이너 안에** 남으므로, 렌더도 같은 컨테이너에서 해야 합니다.

```powershell
# 1) 분석 — request_id 와 관절 15개가 돌아옵니다
docker run --rm --network drawtale-ai_default -v "${PWD}:/d:ro" --entrypoint sh curlimages/curl:latest `
  -c "cp /d/그림.png /tmp/x.png && curl -sS -X POST -F 'file=@/tmp/x.png' http://ai:8001/internal/v1/analyze" > analyze.json

# 2) 이 폴더를 컨테이너에 넣고 렌더
docker cp . drawtale-ai-ai-1:/w
docker exec drawtale-ai-ai-1 python /w/render_human.py <request_id> walk /w/analyze.json
docker cp drawtale-ai-ai-1:/w/out/human_walk.gif .
```

`render_human.py` 는 서비스 코드(`ai_server.renderer.prepare_character`)를 그대로 부릅니다.
그래서 여기서 나온 결과가 **실제 서비스에서 나올 결과와 같습니다.**

**AI 가 관절을 어디에 찍었는지** 보려면 네 발 폴더의 `overlay.py` 를 쓰면 됩니다.

```powershell
docker run --rm -v "${PWD}:/w" -w /w drawtale-ai-ai:latest `
  python ../quadruped/overlay.py 그림.png analyze.json out/joints.png
```

## 서비스에 넣을 때

| 할 일 | |
|---|---|
| `gentle_cmu.py` 를 `runtime/tools/` 로 | 기존 `make_gentle_motions.py` 옆에 |
| 다듬은 BVH 를 `runtime/ai_server/motions/` 로 | Meta 것과 같은 자리 |
| `MOTIONS` 에 등록 | 대응표 경로도 같이 |
| 행동 카드 연결표(`ACTION_MOTIONS`) 고치기 | Backend. 설계는 [03 5.4](https://github.com/seominji58/drawtale-pt_docs/blob/main/03-feature-spec.md) |

**대기와 좌절까지 받은 뒤에 한 번에 하는 편**이 낫습니다. 그때 행동 카드 여섯 장이 한꺼번에 달라집니다.

## 남은 문제

- **클립이 짧습니다.** 받은 달리기가 1.2초뿐이라 이어 붙여 반복해야 합니다
- **대기 · 좌절을 아직 못 찾았습니다**
- CMU 라이선스는 「연구와 상업 목적으로 자유롭게」로 안내돼 있지만
  **쓰기 전에 현재 문구를 다시 확인하고 출처를 표기**합니다 (설계 문서 L-10)
