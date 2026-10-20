# 네 발 걷기 시험

아이가 그린 **네 발 동물**에 걷는 동작을 입혀 보는 실험입니다.
결과와 판단은 설계 문서 [14 확장 설계](https://github.com/seominji58/drawtale-pt_docs/blob/main/14-확장-설계.md) 12장에 있습니다.

> **서비스 코드가 아닙니다.** `runtime/` 의 AI 서버는 사람 관절 15개만 받습니다.
> 이 실험은 그 API 를 건너뛰고 **Meta 렌더 엔진을 직접** 부릅니다.
> 네 발을 서비스에 넣기로 하면 그때 `runtime/ai_server/renderer.py` 를 고칩니다.

## 왜 관절을 손으로 찍나

관절을 찾는 모델이 **사람 전용**입니다. 동물 관절은 찍어 주지 않습니다.
Meta 가 네 발 골격(16관절)과 동작 입히는 설정은 주지만, 관절 위치는 사람이 넣어야 합니다.
이 실험의 목적은 **자동화가 아니라 움직임 품질을 보는 것**이라 그렇게 합니다.

## 준비물

Docker 와 이 저장소. **모델 서버(TorchServe)는 필요 없습니다** — 관절을 찍지 않으니까요.
AI 런타임 이미지만 있으면 됩니다.

```powershell
# 저장소 뿌리에서. 몇 분 걸립니다 (1.7GB)
docker compose build ai
```

이름이 `drawtale-ai-ai:latest` 인 이미지가 생깁니다. `docker images` 로 확인할 수 있습니다.
다른 이름이면 아래 명령의 이미지 이름을 바꿔 주세요.

> **동작 데이터를 더 쓰고 싶으면** 옆 폴더 [`../cmu-motions/`](../cmu-motions/README.md) 를 보세요.
> CMU 에서 걷기 · 뛰기를 받아 다듬는 방법이 있습니다 (2026-10-20 에 확인).

## 1. 그림 준비

- **정측면.** 네 다리가 다 보이게
- 얼굴은 정면이어도 됩니다
- **흰 배경** (마스크를 밝기로 가릅니다)
- 좌우 어느 쪽을 보든 상관없습니다 — 엔진이 관절 좌표로 앞쪽을 계산합니다

`samples/` 에 시험해 본 그림 두 장이 있습니다.

## 2. 관절 16개 찍기

`pick_joints.html` 을 브라우저로 엽니다. 설치할 것이 없습니다.

1. 그림 파일을 고릅니다
2. 화면이 알려 주는 순서대로 **15번 클릭**합니다
3. 아래에 나온 좌표 문자열을 복사합니다

순서는 `hip → torso → neck → 뒷다리 둘 → 앞다리 둘` 입니다.
뒷다리는 꼬리 쪽, 앞다리는 머리 쪽입니다. 다리마다 **위 · 중간 · 발끝** 셋을 찍습니다.

### 두 가지만 지키면 됩니다

골격은 16개인데 **`root` 는 찍지 않습니다.** 엉덩이와 같은 자리에 두는 것이 규칙이라
`make_char.py` 가 자동으로 넣습니다. 서비스 코드도 사람 관절에 같은 방식으로 붙입니다.

그리고 **`torso` 를 몸통 한가운데 찍지 마세요. 목 바로 밑입니다.**

처음 시험할 때 이 둘을 모르고 `root` 를 몸통 중앙에, `torso` 를 등 가운데에 찍었더니
걸을 때 **몸통이 늘어나고 머리가 몸통 안으로 말려 들어갔습니다.**
없던 뼈가 생기고 목 뼈가 길어져서 그렇습니다. 같은 그림으로 다시 찍으니 사라졌습니다.

## 3. 캐릭터 설정 만들기

```bash
cd experiments/quadruped
docker run --rm -v "$PWD:/w" -w /w drawtale-ai-ai:latest \
  python make_char.py samples/dog.png dog "<복사한 좌표 문자열>"
```

`dog/` 폴더에 `texture.png` · `mask.png` · `char_cfg.yaml` 이 생깁니다.
**마스크가 제대로 나왔는지 먼저 보세요** — 여기가 틀리면 결과가 전부 깨집니다.

```powershell
docker run --rm -v "${PWD}:/w" -w /w drawtale-ai-ai:latest python check.py dog
```

`out/check_dog.png` 가 나옵니다 — 마스크는 빨강, 관절은 노란 점, 뼈대는 파란 선입니다.

## 4. 걷게 하기

```bash
sed "s/__NAME__/dog/g" mvc_template.yaml > mvc_dog.yaml
docker run --rm -v "$PWD:/w" -w /opt/AnimatedDrawings drawtale-ai-ai:latest \
  python -c "import animated_drawings.render as r; r.start('/w/mvc_dog.yaml')"
```

`out/dog_walk.gif` 가 나옵니다. 20코어 PC 에서 **25초**쯤 걸렸습니다.

## 시험해 본 좌표

다시 찍기 번거로우면 그대로 쓰세요. `samples/` 의 그림 기준입니다.

```
dog.png
330,450 660,350 670,270 380,440 378,515 376,578 225,430 223,490 221,540 800,430 800,490 800,535 625,440 623,520 621,590

gong.png
400,570 700,500 715,450 460,580 462,670 465,775 310,520 305,650 302,790 765,600 775,700 790,820 650,630 650,740 650,865
```

> `dog.png` 와 `gong.png` 는 원본을 좌우로 뒤집은 것입니다.
> 뒤집을 필요는 없는데(위 1번 참고) 처음에 Meta 예제를 따라 하면서 그렇게 했고,
> 위 좌표가 그 그림 기준이라 그대로 두었습니다.

## 내 그림으로 해 보려면

`samples/` 대신 자기 그림을 쓰면 됩니다. 그림을 이 폴더에 넣고 2번부터 다시 하세요.
이름(`dog` 자리)만 바꾸면 폴더와 결과 파일이 그 이름으로 생깁니다.

조건은 1번과 같습니다 — **정측면, 네 다리가 보이게, 흰 배경.**
사람이나 두 발 캐릭터는 이 골격에 맞지 않습니다. 그건 서비스 쪽(사람 15관절)이 따로 합니다.

## 알아 둘 것

- **쓰는 동작이 제자리 걸음이 아닙니다.** 캐릭터가 화면 밖으로 걸어 나갑니다.
  앞부분 몇 초만 보면 됩니다. Mixamo 에서 받을 때 `In Place` 를 켜야 하는 이유입니다
- 꼬리는 **움직이지 않습니다.** 골격에 꼬리가 없습니다
- `point not inside any triangle in mesh` 경고는 관절이 그림 가장자리에 걸렸다는 뜻입니다.
  결과에는 큰 영향이 없었습니다
- 기준선이 필요하면 Meta 예제를 그대로 돌려 보세요.
  거기서 실패하면 그림 문제가 아니라 **환경 문제**입니다

```bash
docker run --rm -v "$PWD:/w" -w /opt/AnimatedDrawings drawtale-ai-ai:latest \
  python -c "import animated_drawings.render as r; r.start('/w/mvc_baseline.yaml')"
```
