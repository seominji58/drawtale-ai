# 동작 데이터 출처

| | |
|---|---|
| 원본 | **CMU Graphics Lab Motion Capture Database** (`mocap.cs.cmu.edu`) |
| BVH 변환 | Bruce Hahne (cgspeed). 받은 곳은 [`una-dinosauria/cmu-mocap`](https://github.com/una-dinosauria/cmu-mocap) |
| 라이선스 | CMU 는 원본에 제한을 두지 않고, 변환자도 추가 제한을 두지 않는다고 밝혔습니다 |

**배포 전에 현재 문구를 다시 확인하고 출처를 표기합니다** (설계 문서 L-10).

## 여기 있는 파일

| 파일 | 원본 | 손본 것 |
|---|---|---|
| `walk.bvh` | `data/007/07_01.bvh` | 프레임 솎기(120→30fps) · 팔 순화(30% · 아래 25도) · 수평 이동 30% |
| `run.bvh` | `data/009/09_01.bvh` | 같음 |

다시 만들려면 같은 폴더의 `README.md` 2 · 3번을 따르면 됩니다.

### 이야기 쪽 동작 (2026-11-04)

모두 프레임 솎기(120→30fps) · 반복 구간 2~4초 · 팔 순화(30% · 아래 25도) · **제자리(수평 이동 0)**.
행진은 무릎 굽힘 ×2.0 (`walk_steps` 와 같은 이유). 만드는 법은 `runtime/tools/install_more_motions.py`.

| 파일 | 원본 | 무엇 |
|---|---|---|
| `breathe.bvh` | `data/140/140_06.bvh` | 제자리 숨쉬기 |
| `march.bvh` | `data/091/91_19.bvh` | 행진 |
| `point.bvh` | `data/013/13_27.bvh` | 가리키기 |
| `kick.bvh` | `data/010/10_01.bvh` | 공 차기 |
| `search.bvh` | `data/077/77_06.bvh` | 땅을 살피며 찾기 |
| `swim.bvh` | `data/079/79_02.bvh` | 헤엄치기 |
| `fly.bvh` | `data/079/79_58.bvh` | 새처럼 날기 |

`hooray` · `shiver` 는 CMU 동작이 아니다. `breathe` 위에 팔 · 머리 각도를 계산해 덮었다
(drawtale-ai `cmu_test_render` 브랜치의 `experiments/cmu-motions/cand/make_handmade.py`).
