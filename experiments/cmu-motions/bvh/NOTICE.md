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
