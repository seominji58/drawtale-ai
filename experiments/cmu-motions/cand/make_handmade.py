"""CMU 에 없는 동작 넷을 관절 각도로 만든다 (시험용, 2026-11-04).

바탕은 제자리 숨쉬기(work/idle_loop.bvh, 2초) — 몸통 · 다리의 작은 움직임은 그대로 두고 팔 · 머리만 덮어쓴다.
모두 정면(Z축 회전)에서만 움직인다: 평면 그림에서 보이는 축이라서 (앞뒤로 숙이는 끄덕이기는 보이지 않아 뺐다).
2초에 정수 번 돌게 만들어 반복 이음새가 없다.

  hooray        만세 — 두 팔을 아래에서 위로 올렸다 내림 (2초에 한 번), 올릴 때 몸이 살짝 뜸
  shiver        떨기 — 팔을 조금 벌린 채 몸과 팔이 작게 빠르게 떨림 (초당 6번)
  clap_overhead 머리 위 손뼉 — 팔을 위로 올린 채 모았다 벌림 (2초에 네 번). 가슴 앞 손뼉은 팔이 몸통에 겹쳐 뺐다
  head_tilt     고개 갸웃 — 고개를 좌우로 기울임 (2초에 한 번)

    docker run --rm -v "${PWD}:/repo" -w /repo/experiments/cmu-motions/cand drawtale-ai-ai:latest python make_handmade.py
"""

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = Path(__file__).parent
WORK = HERE / "work"
BASE = WORK / "idle_loop.bvh"
FRAMES = 60  # 30fps × 2초

text = BASE.read_text(encoding="utf8").splitlines()
i_frames = next(n for n, l in enumerate(text) if l.strip().startswith("Frames:"))
header = text[:i_frames]
frame_time = text[i_frames + 1]
base = [[float(x) for x in l.split()] for l in text[i_frames + 2:] if l.strip()]

# 관절 이름 → (채널 시작 위치, 채널 이름들)
idx, pos, cur = {}, 0, None
for l in header:
    t = l.strip()
    if t.startswith(("ROOT ", "JOINT ")):
        cur = t.split()[1]
    elif t.startswith("CHANNELS"):
        ch = t.split()[2:]
        idx[cur] = (pos, ch)
        pos += len(ch)


def rot_of(v, joint) -> Rotation:
    start, ch = idx[joint]
    rot = [c for c in ch if c.endswith("rotation")]
    off = start + len(ch) - len(rot)
    return Rotation.from_euler("".join(c[0] for c in rot), [v[off + k] for k in range(len(rot))], degrees=True)


def set_rot(v, joint, R: Rotation) -> None:
    start, ch = idx[joint]
    rot = [c for c in ch if c.endswith("rotation")]
    off = start + len(ch) - len(rot)
    for k, a in enumerate(R.as_euler("".join(c[0] for c in rot), degrees=True)):
        v[off + k] = float(a)


def z(deg: float) -> Rotation:
    return Rotation.from_euler("z", deg, degrees=True)


def arms(v, left_deg: float, fore_deg: float = 0.0) -> None:
    """팔을 정면에서 수평 기준 left_deg 도로 (양수 = 위). 오른팔은 좌우 대칭. 팔꿈치는 fore_deg 만큼 안쪽으로."""
    set_rot(v, "LeftArm", z(left_deg))
    set_rot(v, "RightArm", z(-left_deg))
    set_rot(v, "LeftForeArm", z(fore_deg))
    set_rot(v, "RightForeArm", z(-fore_deg))


def root_add(v, axis: str, d: float) -> None:
    start, ch = idx["Hips"]
    v[start + ch.index(f"{axis}position")] += d


def make(name: str, fn) -> None:
    out = []
    for f in range(FRAMES):
        v = list(base[f % len(base)])
        fn(v, f / FRAMES)  # 0 ≤ t < 1, 2초 한 바퀴
        out.append(" ".join(f"{x:.6f}" for x in v))
    path = WORK / f"{name}_hand.bvh"
    path.write_text("\n".join(header + [f"Frames: {FRAMES}", frame_time] + out) + "\n", encoding="utf8")
    cfg = {
        "filepath": str(path), "start_frame_idx": 0, "end_frame_idx": FRAMES - 1,
        "groundplane_joint": "LeftFoot",
        "forward_perp_joint_vectors": [["LeftShoulder", "RightShoulder"], ["LeftUpLeg", "RightUpLeg"]],
        "scale": 0.025, "up": "+y",
    }
    (WORK / f"{name}_hand.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(name)


def hooray(v, t):
    s = 0.5 - 0.5 * np.cos(2 * np.pi * t)        # 0 → 1 → 0
    arms(v, -55 + 135 * s)                        # 아래 55도 → 위 80도
    root_add(v, "Y", 0.6 * s)


def shiver(v, t):
    w = np.sin(2 * np.pi * 12 * t)                # 2초에 12번 = 초당 6번
    arms(v, -40 + 4 * w)
    root_add(v, "X", 0.15 * w)
    set_rot(v, "Neck1", z(3 * w) * rot_of(v, "Neck1"))


def clap_overhead(v, t):
    s = 0.5 - 0.5 * np.cos(2 * np.pi * 4 * t)     # 2초에 네 번 모음
    arms(v, 68 + 18 * s, fore_deg=15 + 15 * s)


def head_tilt(v, t):
    set_rot(v, "Neck1", z(16 * np.sin(2 * np.pi * t)) * rot_of(v, "Neck1"))


for name, fn in [("hooray", hooray), ("shiver", shiver), ("clap_overhead", clap_overhead), ("head_tilt", head_tilt)]:
    make(name, fn)
