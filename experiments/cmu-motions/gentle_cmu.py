# -*- coding: utf-8 -*-
"""CMU BVH 의 팔을 아이 그림용으로 다듬는다 (시험용).

서비스 코드가 아니다. 쓰는 방법은 같은 폴더의 README.md 에 있다.

Meta fair1 용 도구(make_gentle_motions.py)는 「기본 자세로 당기기」인데,
CMU 는 기본 자세에서 팔이 아래로 내려와 있어 그대로 쓰면 팔이 몸에 더 붙는다.
그래서 여기서는 (1) 팔의 움직임을 줄이고 (2) 어깨를 바깥쪽으로 일정 각도 돌려 둔다.

    python gentle_cmu.py <입력.bvh> <출력.bvh> <남길비율> <바깥각도> <이동비율>

이동비율: 뿌리 관절의 앞뒤·좌우 이동을 얼마나 남길지. 0 이면 제자리, 0.3 이면 화면 안에서 조금 옮겨 간다.
위아래 이동은 그대로 둔다 (점프와 들썩임이 사라지면 안 된다).
"""
import io, sys
import numpy as np
from scipy.spatial.transform import Rotation

src, dst, keep, out_deg = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
travel = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

text = io.open(src, encoding="utf8").read().splitlines()
i_frames = next(n for n, l in enumerate(text) if l.strip().startswith("Frames:"))
header, motion = text[:i_frames + 2], text[i_frames + 2:]

# 뼈 이름과 채널 순서를 머리말에서 읽는다
names, chans, cur = [], [], None
for l in header:
    t = l.strip()
    if t.startswith(("ROOT ", "JOINT ")):
        cur = t.split()[1]
    elif t.startswith("CHANNELS"):
        p = t.split()
        names.append(cur)
        chans.append(p[2:])

idx, pos = {}, 0
for n, c in zip(names, chans):
    idx[n] = (pos, c)
    pos += len(c)

ARMS = {"LeftArm": 1, "RightArm": -1, "LeftForeArm": 1, "RightForeArm": -1}
ELBOW_KEEP = 0.2

out = []
for line in motion:
    if not line.strip():
        continue
    v = [float(x) for x in line.split()]
    # 뿌리 관절의 수평 이동을 줄인다 (화면 밖으로 달려 나가지 않게). 위아래는 그대로
    root_start, root_order = idx[names[0]]
    for k, ch in enumerate(root_order):
        if ch in ("Xposition", "Zposition"):
            v[root_start + k] *= travel
    for joint, side in ARMS.items():
        if joint not in idx:
            continue
        start, order = idx[joint]
        rot = [c for c in order if c.endswith("rotation")]
        off = start + len(order) - len(rot)
        ang = [v[off + k] for k in range(len(rot))]
        seq = "".join(c[0].upper() for c in rot)
        R = Rotation.from_euler(seq, ang, degrees=True)
        k = ELBOW_KEEP if "ForeArm" in joint else keep
        R = Rotation.from_rotvec(R.as_rotvec() * k)          # 움직임을 줄인다
        if "ForeArm" not in joint and out_deg:
            # 어깨를 바깥쪽으로 — 팔이 몸통에서 떨어지게
            R = Rotation.from_euler("z", side * out_deg, degrees=True) * R
        new = R.as_euler(seq, degrees=True)
        for k2 in range(len(rot)):
            v[off + k2] = float(new[k2])
    out.append(" ".join(f"{x:.6f}" for x in v))

io.open(dst, "w", encoding="utf8").write("\n".join(header + out) + "\n")
print(f"{dst}: 팔 {keep:.0%} 유지 · 바깥 {out_deg:.0f}도 · 이동 {travel:.0%}")
