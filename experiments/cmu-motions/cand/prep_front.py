"""11/3 에 고른 15개를 새 대응표(cmu1_front — 다리 정면 투영)용으로 (2026-11-04).

팀원 955f281: PCA 다리 투영은 걷기에서 프레임마다 뒤집혀 다리가 꼬였다. 정면 투영으로 바꾸면 걸음이
거의 안 보여 무릎 굽힘을 키웠다(걷기 ×2.0). 걷는 동작 셋에 같은 배율을 준다. 나머지는 그대로 복사.
입력 work/{이름}_soft.bvh → 출력 work/{이름}_front.bvh · .yaml
"""

from pathlib import Path

import yaml
from scipy.spatial.transform import Rotation

WORK = Path(__file__).parent / "work"
PICKED = ["idle", "stand_still", "very_happy", "laugh", "happy_walk", "point", "reach_tiptoe", "kick_ball",
          "march", "search_ground", "swim", "bird", "monkey", "elephant", "sad_walk"]
WALKING = {"happy_walk": 2.0, "march": 2.0, "sad_walk": 2.0}
KNEES = ("LeftLeg", "RightLeg")

for name in PICKED:
    lines = (WORK / f"{name}_soft.bvh").read_text(encoding="utf8").splitlines()
    i = next(n for n, l in enumerate(lines) if l.strip().startswith("Frames:"))
    head, motion = lines[:i + 2], [l for l in lines[i + 2:] if l.strip()]
    idx, pos, cur = {}, 0, None
    for l in head:
        t = l.strip()
        if t.startswith(("ROOT ", "JOINT ")):
            cur = t.split()[1]
        elif t.startswith("CHANNELS"):
            ch = t.split()[2:]
            idx[cur] = (pos, ch)
            pos += len(ch)
    k = WALKING.get(name)
    if k:
        out = []
        for l in motion:
            v = [float(x) for x in l.split()]
            for j in KNEES:
                start, ch = idx[j]
                seq = "".join(c[0] for c in ch)
                R = Rotation.from_euler(seq, v[start:start + 3], degrees=True)
                v[start:start + 3] = Rotation.from_rotvec(R.as_rotvec() * k).as_euler(seq, degrees=True).tolist()
            out.append(" ".join(f"{x:.6f}" for x in v))
        motion = out
    dst = WORK / f"{name}_front.bvh"
    dst.write_text("\n".join(head + motion) + "\n", encoding="utf8")
    cfg = yaml.safe_load((WORK / f"{name}_soft.yaml").read_text())
    cfg["filepath"] = str(dst)
    (WORK / f"{name}_front.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(name, f"무릎 ×{k}" if k else "")
