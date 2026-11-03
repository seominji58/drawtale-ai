"""후보 동작을 한꺼번에 다듬는다 (시험용, 2026-11-03).

list.txt 의 BVH 마다: 30fps 로 솎기 → 반복해도 덜 튀는 2~4초 구간 고르기 → 제자리 고정 →
팔 순화 두 가지(soft: 30% · 아래 25도 = 지금 걷기와 같음 / live: 70% · 아래 15도).
결과: work/{이름}_{soft|live}.bvh · .yaml, work/windows.txt

    docker run --rm -v "${PWD}:/repo" -w /repo/experiments/cmu-motions/cand drawtale-ai-ai:latest python prep.py
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/opt/AnimatedDrawings")

import numpy as np
import yaml
from animated_drawings.model.bvh import BVH

HERE = Path(__file__).parent
WORK = HERE / "work"
WORK.mkdir(exist_ok=True)
VARIANTS = {"soft": (0.3, -25), "live": (0.7, -15)}
SHORTEST, LONGEST = 2.0, 4.0


def positions(bvh: BVH) -> np.ndarray:
    names = bvh.get_joint_names()
    joints = [bvh.root_joint.get_transform_by_name(n) for n in names]
    hips = bvh.root_joint.get_transform_by_name("Hips")
    out = np.zeros((len(bvh.pos_data), len(joints), 3), dtype=np.float32)
    for f in range(len(bvh.pos_data)):
        bvh.apply_frame(f)
        bvh.root_joint.update_transforms()
        o = hips.get_world_position()
        for i, j in enumerate(joints):
            out[f, i] = j.get_world_position() - o
    return out


def pick(path: Path) -> tuple[int, int, float, float]:
    """움직임 상위 30% 구간 중 이음새가 가장 작은 (시작, 끝, 이음새, 움직임)."""
    bvh = BVH.from_file(str(path))
    pos = positions(bvh)
    n = len(pos)
    moved = np.linalg.norm(np.diff(pos, axis=0), axis=2).sum(axis=1)
    cum = np.concatenate([[0.0], np.cumsum(moved)])
    lo, hi = int(SHORTEST / bvh.frame_time), int(LONGEST / bvh.frame_time)
    rows = []
    for s in range(0, max(1, n - lo)):
        for e in range(s + lo, min(s + hi, n), 2):
            seam = float(np.linalg.norm(pos[e] - pos[s], axis=1).mean())
            rows.append((seam, float((cum[e] - cum[s]) / (e - s)), s, e))
    th = np.percentile([r[1] for r in rows], 70)
    seam, move, s, e = min((r for r in rows if r[1] >= th), key=lambda r: r[0])
    return s, e, seam, move


def cut(src: Path, dst: Path, s: int, e: int) -> None:
    lines = src.read_text(encoding="utf8").splitlines()
    i = next(k for k, l in enumerate(lines) if l.strip().startswith("Frames:"))
    frames = lines[i + 2:][s:e + 1]
    dst.write_text("\n".join(lines[:i] + [f"Frames: {len(frames)}", lines[i + 1]] + frames) + "\n", encoding="utf8")


report = []
for row in (HERE / "list.txt").read_text().split("\n"):
    if not row.strip():
        continue
    name = row.split()[0]
    r30 = WORK / f"{name}_30.bvh"
    subprocess.run([sys.executable, str(HERE.parent / "resample.py"), str(HERE / f"{name}.bvh"), str(r30), "4"],
                   check=True, capture_output=True)
    s, e, seam, move = pick(r30)
    loop = WORK / f"{name}_loop.bvh"
    cut(r30, loop, s, e)
    for v, (keep, deg) in VARIANTS.items():
        out = WORK / f"{name}_{v}.bvh"
        subprocess.run([sys.executable, str(HERE.parent / "gentle_cmu.py"), str(loop), str(out), str(keep), str(deg), "0"],
                       check=True, capture_output=True)
        cfg = {
            "filepath": str(out), "start_frame_idx": 0, "end_frame_idx": e - s,
            "groundplane_joint": "LeftFoot",
            "forward_perp_joint_vectors": [["LeftShoulder", "RightShoulder"], ["LeftUpLeg", "RightUpLeg"]],
            "scale": 0.025, "up": "+y",
        }
        (WORK / f"{name}_{v}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    line = f"{name:15s} {s / 30:5.1f}~{e / 30:5.1f}s ({(e - s) / 30:.1f}s) seam {seam:5.2f} move {move:5.2f}"
    print(line, flush=True)
    report.append(line)
(WORK / "windows.txt").write_text("\n".join(report) + "\n", encoding="utf8")
