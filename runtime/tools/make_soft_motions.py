"""Fix body deformations in walk / run / jumping_gentle (2026-11-03).

Why (checked with strips of several drawings, frontend worklog 2026-11-03):
- walk / run: the legs crossed and swapped in some frames. cmu1_human.yaml projects the lower limbs
  with PCA, and for a side-walking mocap the PCA plane flips between frames, so the drawn legs
  slid over each other. Projecting them frontally keeps each leg on its own side, but then the
  legs barely move (the walk happens toward the camera), so the knee bend is amplified to read
  as steps.
- jumping_gentle: on the crouch the spine bent forward and the knees folded, so the drawing's
  body and head squashed down to 83% of their height. The jump keeps its movement but every
  joint moves less from the first pose, and the root sinks less on the crouch (rises the same).

What this writes (new files only; the originals stay as they are):
- cmu1_front.yaml       cmu1_human.yaml with the lower limbs projected frontally
- walk_steps.bvh/.yaml  walk with LeftLeg / RightLeg (knee) rotations x2.0
- run_steps.bvh/.yaml   run with knee rotations x1.4 (x2 put the feet onto the body)
- jumping_soft.bvh/.yaml  jumping_gentle with spine/neck/head 0.3, legs 0.8, hips 0.6 of the
                          movement from the first frame, and the crouch depth at 0.7

Run inside the AI image (it has numpy + scipy), from the repo root:

    docker compose run --rm -v "$PWD:/repo" -w /repo ai python runtime/tools/make_soft_motions.py
"""

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
M = REPO / "runtime" / "ai_server" / "motions"

KNEES = ("LeftLeg", "RightLeg")
WALK_KNEE = 2.0
RUN_KNEE = 1.4

SPINE = ("Spine", "Spine1", "Spine2", "Spine3", "Neck", "Head")
LEGS = ("RightUpLeg", "RightLeg", "RightFoot", "LeftUpLeg", "LeftLeg", "LeftFoot")
JUMP_KEEP = {**{j: 0.3 for j in SPINE}, **{j: 0.8 for j in LEGS}, "Hips": 0.6}
JUMP_CROUCH = 0.7


def parse_channels(header: list[str]) -> list[tuple[str, list[str]]]:
    out, joint = [], None
    for line in header:
        w = line.split()
        if not w:
            continue
        if w[0] in ("ROOT", "JOINT"):
            joint = w[1]
        elif w[0] == "CHANNELS":
            out.append((joint, w[2:]))
    return out


def rewrite(src: Path, dst: Path, scales: dict[str, float], relative: bool, crouch: float = 1.0, up: str = "z") -> None:
    """Scale each listed joint's rotation (as a rotation vector).

    relative=False scales the rotation itself (toward the BVH rest pose); relative=True scales only
    the movement away from the first frame, so the starting pose stays as it is.
    crouch < 1 makes the root sink less below its first-frame height (rising is unchanged)."""

    lines = src.read_text().splitlines()
    mi = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    data = np.array([[float(v) for v in l.split()] for l in lines[mi + 3:] if l.strip()])
    idx, root_up = 0, None
    for joint, ch in parse_channels(lines[:mi]):
        if root_up is None:
            root_up = idx + [c[0].lower() for c in ch].index(up)
        rot = [c for c in ch if c.endswith("rotation")]
        if joint in scales and rot:
            first = idx + ch.index(rot[0])
            order = "".join(c[0].upper() for c in rot)
            r = Rotation.from_euler(order, data[:, first:first + 3], degrees=True)
            if relative:
                r0 = r[0]
                r = r0 * Rotation.from_rotvec((r0.inv() * r).as_rotvec() * scales[joint])
            else:
                r = Rotation.from_rotvec(r.as_rotvec() * scales[joint])
            data[:, first:first + 3] = r.as_euler(order, degrees=True)
        idx += len(ch)
    if crouch != 1.0:
        z0, z = data[0, root_up], data[:, root_up]
        data[:, root_up] = np.where(z < z0, z0 + (z - z0) * crouch, z)
    dst.write_text("\n".join(lines[:mi + 3] + [" ".join(f"{v:.5f}" for v in r) for r in data]) + "\n")


def copy_yaml(src: str, dst: str, note: str) -> None:
    text = (M / f"{src}.yaml").read_text(encoding="utf-8")
    text = text.replace(f"filepath: {src}.bvh", f"filepath: {dst}.bvh")
    (M / f"{dst}.yaml").write_text(f"# {note}\n" + text, encoding="utf-8")


def main() -> None:
    front = (M / "cmu1_human.yaml").read_text(encoding="utf-8")
    old = "  method: pca\n  name: Lower Limbs"
    assert old in front, "cmu1_human.yaml lower limbs are no longer PCA - check before regenerating"
    (M / "cmu1_front.yaml").write_text(
        "# cmu1_human.yaml 에서 다리만 정면 투영 (make_soft_motions.py, 2026-11-03 — PCA 는 걷기 다리가 꼬였다)\n"
        + front.replace(old, "  method: frontal\n  name: Lower Limbs"), encoding="utf-8")

    rewrite(M / "walk.bvh", M / "walk_steps.bvh", {j: WALK_KNEE for j in KNEES}, relative=False, up="y")
    copy_yaml("walk", "walk_steps", f"walk 무릎 굽힘 x{WALK_KNEE} (make_soft_motions.py) — cmu1_front.yaml 과 함께")
    rewrite(M / "run.bvh", M / "run_steps.bvh", {j: RUN_KNEE for j in KNEES}, relative=False, up="y")
    copy_yaml("run", "run_steps", f"run 무릎 굽힘 x{RUN_KNEE} (make_soft_motions.py) — cmu1_front.yaml 과 함께")
    rewrite(M / "jumping_gentle.bvh", M / "jumping_soft.bvh", JUMP_KEEP, relative=True, crouch=JUMP_CROUCH, up="z")
    copy_yaml("jumping_gentle", "jumping_soft", "jumping_gentle 의 웅크림을 줄인 것 (make_soft_motions.py) — 몸 · 머리 눌림")
    print("ok")


if __name__ == "__main__":
    main()
