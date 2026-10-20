"""Make "gentle" versions of Meta's BVH motions for children's drawings.

Why: children often draw arms fused to the torso (no gap under the arm). When a motion brings
the arm down toward the body or bends the elbow, the mesh around the armpit is squashed onto
the torso and the arm collapses into a blob. With the arm held out, the same drawing looks fine.
(Frontend worklog 2026-09-29, "MP4 에서 팔 모양이 무너지는 원인")

What: (1) every frame's arm rotations are pulled toward the BVH rest pose. The fair1 skeleton's rest
pose is a T-pose (arms horizontal), so pulling toward it keeps the arms away from the body.
The forearm is pulled much harder so the elbow stays nearly straight. Rotations are scaled as
rotation vectors (axis-angle), not per Euler component. (2) The root's horizontal travel is
frozen so the character stays in place and outstretched hands do not leave the frame.

(3) the motion is cut to the one loop the scene plays and thinned from 120 to 30 fps. The source
BVHs are 7.0s and 5.5s long but hold only ~0.5s and ~1.7s of actual movement; the rest is standing
still. Rendering cost is linear in frame count (fixed 5s + 0.03s per frame, measured 2026-10-20 —
see experiments/render-cost/), so drawing all 839 frames cost 30-47s per video. Thinning also fixes
playback speed: Pillow floors a .gif frame to 20ms, so a 120fps BVH played back 2.4x slow.

Only fair1 motions qualify. The cmu1 skeleton (jumping_jacks) rests with the arms DOWN, so the
same pull would press the arms onto the body — the opposite of what we want.

Run inside the AI image (it has numpy + scipy), from the repo root:

    docker compose run --rm -v "$PWD:/repo" -w /repo ai python runtime/tools/make_gentle_motions.py

Outputs go to runtime/ai_server/motions/ and are committed, so the server needs no build step.
"""

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[2]
AD = REPO / "third_party" / "AnimatedDrawings"
OUT = REPO / "runtime" / "ai_server" / "motions"

# How much of the original rotation to keep (1 = unchanged, 0 = rest pose)
KEEP = {
    "RightShoulder": 0.5, "LeftShoulder": 0.5,  # collarbone
    "RightArm": 0.5, "LeftArm": 0.5,            # upper arm
    "RightForeArm": 0.2, "LeftForeArm": 0.2,    # elbow: nearly straight
}

TARGET_FPS = 30

# The slice of each source BVH to keep, in source (120fps) frame numbers.
#
# The frontend plays the clip with <video loop>, so the clip must start and end in nearly the same
# pose or the seam pops. These were found by running experiments/render-cost/pick_window.py on the
# generated _gentle.bvh (it scores every (start, end) pair by seam distance against how much
# movement the window holds). Run on the source instead it lands 2-7 frames away, which does not
# matter: experiments/render-cost/check_seam.py on the rendered video puts the seam at 1.5x the
# average frame-to-frame change and below the largest one, i.e. not visible.
#
#   wave_hello  465-524  one wave of the raised hand       seam 0.52 (2 waves scored 1.06 — the
#                                                          arm is on its way down, so it drifts)
#   jumping     218-426  crouch, jump, land, stand again   seam 1.40 (both ends standing)
#
# Note wave_hello's first 3.4s are the character standing still, so cutting "the first 3 seconds"
# would have kept the part where nothing happens.
CLIPS = {
    "wave_hello": (465, 524),
    "jumping": (218, 426),
}

# Keep the character in place: freeze the root's horizontal travel. With the arms held out,
# sideways travel pushes a hand out of the rendered frame (seen with jumping). Up/down is kept.


def parse_channels(header: list[str]) -> list[tuple[str, list[str]]]:
    """[(joint, [channel, ...]), ...] in the order their values appear in a frame line."""

    out, joint = [], None
    for line in header:
        words = line.split()
        if not words:
            continue
        if words[0] in ("ROOT", "JOINT"):
            joint = words[1]
        elif words[0] == "CHANNELS":
            out.append((joint, words[2:]))
    return out


def gentle(bvh_path: Path, out_path: Path, up: str, clip: tuple[int, int]) -> tuple[int, float]:
    lines = bvh_path.read_text().splitlines()
    motion_at = next(i for i, l in enumerate(lines) if l.strip() == "MOTION")
    header, frames_at = lines[:motion_at], motion_at + 3  # MOTION, Frames:, Frame Time:
    frame_time = float(lines[motion_at + 2].split(":")[1])

    targets = []  # (first value index, euler order, keep)
    root_horizontal = []  # value indexes of the root's position channels other than "up"
    index = 0
    for joint, channels in parse_channels(header):
        if not root_horizontal:
            root_horizontal = [
                index + i for i, c in enumerate(channels)
                if c.endswith("position") and c[0].lower() != up
            ]
        rot = [c for c in channels if c.endswith("rotation")]
        if joint in KEEP:
            first = index + channels.index(rot[0])
            assert channels[channels.index(rot[0]):][:3] == rot, f"{joint}: rotations not contiguous"
            order = "".join(c[0].upper() for c in rot)  # intrinsic, as BVH applies them
            targets.append((first, order, KEEP[joint]))
        index += len(channels)
    assert len(targets) == len(KEEP), f"{bvh_path.name}: missing arm joints"

    data = np.array([[float(v) for v in l.split()] for l in lines[frames_at:] if l.strip()])
    # Keep only the loop the scene plays, then thin 120fps down to TARGET_FPS.
    start, end = clip
    assert 0 <= start < end <= len(data), f"{bvh_path.name}: clip {clip} outside {len(data)} frames"
    data = data[start:end]
    step = max(1, round(1 / (TARGET_FPS * frame_time)))
    data = data[::step]
    out_frame_time = frame_time * step
    # Pillow floors a .gif frame to 20ms. A shorter frame time plays back in slow motion.
    assert round(out_frame_time * 1000) >= 20, f"{bvh_path.name}: {1 / out_frame_time:.0f}fps too fast for .gif"

    for first, order, keep in targets:
        euler = data[:, first:first + 3]
        rotvec = Rotation.from_euler(order, euler, degrees=True).as_rotvec() * keep
        data[:, first:first + 3] = Rotation.from_rotvec(rotvec).as_euler(order, degrees=True)

    for i in root_horizontal:
        data[:, i] = data[0, i]

    body = [" ".join(f"{v:.5f}" for v in row) for row in data]
    out_path.write_text("\n".join(
        lines[:motion_at + 1]
        + [f"Frames: {len(data)}", f"Frame Time: {out_frame_time:.7f}"]
        + body
    ) + "\n")
    return len(data), 1 / out_frame_time


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, clip in CLIPS.items():
        cfg = yaml.safe_load((AD / f"examples/config/motion/{name}.yaml").read_text())
        src = AD / cfg["filepath"]
        up = cfg["up"].lstrip("+-").lower()  # "+z" -> "z"
        n, fps = gentle(src, OUT / f"{name}_gentle.bvh", up, clip)
        # Same ground joint, scale and orientation as Meta's config. The BVH is already cut, so the
        # frame range covers all of it. A relative filepath is resolved next to this yaml.
        cfg["filepath"] = f"{name}_gentle.bvh"
        cfg["start_frame_idx"] = 0
        cfg["end_frame_idx"] = n
        (OUT / f"{name}_gentle.yaml").write_text(
            f"# Generated by runtime/tools/make_gentle_motions.py from {src.relative_to(REPO)}\n"
            f"# Source frames {clip[0]}-{clip[1]} thinned to {fps:.0f}fps. See the tool for why.\n"
            + yaml.safe_dump(cfg, sort_keys=False)
        )
        print(f"{name}_gentle: {n} frames, {fps:.0f}fps, {n / fps:.2f}s")


if __name__ == "__main__":
    main()
