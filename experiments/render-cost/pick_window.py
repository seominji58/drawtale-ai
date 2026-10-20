"""반복 재생해도 이음새가 안 튀는 구간을 찾는다.

프론트는 장면 영상을 `<video loop>` 로 틉니다. 그래서 자른 구간의 **첫 자세와 끝 자세가
비슷해야** 반복할 때 안 튑니다. 눈으로 고르면 틀리기 쉬워서 재서 고릅니다.

관절 위치를 프레임마다 모두 구한 다음, (시작, 끝) 짝마다 둘을 잽니다.

  이음새 = 첫 자세와 끝 자세의 관절 거리 평균   (작아야 좋다)
  움직임 = 구간 안에서 관절이 움직인 양          (커야 좋다 — 가만히 있는 구간을 걸러낸다)

움직임이 상위 30% 안에 드는 것들 중 이음새가 가장 작은 것을 알려 줍니다.

    python pick_window.py <동작.bvh> <이름> <가장 짧게(초)> <가장 길게(초)>

AI 런타임 이미지 안에서 돌립니다 (Meta 의 BVH 파서를 그대로 씁니다).

    docker run --rm -v "${PWD}:/repo" -w /repo drawtale-ai-ai:latest \
      python experiments/render-cost/pick_window.py \
        runtime/ai_server/motions/wave_hello_gentle.bvh 손흔들기 0.5 2.5
"""

import sys

sys.path.insert(0, "/opt/AnimatedDrawings")

import numpy as np
from animated_drawings.model.bvh import BVH

MOVE_PERCENTILE = 70  # 움직임이 이 분위 아래인 구간은 버린다 (가만히 있는 곳)


def joint_positions(bvh: BVH) -> np.ndarray:
    """(프레임, 관절, xyz) — 엉덩이를 원점으로 옮긴 관절 위치."""

    names = bvh.get_joint_names()
    joints = [bvh.root_joint.get_transform_by_name(n) for n in names]
    hips = bvh.root_joint.get_transform_by_name("Hips")

    out = np.zeros((len(bvh.pos_data), len(joints), 3), dtype=np.float32)
    for f in range(len(bvh.pos_data)):
        bvh.apply_frame(f)
        bvh.root_joint.update_transforms()
        origin = hips.get_world_position()
        for i, j in enumerate(joints):
            out[f, i] = j.get_world_position() - origin
    return out


def main() -> None:
    path, label = sys.argv[1], sys.argv[2]
    shortest, longest = float(sys.argv[3]), float(sys.argv[4])

    bvh = BVH.from_file(path)
    frame_time = bvh.frame_time
    pos = joint_positions(bvh)
    n = len(pos)

    # 프레임마다 관절이 움직인 거리의 합 — 구간 평균을 빨리 내려고 누적해 둔다
    moved = np.linalg.norm(np.diff(pos, axis=0), axis=2).sum(axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(moved)])

    lo, hi = int(shortest / frame_time), int(longest / frame_time)
    rows = []
    for start in range(0, n - lo):
        for end in range(start + lo, min(start + hi, n)):
            seam = float(np.linalg.norm(pos[end] - pos[start], axis=1).mean())
            move = float((cumulative[end] - cumulative[start]) / (end - start))
            rows.append((seam, move, start, end))

    rows.sort(key=lambda r: r[0])
    threshold = np.percentile([r[1] for r in rows], MOVE_PERCENTILE)
    best = [r for r in rows if r[1] >= threshold][:6]

    print(f"== {label}  전체 {n}프레임 {n * frame_time:.1f}초 ({1 / frame_time:.0f}fps)")
    print("  이음새  움직임   구간(초)              프레임")
    for seam, move, start, end in best:
        print(f"  {seam:6.2f}  {move:6.2f}   "
              f"{start * frame_time:4.2f}~{end * frame_time:4.2f} ({(end - start) * frame_time:.2f}초)   "
              f"{start}~{end}")


if __name__ == "__main__":
    main()
