"""마스크 모양만 보고 손·발 관절을 팔다리 끝으로 옮긴다 (AI 없이).

원리: 몸통 중심에서 가장 멀리 뻗은 지점들이 손·발이다.
각 관절은 자기 팔다리 방향에 있는 끝점 중 가장 가까운 것으로 옮긴다.
"""
import json
import sys

import numpy as np
from PIL import Image

BASE = r'C:\Users\USER\drawtale\drawtale-ai\output'

# 손·발과 그 바로 위 관절 (방향을 계산하는 데 쓴다)
LIMBS = {
    'right_hand': 'right_elbow',
    'left_hand': 'left_elbow',
    'right_foot': 'right_knee',
    'left_foot': 'left_knee',
}


def find_tips(mask: np.ndarray, center: tuple[float, float], count: int = 12):
    """몸통 중심에서 멀리 뻗은 끝점 후보를 찾는다.

    경계 픽셀 중 중심에서 먼 것들을 골라, 서로 떨어진 것만 남긴다.
    """
    from scipy import ndimage

    edge = mask & ~ndimage.binary_erosion(mask)
    ys, xs = np.where(edge)
    dist = np.hypot(xs - center[0], ys - center[1])
    order = np.argsort(-dist)

    tips = []
    min_gap = 0.12 * max(mask.shape)
    for i in order:
        p = (xs[i], ys[i])
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) > min_gap for q in tips):
            tips.append(p)
        if len(tips) >= count:
            break
    return tips


def fit(joints: list[dict], mask: np.ndarray) -> list[dict]:
    by = {j['name']: j for j in joints}
    center = (by['torso']['x'], by['torso']['y'])
    tips = find_tips(mask, center)

    for tip_joint, parent_name in LIMBS.items():
        joint = by[tip_joint]
        parent = by[parent_name]

        # 팔다리가 뻗은 방향
        vx, vy = joint['x'] - parent['x'], joint['y'] - parent['y']
        length = np.hypot(vx, vy) or 1
        vx, vy = vx / length, vy / length

        best, best_score = None, -1e9
        for tx, ty in tips:
            # 부모 관절에서 끝점까지의 방향
            dx, dy = tx - parent['x'], ty - parent['y']
            d = np.hypot(dx, dy) or 1
            # 팔다리 방향과 얼마나 일치하는지 (1이면 완전히 같은 방향)
            align = (dx / d) * vx + (dy / d) * vy
            if align < 0.55:        # 엉뚱한 방향은 제외
                continue
            # 방향이 맞으면서 멀리 있는 점을 선호
            score = align * 2 + d / max(mask.shape)
            if score > best_score:
                best, best_score = (tx, ty), score

        if best is None:
            print(f'  {tip_joint:14} 맞는 끝점을 찾지 못해 유지')
            continue

        # 끝점에 그대로 두면 테두리에 닿으므로 살짝 안으로 당긴다
        nx = best[0] - vx * 0.03 * max(mask.shape)
        ny = best[1] - vy * 0.03 * max(mask.shape)
        moved = np.hypot(nx - joint['x'], ny - joint['y'])
        print(f"  {tip_joint:14} ({joint['x']:4.0f},{joint['y']:4.0f}) -> ({nx:4.0f},{ny:4.0f})  {moved:4.0f}px 이동")
        joint['x'], joint['y'] = round(float(nx)), round(float(ny))

    return joints


if __name__ == '__main__':
    joints_path = sys.argv[1] if len(sys.argv) > 1 else BASE + r'\joints.json'
    mask_path = sys.argv[2] if len(sys.argv) > 2 else BASE + r'\mask.png'
    out_path = sys.argv[3] if len(sys.argv) > 3 else BASE + r'\joints_fitted.json'

    joints = json.load(open(joints_path))
    mask = np.array(Image.open(mask_path).convert('L')) > 127
    result = fit(joints, mask)
    open(out_path, 'w').write(json.dumps(result))
    print('저장:', out_path)
