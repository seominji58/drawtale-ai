"""사슬형 · 지느러미형 마디 점 찾기 시험 (11/5). 그림 → 마스크 → 몸 가운데 선 → 머리 쪽 → 같은 간격 마디.
python find_chain.py 그림.png 마디수 → 그림.mask.png, 그림.joints.json, 그림.overlay.png"""
import json, sys
from collections import deque
import cv2
import numpy as np
from skimage.morphology import skeletonize

def mask_of(img):
    ink = (img.min(axis=2) < 235).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    body = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    flood = body.copy(); h, w = body.shape
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    return body | (1 - flood)                                    # 안의 구멍(눈 흰자)까지 채운다

def far(skel, start):
    """뼈대 위 너비 우선 — 가장 먼 점과 거기까지의 길."""
    prev = {start: None}; q = deque([start]); last = start
    while q:
        p = q.popleft(); last = p
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                n = (p[0] + dy, p[1] + dx)
                if n not in prev and 0 <= n[0] < skel.shape[0] and 0 <= n[1] < skel.shape[1] and skel[n]:
                    prev[n] = p; q.append(n)
    path = [last]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return last, path[::-1]

def extend(path, mask, k=12):
    """뼈대 끝은 몸 끝보다 안쪽에서 끝난다 — 끝 방향으로 마스크 끝까지 늘인다."""
    y, x = path[-1]; py, px = path[max(0, len(path) - 1 - k)]
    d = np.array([y - py, x - px], float); d /= max(1e-6, np.linalg.norm(d))
    out = list(path); p = np.array([y, x], float)
    while True:
        p = p + d; iy, ix = int(round(p[0])), int(round(p[1]))
        if not (0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]) or not mask[iy, ix]:
            break
        out.append((iy, ix))
    return out

def main(path, n_joints, fin=False):
    img = cv2.imread(path); mask = mask_of(img)
    skel = skeletonize(mask > 0)
    start = tuple(np.argwhere(skel)[0])
    a, _ = far(skel, start); b, line = far(skel, a)              # 가장 긴 길 (지름)
    line = extend(extend(line, mask)[::-1], mask)[::-1]
    # 머리 = 몸 안쪽의 어두운 점(눈 · 콧구멍)이 가까운 끝. 윤곽선은 마스크 가장자리 띠라 뺀다
    inner = (img.min(axis=2) < 80) & (cv2.erode(mask, np.ones((9, 9), np.uint8)) > 0)
    ys, xs = np.nonzero(inner)
    def score(end):
        if not len(ys): return 0
        d = np.hypot(ys - end[0], xs - end[1]); return float((d < 0.25 * len(line)).sum())
    if score(line[-1]) > score(line[0]):
        line = line[::-1]                                         # line[0] 이 머리
    pts = np.array(line, float); seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    at = np.linspace(0, seg[-1], n_joints)
    ys_ = np.interp(at, seg, pts[:, 0]); xs_ = np.interp(at, seg, pts[:, 1])
    # 꼬리 끝(지느러미형만): 뼈대 끝은 갈라진 지느러미의 한쪽 모서리로 간다 — 마지막 앞 마디에서 몸 방향으로 쭉 간 끝.
    # 사슬형은 꼬리가 휘어 있으면 몸 방향으로 가면 꼬리가 잘려 뼈대 끝 그대로
    d = np.array([ys_[-2] - ys_[0], xs_[-2] - xs_[0]]); d /= max(1e-6, np.linalg.norm(d))
    p = np.array([ys_[-2], xs_[-2]])
    while fin:
        q = p + d; iy, ix = int(round(q[0])), int(round(q[1]))
        if not (0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]) or not mask[iy, ix]:
            break
        p = q
    if fin:
        ys_[-1], xs_[-1] = p
    names = ["head"] + [f"b{i}" for i in range(1, n_joints - 1)] + ["tail"]
    joints = [{"name": nm, "x": round(float(x)), "y": round(float(y))} for nm, x, y in zip(names, xs_, ys_)]
    base = path.rsplit(".", 1)[0]
    cv2.imwrite(base + ".mask.png", mask * 255)
    json.dump(joints, open(base + ".joints.json", "w"))
    ov = img.copy()
    for j0, j1 in zip(joints, joints[1:]):
        cv2.line(ov, (j0["x"], j0["y"]), (j1["x"], j1["y"]), (255, 255, 255), 5)
    for j in joints:
        cv2.circle(ov, (j["x"], j["y"]), 9, (0, 0, 255) if j["name"] == "head" else (255, 120, 0), -1)
    cv2.imwrite(base + ".overlay.png", ov)
    print(base, "마디", n_joints, "머리", joints[0], "꼬리", joints[-1])

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), fin=len(sys.argv) > 3 and sys.argv[3] == "fin")
