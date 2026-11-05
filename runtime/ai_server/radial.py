"""촉수형(문어 · 해파리 · 불가사리) 분석 — 몸통에서 뻗은 다리(촉수)를 뼈대선 끝으로 찾는다 (pt_docs 19, 11/5 시험).

문어 그림은 머리와 다리가 한 윤곽선이라 칸으로 나눌 수 없다(날개형과 다름). 마스크의 뼈대선을 그려
- 몸통: 가장 두꺼운 곳(거리 변환 최댓값)
- 다리 끝: 뼈대선 끝점 중 몸통에서 몸 반지름보다 멀리 나간 것 (위로 난 머리 꼭대기는 빼려고 몸통보다 아래쪽만)
- 다리 뿌리: 끝에서 몸통 쪽으로 뼈대선을 따라가다 몸이 다리 굵기의 두 배를 넘게 굵어지는 곳
다리는 8개까지(몸통 중심 둘레 각도 순서). 모자라면 남는 다리 점을 몸통에 겹친다(움직여도 아무것도 끌지 않는다).
그림 영역은 4족형과 같다(quadruped.cut_out). 움직임은 renderer._apply_wave (다리마다 박자를 엇갈린 물결 · 몸 둥실).
시험: experiments/radial
"""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ai_server import analyzer as A
from ai_server import quadruped as Q

LEGS = 8
NAMES = ["body"] + [f"leg{i}_{p}" for i in range(1, LEGS + 1) for p in ("root", "mid", "tip")]


def _paths_from(skel: np.ndarray, start: tuple[int, int]) -> dict:
    """뼈대 위 너비 우선 — 점마다 앞 점 (start 로 거슬러 가는 길)."""
    prev = {start: None}
    q = deque([start])
    H, W = skel.shape
    while q:
        p = q.popleft()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                n = (p[0] + dy, p[1] + dx)
                if n not in prev and 0 <= n[0] < H and 0 <= n[1] < W and skel[n]:
                    prev[n] = p
                    q.append(n)
    return prev


def find_joints(mask: np.ndarray) -> tuple[dict[str, tuple[float, float]], bool]:
    """관절 (몸통 + 다리마다 뿌리 · 가운데 · 끝), 다리를 셋 넘게 찾았나."""
    from skimage.morphology import skeletonize

    m = mask.astype(np.uint8)
    h, w = m.shape
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    m = m | (1 - flood)                                       # 눈 · 입 구멍을 채운다
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    R = float(dt[cy, cx])
    body = (float(cx), float(cy))
    none = {n: body for n in NAMES}
    skel = skeletonize(m > 0)
    pts = np.argwhere(skel)
    if not len(pts):
        return none, False
    start = tuple(pts[int(np.argmin(np.hypot(pts[:, 0] - cy, pts[:, 1] - cx)))])
    prev = _paths_from(skel, start)
    # 끝점: 이웃 뼈대 점이 하나뿐인 점
    nb = cv2.filter2D(skel.astype(np.uint8), -1, np.ones((3, 3), np.float32)) - skel.astype(np.uint8)
    ends = [tuple(p) for p in np.argwhere(skel & (nb == 1)) if tuple(p) in prev]
    legs = []
    for e in ends:
        far = np.hypot(e[0] - cy, e[1] - cx)
        if far < 1.3 * R or e[0] < cy - 0.3 * R:             # 몸통 안 · 머리 꼭대기 쪽 잔가지
            continue
        path = [e]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])                       # 끝 → 몸통
        th = np.array([dt[p] for p in path])
        leg_w = float(np.median(th[:max(3, len(th) // 3)]))
        root_i = next((i for i, t in enumerate(th) if t > 2.0 * max(leg_w, 1.5)), len(path) - 1)
        if root_i < 0.3 * R:                                   # 다리라기엔 짧다
            continue
        tip, root = path[0], path[root_i]
        mid = path[root_i // 2]
        legs.append(((root[1], root[0]), (mid[1], mid[0]), (tip[1], tip[0])))
    # 몸통 둘레 각도 순서 (왼쪽 아래 → 오른쪽 아래), 너무 가까운 끝(같은 다리의 갈래)은 하나만
    legs.sort(key=lambda L: np.arctan2(L[2][1] - cy, cx - L[2][0]))
    kept = []
    for L in legs:
        if all(np.hypot(L[2][0] - K[2][0], L[2][1] - K[2][1]) > 0.35 * R for K in kept):
            kept.append(L)
    kept = kept[:LEGS]
    J = {"body": body}
    for i in range(LEGS):
        r_, m_, t_ = kept[i] if i < len(kept) else (body, body, body)
        J[f"leg{i + 1}_root"], J[f"leg{i + 1}_mid"], J[f"leg{i + 1}_tip"] = (
            tuple(map(float, r_)), tuple(map(float, m_)), tuple(map(float, t_)))
    return J, len(kept) >= 3


def analyze(image_path: str, mask_path: str | None, facing: str | None = None) -> dict:
    """analyzer.analyze_image 와 같은 모양 (+ form · joints_found). 다리를 셋 넘게 못 찾으면 joints_found=False."""
    started = time.perf_counter()
    cut = Q.cut_out(image_path, grow=True)
    if "error" in cut:
        return {"success": False, "bbox": None, "joints": [], "message": cut["error"]}
    joints, found = find_joints(cut["mask"] > 127)
    saved = False
    if mask_path is not None:
        Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
        saved = bool(cv2.imwrite(mask_path, cut["mask"]))
    box = cut["box"]
    return {
        "success": True,
        "form": "radial",
        "joints_found": found,
        "bbox": cut["big_box"],
        "joints": [{"name": n, "x": round(joints[n][0]), "y": round(joints[n][1]), "score": None} for n in NAMES],
        "confidence": round(box["score"] if found else min(box["score"], 0.2), 3),
        "mask": {"width": cut["original"].shape[1], "height": cut["original"].shape[0],
                 "saved": saved, "covers": "full_image"},
        "message": "Analysis complete" if found else "JOINTS_NOT_FOUND: legs not found, template skeleton",
        "model_version": A.MODEL_VERSION,
        "pipeline_version": A.PIPELINE_VERSION,
        "coordinate_space": A.COORDINATE_SPACE,
        "processing_time_ms": round((time.perf_counter() - started) * 1000),
    }
