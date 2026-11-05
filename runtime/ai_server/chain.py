"""사슬형(뱀 · 지렁이) · 지느러미형(물고기 · 고래) 분석 — 몸 가운데 선을 같은 간격 마디로 (pt_docs 19, 11/5).

그림 영역은 4족형과 같은 방법(quadruped.cut_out). 마스크의 뼈대 선에서 가장 긴 길이 몸 가운데 선이고,
눈 · 콧구멍처럼 몸 안의 어두운 점이 가까운 끝이 머리다(없으면 굵은 끝). 마디 수 · 이름은 형태 정의의 joints.
지느러미형은 뼈대 끝이 갈라진 꼬리지느러미의 한쪽 모서리로 가서, 꼬리 끝만 몸 방향으로 쭉 뻗은 지느러미 끝에 둔다.
움직임은 renderer._apply_wave (형태 정의 waves). 시험: experiments/chain (디지털 그림 뱀 · 물고기).
"""

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from ai_server import analyzer as A, forms
from ai_server import quadruped as Q

# 몸 가운데 선이 그림 긴 변의 이만큼도 안 되면 못 찾은 것 (점 하나 · 동그라미)
MIN_LINE = 0.35


def _far(skel: np.ndarray, start: tuple[int, int]) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """뼈대 위 너비 우선 — 가장 먼 점과 거기까지의 길 (y, x)."""
    prev = {start: None}
    q = deque([start])
    last = start
    H, W = skel.shape
    while q:
        p = q.popleft()
        last = p
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                n = (p[0] + dy, p[1] + dx)
                if n not in prev and 0 <= n[0] < H and 0 <= n[1] < W and skel[n]:
                    prev[n] = p
                    q.append(n)
    path = [last]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return last, path[::-1]


def _march(mask: np.ndarray, p: np.ndarray, d: np.ndarray) -> np.ndarray:
    """p 에서 d 방향으로 마스크 끝까지."""
    d = d / max(1e-6, float(np.linalg.norm(d)))
    while True:
        q = p + d
        iy, ix = int(round(q[0])), int(round(q[1]))
        if not (0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]) or not mask[iy, ix]:
            return p
        p = q


def _extend(path: list, mask: np.ndarray, k: int = 12) -> list:
    """뼈대 끝은 몸 끝보다 안쪽에서 끝난다 — 끝 방향으로 마스크 끝까지 늘인다."""
    end = np.array(path[-1], float)
    back = np.array(path[max(0, len(path) - 1 - k)], float)
    tip = _march(mask, end, end - back)
    n = int(np.linalg.norm(tip - end))
    return path + [tuple(np.round(end + (tip - end) * i / max(1, n)).astype(int)) for i in range(1, n + 1)]


def template(mask: np.ndarray, names: list[str], facing: str | None) -> dict[str, tuple[float, float]]:
    """못 찾았을 때 — 마스크 상자 가운데 가로줄 (머리 오른쪽, facing=left 면 왼쪽). 사람이 옮겨 맞춘다."""
    ys, xs = np.nonzero(mask)
    left, right, cy = float(xs.min()), float(xs.max()), float((ys.min() + ys.max()) / 2)
    xs_ = np.linspace(left, right, len(names)) if facing == "left" else np.linspace(right, left, len(names))
    return {n: (float(x), cy) for n, x in zip(names, xs_)}


def find_joints(mask: np.ndarray, original: np.ndarray, names: list[str], fin: bool,
                facing: str | None = None) -> tuple[dict[str, tuple[float, float]], bool]:
    """마디 (머리 → 꼬리 끝), 찾았나."""
    from skimage.morphology import skeletonize

    filled = mask.astype(np.uint8)
    h, w = filled.shape
    flood = filled.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 1)
    filled = filled | (1 - flood)                              # 눈 흰자 같은 구멍을 채운다 (뼈대가 돌아 나가지 않게)
    skel = skeletonize(filled > 0)
    if not skel.any():
        return template(mask, names, facing), False
    a, _ = _far(skel, tuple(np.argwhere(skel)[0]))
    _, line = _far(skel, a)                                     # 가장 긴 길 (지름)
    line = _extend(_extend(line, filled)[::-1], filled)[::-1]
    ys, xs = np.nonzero(mask)
    if len(line) < MIN_LINE * max(np.ptp(xs), np.ptp(ys)):
        return template(mask, names, facing), False

    # 머리 쪽: 지정했으면(「얼굴이 반대쪽이에요」) 그대로, 아니면 몸 안 어두운 점(눈)이 가까운 끝, 없으면 굵은 끝
    if facing in ("left", "right"):
        head_first = (line[0][1] <= line[-1][1]) == (facing == "left")
    else:
        inner = (original.min(axis=2) < 80) & (cv2.erode(filled, np.ones((9, 9), np.uint8)) > 0)
        iy, ix = np.nonzero(inner)
        reach = 0.25 * len(line)
        dt = cv2.distanceTransform(filled, cv2.DIST_L2, 5)

        def marks(end: tuple[int, int]) -> float:
            return float((np.hypot(iy - end[0], ix - end[1]) < reach).sum()) if len(iy) else 0.0

        def thick(part: list) -> float:
            return float(np.mean([dt[p] for p in part]))

        m0, m1 = marks(line[0]), marks(line[-1])
        quarter = max(1, len(line) // 4)
        head_first = m0 > m1 if m0 != m1 else thick(line[:quarter]) >= thick(line[-quarter:])
    if not head_first:
        line = line[::-1]

    pts = np.array(line, float)
    seg = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
    at = np.linspace(0, seg[-1], len(names))
    ys_, xs_ = np.interp(at, seg, pts[:, 0]), np.interp(at, seg, pts[:, 1])
    if fin:
        # 사슬형은 꼬리가 휘어 있으면 몸 방향으로 가면 꼬리가 잘려 뼈대 끝 그대로
        tip = _march(filled, np.array([ys_[-2], xs_[-2]]), np.array([ys_[-2] - ys_[0], xs_[-2] - xs_[0]]))
        ys_[-1], xs_[-1] = tip
    return {n: (float(x), float(y)) for n, x, y in zip(names, xs_, ys_)}, True


def analyze(image_path: str, mask_path: str | None, form: str, facing: str | None = None) -> dict:
    """analyzer.analyze_image 와 같은 모양 (+ form · joints_found)."""
    started = time.perf_counter()
    cut = Q.cut_out(image_path)
    if "error" in cut:
        return {"success": False, "bbox": None, "joints": [], "message": cut["error"]}
    names = [j["name"] for j in forms.get(form)["joints"]]
    joints, found = find_joints(cut["mask"] > 127, cut["original"], names, form == "finned", facing)
    saved = False
    if mask_path is not None:
        Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
        saved = bool(cv2.imwrite(mask_path, cut["mask"]))
    box = cut["box"]
    return {
        "success": True,
        "form": form,
        "joints_found": found,
        "bbox": cut["big_box"],
        "joints": [{"name": n, "x": round(joints[n][0]), "y": round(joints[n][1]), "score": None} for n in names],
        "confidence": round(box["score"] if found else min(box["score"], 0.2), 3),
        "mask": {"width": cut["original"].shape[1], "height": cut["original"].shape[0],
                 "saved": saved, "covers": "full_image"},
        "message": "Analysis complete" if found else "JOINTS_NOT_FOUND: body line not found, template skeleton",
        "model_version": A.MODEL_VERSION,
        "pipeline_version": A.PIPELINE_VERSION,
        "coordinate_space": A.COORDINATE_SPACE,
        "processing_time_ms": round((time.perf_counter() - started) * 1000),
    }
