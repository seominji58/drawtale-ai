"""날개형(새 · 나비 · 잠자리) 분석 — 선으로 닫힌 칸으로 머리 · 몸통 · 날개를 나눈다 (pt_docs 19, 11/5 시험).

아이 선 그림은 머리 · 몸통 · 날개가 선으로 닫힌 칸이다. 굵기로는 나눌 수 없었다(날개가 몸통만큼 굵다).
- 머리: 눈(칸 안에 갇힌 작은 잉크 점)이 든 칸
- 몸통: 남은 큰 칸 중 가장 아래(다리가 붙는 쪽)
- 날개: 몸통에 붙은 나머지 큰 칸 중 바깥 종이에 닿는 것(몸 밖으로 솟은 날개). 몸통에 둘러싸인 칸은 몸 안에 그린
  날개라 퍼덕이지 않는다 — 날개 점을 몸통에 겹친다
날개는 둘까지. 하나면 날개 2 점을 날개 1 에 겹쳐 같이 퍼덕인다. 칸이 안 나뉘는 그림(선 없이 색만 칠한 그림 등)은
날개 점을 몸통에 겹친 기본 자리로 두고 어른이 맞춘다. 그림 영역(마스크)은 4족형과 같다(quadruped.cut_out).
움직임은 renderer._apply_wave (형태 정의 waves — 날개 뿌리를 축으로 위아래). 시험: experiments/winged
"""

import time
from pathlib import Path

import cv2
import numpy as np

from ai_server import analyzer as A
from ai_server import quadruped as Q

NAMES = ["body", "head", "wing1_root", "wing1_mid", "wing1_tip", "wing2_root", "wing2_mid", "wing2_tip"]
# 칸으로 치는 넓이 (그림 안 종이 넓이 대비)
MIN_CELL = 0.03
# 눈으로 치는 잉크 점 넓이 상한
MAX_EYE = 0.01


def find_joints(img: np.ndarray, mask: np.ndarray, facing: str | None = None) -> tuple[dict[str, tuple[float, float]], bool]:
    """관절 (몸통 · 머리 · 날개 둘), 찾았나(머리 · 몸통을 나눴나). 몸 안 날개는 정상 그림이라 찾은 것으로 — 날개 점만 몸통에 겹친다."""
    ys0, xs0 = np.nonzero(mask)
    box = (xs0.min(), ys0.min(), xs0.max() + 1, ys0.max() + 1)
    crop = img[box[1]:box[3], box[0]:box[2]]
    inside = mask[box[1]:box[3], box[0]:box[2]] > 0
    ink = ((crop.min(axis=2) < 160) & inside).astype(np.uint8)
    lw = max(3.0, 2 * float(cv2.distanceTransform(ink, cv2.DIST_L2, 3).max()))   # 선 굵기
    # 날개 선과 몸 선이 만나는 곳의 작은 틈을 막는다 — 틈으로 날개 칸이 바깥과 이어져 칸이 안 잡혔다
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    paper = ((ink == 0) & inside).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(paper, connectivity=4)
    outside = (~inside) | np.zeros_like(inside)
    total = float(paper.sum()) or 1.0
    cells = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > MIN_CELL * total]

    def centre(sel: np.ndarray) -> tuple[float, float]:
        yy, xx = np.nonzero(sel)
        return float(xx.mean()) + box[0], float(yy.mean()) + box[1]

    body_c = centre(inside)
    fallback = {n_: body_c for n_ in NAMES}
    if len(cells) < 2:
        return fallback, False

    # 머리: 눈이 든 칸 (눈 = 큰 선 덩어리가 아닌 작은 잉크 점)
    m_ink, lab_ink, st_ink, cen_ink = cv2.connectedComponentsWithStats(ink, 8)
    big = 1 + int(np.argmax(st_ink[1:, cv2.CC_STAT_AREA])) if m_ink > 1 else -1
    head = None
    for j in range(1, m_ink):
        if j == big or st_ink[j, cv2.CC_STAT_AREA] > MAX_EYE * total:
            continue
        ex, ey = int(cen_ink[j][0]), int(cen_ink[j][1])
        r = int(2 * lw)
        near = [i for i in cells if (lab[max(0, ey - r):ey + r, max(0, ex - r):ex + r] == i).any()]
        if near:
            head = max(near, key=lambda i: st[i, cv2.CC_STAT_AREA])
            break
    rest = [i for i in cells if i != head]
    if not rest:
        return fallback, False
    body = max(rest, key=lambda i: st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT])

    def ring(i: int) -> np.ndarray:
        k = int(2.5 * lw) | 1
        return (cv2.dilate((lab == i).astype(np.uint8), np.ones((k, k), np.uint8)) > 0) & (lab != i)

    wings = []
    for i in rest:
        if i == body:
            continue
        around = ring(i)
        if not (lab[around] == body).any():
            continue
        if (around & outside).any() or (around & (lab == 0) & ~(ink > 0)).any():
            wings.append(i)                       # 바깥에 닿는 칸 = 몸 밖으로 솟은 날개
    body_c = centre(lab == body)
    J = {"body": body_c, "head": centre(lab == head) if head is not None else body_c}
    found = []
    for i in sorted(wings, key=lambda i: -st[i, cv2.CC_STAT_AREA])[:2]:
        yy, xx = np.nonzero(lab == i)
        k = int(3 * lw) | 1
        touch = cv2.dilate((ring(i) & (lab == body)).astype(np.uint8), np.ones((k, k), np.uint8)) & (lab == i)
        ty, tx = np.nonzero(touch)
        root = (float(tx.mean()), float(ty.mean())) if len(tx) else (float(xx.mean()), float(yy.mean()))
        far = int(np.argmax(np.hypot(xx - root[0], yy - root[1])))
        tip = (float(xx[far]), float(yy[far]))
        mid = ((root[0] + tip[0]) / 2, (root[1] + tip[1]) / 2)
        found.append(tuple((x + box[0], y + box[1]) for x, y in (root, mid, tip)))
    ok = head is not None
    if not found:
        found = [(body_c, body_c, body_c)]
    if len(found) == 1:
        found.append(found[0])
    for k_, (r_, m_, t_) in zip((1, 2), found):
        J[f"wing{k_}_root"], J[f"wing{k_}_mid"], J[f"wing{k_}_tip"] = r_, m_, t_
    return J, ok


def analyze(image_path: str, mask_path: str | None, facing: str | None = None) -> dict:
    """analyzer.analyze_image 와 같은 모양 (+ form · joints_found). 머리 · 몸통 칸을 못 나눈 그림은 joints_found=False."""
    started = time.perf_counter()
    cut = Q.cut_out(image_path, grow=True)       # 크게 솟은 날개는 사람 탐지 상자 밖일 때가 많다
    if "error" in cut:
        return {"success": False, "bbox": None, "joints": [], "message": cut["error"]}
    joints, found = find_joints(cut["original"], cut["mask"] > 127, facing)
    saved = False
    if mask_path is not None:
        Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
        saved = bool(cv2.imwrite(mask_path, cut["mask"]))
    box = cut["box"]
    return {
        "success": True,
        "form": "winged",
        "joints_found": found,
        "bbox": cut["big_box"],
        "joints": [{"name": n, "x": round(joints[n][0]), "y": round(joints[n][1]), "score": None} for n in NAMES],
        "confidence": round(box["score"] if found else min(box["score"], 0.2), 3),
        "mask": {"width": cut["original"].shape[1], "height": cut["original"].shape[0],
                 "saved": saved, "covers": "full_image"},
        "message": "Analysis complete" if found else "JOINTS_NOT_FOUND: head and body regions not found, template skeleton",
        "model_version": A.MODEL_VERSION,
        "pipeline_version": A.PIPELINE_VERSION,
        "coordinate_space": A.COORDINATE_SPACE,
        "processing_time_ms": round((time.perf_counter() - started) * 1000),
    }
