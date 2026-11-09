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


def _shape_guess(img: np.ndarray, mask: np.ndarray) -> dict[str, tuple[float, float]] | None:
    """선으로 칸을 못 나눈 그림의 관절 어림 — 몸통: 가장 두꺼운 곳, 머리: 눈(몸 안의 작은 어두운 점) 쪽,
    날개 끝: 몸통 위로 머리에서 먼 쪽 가장 멀리 솟은 곳, 날개 뿌리: 몸통에서 날개 끝 쪽 몸 반지름만큼. 날개는 하나(두 번째는 겹침)."""
    m = mask.astype(np.uint8)
    if m.sum() < 50:
        return None
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    filled = m | (1 - flood)
    dt = cv2.distanceTransform(filled, cv2.DIST_L2, 5)
    by, bx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    R = float(dt[by, bx])
    body = (float(bx), float(by))
    light = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    ink = ((light < 90) & (dt > 0.15 * R)).astype(np.uint8)          # 몸 안쪽의 어두운 점 (테두리 선은 빼고)
    n, lab, st, cen = cv2.connectedComponentsWithStats(ink, 8)
    # 눈: 몸 안쪽의 작고 동그란 어두운 점 — 테두리에 붙은 선 조각 · 날개 깃 선(길쭉함)은 눈이 아니다. 가장 작은 것
    side = np.sqrt(filled.sum())
    eyes = sorted(((st[i, cv2.CC_STAT_AREA], cen[i]) for i in range(1, n)
                   if st[i, cv2.CC_STAT_AREA] < 0.02 * filled.sum()
                   and dt[int(cen[i][1]), int(cen[i][0])] > 0.2 * R
                   and max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]) < 0.08 * side
                   and max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])
                   < 2.5 * max(1, min(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]))), key=lambda e: e[0])
    eyes = [e[1] for e in eyes]
    ys, xs = np.nonzero(filled)
    # 날개 끝: 몸통보다 위로 가장 높이 솟은 곳 (아이들은 날개를 위로 그린다). 눈 위치로 먼저 정하면 날개 깃 사이 선을
    # 눈으로 잘못 봐 머리 · 날개가 뒤바뀌었다 (11/9 분홍 새)
    up = ys < by - 0.3 * R
    if not up.any():
        return {n_: body for n_ in NAMES}
    k = int(np.argmin(ys[up]))
    tip = (float(xs[up][k]), float(ys[up][k]))
    # 머리: 날개 끝 반대쪽(가로)으로 몸통에서 가장 멀리 나간 곳 — 그 근처에 눈 점이 있으면 눈
    side = -1.0 if tip[0] >= bx else 1.0
    mid_band = (ys > by - 2.0 * R) & (ys < by + 0.5 * R)
    far = (xs - bx) * side
    cand = mid_band & (far > 0)
    if cand.any():
        i = int(np.argmax(far[cand]))
        edge = (float(xs[cand][i]), float(ys[cand][i]))
        near = [e for e in eyes if (e[0] - bx) * side > 0 and np.hypot(e[0] - edge[0], e[1] - edge[1]) < 1.5 * R]
        head = (float(near[0][0]), float(near[0][1])) if near else (bx + (edge[0] - bx) * 0.75, edge[1] + 0.3 * R)
    else:
        head = body
    # 날개 뿌리: 몸통에서 날개 끝으로 가다 몸이 가늘어지는 곳(두께가 몸 반지름의 반 아래) — 몸 안쪽에 두면
    # 날개를 흔들 때 몸 앞쪽까지 같이 기울었다
    root = body
    for f in np.linspace(0, 1, 60):
        x, y = bx + (tip[0] - bx) * f, by + (tip[1] - by) * f
        if dt[int(round(y)), int(round(x))] < 0.5 * R:
            root = (float(x), float(y))
            break
    mid = ((root[0] + tip[0]) / 2, (root[1] + tip[1]) / 2)
    J = {"body": body, "head": head}
    for w in ("wing1", "wing2"):
        J[f"{w}_root"], J[f"{w}_mid"], J[f"{w}_tip"] = root, mid, tip
    return J


def find_joints(img: np.ndarray, mask: np.ndarray, facing: str | None = None) -> tuple[dict[str, tuple[float, float]], bool]:
    """관절 (몸통 · 머리 · 날개 둘), 찾았나(머리 · 몸통을 나눴나). 몸 안 날개는 정상 그림이라 찾은 것으로 — 날개 점만 몸통에 겹친다."""
    ys0, xs0 = np.nonzero(mask)
    box = (xs0.min(), ys0.min(), xs0.max() + 1, ys0.max() + 1)
    crop = img[box[1]:box[3], box[0]:box[2]]
    inside = mask[box[1]:box[3], box[0]:box[2]] > 0
    # 선 = 어두운 곳 — 밝기(Lab L)로 본다. RGB 최솟값으로 보면 갈색 · 진한 색으로 칠한 몸까지 선이 되어 칸이 안 나뉘었다 (11/5)
    light = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)[..., 0]
    ink = ((light < 90) & inside).astype(np.uint8)
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
    # 칸이 안 나뉘면(선 없이 색으로 한 덩어리로 칠한 새 — 11/9 사용자) 모든 점을 몸통에 겹치지 않고 모양으로 어림한다.
    # 겹쳐 두면 미리보기가 움직이지 않고, 관절 맞추기에서도 점이 한 곳에 쌓여 옮기기 어려웠다
    guess = _shape_guess(img, mask)
    fallback = guess or {n_: body_c for n_ in NAMES}
    # 모양 어림이 머리 · 날개를 나눴으면(머리 ≠ 몸통, 날개 끝 ≠ 몸통) 찾은 것으로 — 관절 고치기로 보내지 않고 바로 움직임 보기 (11/9)
    shaped = bool(guess) and guess["head"] != guess["body"] and guess["wing1_tip"] != guess["body"]
    if len(cells) < 2:
        return fallback, shaped

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
