"""4족형 분석 — 사람 관절 모델 없이 마스크 모양으로 (pt_docs 19, 2026-11-04).

사람 관절 모델은 강아지 · 공룡도 사람으로 통과시키고(신뢰도 0.93 · 0.95, 꼬리에 팔), 사람으로 안 보이면 거절한다
(실제 아이 그림: 악어 LOW_CONFIDENCE, 기린 + 작은 새 MULTIPLE). 그래서 4족형은

  1. 자리: 사람 탐지 상자를 점수와 상관없이 받는다. 여럿이면 가장 큰 상자(주인공). 없으면 종이 위 선의 범위
  2. 마스크: 사람 그림과 같은 선 찾기(create_mask). 사람 관절 대신 「상자 안 선이 마스크에 얼마나 들어오나」로 고른다
  3. 종이색 깎기(refine_mask) 뒤, 깎기가 지운 가는 선 다리 · 몸에서 조금 떨어져 그은 다리를 되살린다
  4. 관절: 다리가 아래라고 보고(방향은 화면 「돌리기」로 사람이 맞춘다 — 19 7-1) 다리 넷 · 몸통 · 목을 찾는다.
     넷을 못 찾으면 **틀린 자리에 찍지 않고** 그림 크기에 맞춘 기본 뼈대를 joints_found=False 로 돌려준다 —
     화면이 관절 고치기로 보낸다 (19 7장 ②)

시험: drawtale-ai experiments/quadruped (q_mask.py · auto_joints.py). 실제 아이 그림 4족형 8장 중 마스크 5장 좋음 · 2장 일부.
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from ai_server import analyzer as A

ORDER = ["hip", "torso", "neck",
         "right_hip_b", "right_knee_b", "right_foot_b", "left_hip_b", "left_knee_b", "left_foot_b",
         "right_hip_f", "right_knee_f", "right_foot_f", "left_hip_f", "left_knee_f", "left_foot_f",
         "tail", "tail_mid", "tail_tip"]
TAIL = ("tail", "tail_mid", "tail_tip")

# 사람 탐지 상자는 점수가 이보다 낮으면 버린다 (사람으로 안 보여도 자리는 잘 잡는다 — 실제 그림 8장 0.91~1.00)
MIN_BOX_SCORE = 0.05
# 다리를 몸에서 조금 떨어뜨려 그어도 닿은 것으로 볼 거리 (몸 크기 대비)
LEG_REACH = 0.04
# 상자를 거의 통째로 덮은 마스크는 안팎이 뒤집힌 것
FLIPPED_FILL = 0.92
# 이만큼 넓은 가로 덩어리는 다리가 아니라 몸통 (그림 폭 대비) — 배 선을 여기서 멈춘다
BODY_WIDTH = 0.35


def _detections(img: np.ndarray) -> list[dict]:
    r = requests.post(f"{A.TORCHSERVE_URL}/predictions/drawn_humanoid_detector",
                      files={"data": cv2.imencode(".png", img)[1].tobytes()}, verify=False, timeout=30)
    if r.status_code >= 300:
        raise RuntimeError(f"Detector request failed: HTTP {r.status_code}")
    out = json.loads(r.content)
    return out if isinstance(out, list) else []


def _ink(img: np.ndarray) -> np.ndarray:
    """선 · 칠한 곳 (create_mask 의 첫 단계와 같은 문턱)."""
    g = np.min(img, axis=2)
    return cv2.bitwise_not(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 115, 8)) > 0


def _place(img: np.ndarray) -> dict:
    boxes = [d for d in _detections(img) if d.get("score", 0) >= MIN_BOX_SCORE]
    if boxes:
        best = max(boxes, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        l, t, r, b = [round(v) for v in best["bbox"]]
        score = float(best["score"])
    else:
        ys, xs = np.where(_ink(img))
        if ys.size == 0:
            raise RuntimeError("NO_CHARACTER_DETECTED: nothing drawn")
        l, t = int(np.percentile(xs, 2)), int(np.percentile(ys, 2))
        r, b = int(np.percentile(xs, 98)), int(np.percentile(ys, 98))
        score = 0.0
    h, w = img.shape[:2]
    return {"left": max(0, l), "top": max(0, t), "right": min(w, r), "bottom": min(h, b), "score": score}


def build_mask(img: np.ndarray, box: dict) -> np.ndarray:
    """분석 크기 이미지 기준 마스크 (0/255). 종이색 깎기 전."""
    l, t, r, b = box["left"], box["top"], box["right"], box["bottom"]
    m = A.MASK_CROP_MARGIN
    y0, y1, x0, x1 = max(0, t - m), min(img.shape[0], b + m), max(0, l - m), min(img.shape[1], r + m)
    pad = A.MASK_PADDING
    padded = cv2.copyMakeBorder(img[y0:y1, x0:x1], pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    lines = _ink(padded)
    inside = np.zeros(lines.shape, bool)
    inside[pad + (t - y0):pad + (b - y0), pad + (l - x0):pad + (r - x0)] = True
    best, best_score = None, -1.0
    for it in A.MASK_CLOSE_ATTEMPTS:
        try:
            cm = A.create_mask(padded, close_iterations=it) > 0
        except RuntimeError:
            continue
        cover = (cm & lines & inside).sum() / max(1, (lines & inside).sum())
        flipped = (cm & inside).sum() / max(1, inside.sum()) > FLIPPED_FILL
        score = cover - (1.0 if flipped else 0.0)
        if score > best_score:
            best, best_score = cm, score
    if best is None:
        raise RuntimeError("NO_CHARACTER_DETECTED: Mask contour not found")
    full = np.zeros(img.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = best[pad:pad + y1 - y0, pad:pad + x1 - x0].astype(np.uint8) * 255
    return full


def restore_lines(img: np.ndarray, before: np.ndarray, after: np.ndarray, box: dict) -> np.ndarray:
    """종이색 깎기(사람 그림 기준)는 다리 사이 종이를 지우며 가는 선 다리까지 지운다 (검은 선 다리 · 꼬리).
    깎기 전 마스크 안의 선, 또는 몸 가까이의 선 중 남은 몸에 닿은 것을 조금 굵혀 되살린다. 모두 0/255, 같은 크기."""
    l, t, r, b = box["left"], box["top"], box["right"], box["bottom"]
    kept, was = after > 127, before > 127
    stroke = _ink(img)
    k = max(3, round(min(b - t, r - l) * 0.012))
    reach = max(k, round(min(b - t, r - l) * LEG_REACH))
    near = cv2.dilate(kept.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))) > 0
    lost = stroke & ~kept & (was | near)
    n, lab = cv2.connectedComponents(lost.astype(np.uint8))
    hit = np.unique(lab[near & lost])
    back = cv2.dilate(np.isin(lab, hit[hit > 0]).astype(np.uint8), np.ones((max(3, k // 2),) * 2, np.uint8)) > 0
    in_box = np.zeros(kept.shape, bool)
    in_box[t:b, l:r] = True
    out = kept | (back & in_box)
    # 가장 큰 덩어리만 — Meta 는 한 덩어리만 움직인다 (떨어진 조각은 렌더의 뼈대 띠가 다시 잇는다)
    n, lab, st, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8))
    if n > 2:
        out = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return out.astype(np.uint8) * 255


# ── 관절 ─────────────────────────────────────────────────────────

def _runs(row: np.ndarray, min_w: int) -> list[tuple[int, int]]:
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return [(a, b) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]) if b - a >= min_w]


def _legs(runs: list[tuple[int, int]], W: int) -> list[tuple[int, int]]:
    """다리 높이의 가로 덩어리를 다리 단위로 합친다 (실제 그림 11/4).
    - 거의 맞닿은 덩어리(틈 < 폭 0.5%) — 색칠이 끊겨 한 다리가 둘로 갈렸다 (악어 3px).
      1%면 붙여 그린 진짜 두 다리(회색 고양이 7px)까지 합쳐져 다리를 못 찾았다
    - 가는 선 둘이 가까이 — 속을 칠하지 않은 다리의 양쪽 테두리다 (선 고양이: 3다리가 4로 세어졌다)
    - 다리 바로 옆의 가는 선 — 그 다리의 테두리가 떨어져 나온 것 (소: 뒷다리 하나가 둘로 세어져 다리가 따로 붙었다)"""
    thin = W * 0.03
    out: list[tuple[int, int]] = []
    for a, b in runs:
        if out:
            pa, pb = out[-1]
            touching = a - pb < W * 0.005
            outline = pb - pa < thin and b - a < thin and a - pb < W * 0.1
            edge = (pb - pa < thin or b - a < thin) and a - pb < W * 0.04
            if touching or outline or edge:
                out[-1] = (pa, b)
                continue
        out.append((a, b))
    return out


def find_joints(mask: np.ndarray, facing: str | None = None) -> tuple[dict[str, tuple[float, float]], bool]:
    """다리가 아래인 마스크(bool)에서 관절 15개. (관절, 다리 넷을 찾았나).

    아래에서 위로 올라가며 가로로 끊긴 덩어리를 센다 — 둘 이상인 구간이 다리, 그 맨 위가 배 선.
    다리 바깥 양쪽 중 두꺼운 쪽이 머리(꼬리는 가늘다). 엉덩이는 뒷다리 위에서 몸 안쪽으로,
    어깨(torso)는 앞다리 위에서 등 쪽으로 — 「목 바로 밑」, 등 가운데면 걸을 때 몸통이 늘어난다 (10/20 시험)."""
    ys, xs = np.where(mask)
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    H, W = bottom - top + 1, right - left + 1
    min_w = max(2, round(W * 0.012))
    # 배 선: 아래에서 올라가다 몸통(그림 폭의 35%가 넘는 넓은 덩어리)이 나오는 줄. 덩어리가 하나가 되는 줄로 멈추면
    # 몸통 위에서도 머리와 꼬리가 양쪽에 따로 있는 그림(호랑이 · 공룡)은 배 선이 꼭대기까지 올라갔다 (실제 그림 11/4)
    belly = bottom
    for y in range(bottom, top, -1):
        row = _runs(mask[y], min_w)
        if any(b - a > W * BODY_WIDTH for a, b in row):
            break
        if len(row) >= 2:
            belly = y
    # 다리를 세는 높이: 한 줄만 보면 붙어 그린 두 다리(뒷다리끼리 · 발끝)가 하나로 세어졌다 (실제 그림 호랑이, 11/4).
    # 다리 구간의 여러 높이에서 세어 넷에 가장 가깝게 갈라진 줄을 쓴다 (같으면 위쪽 — 발끝보다 덜 붙는다)
    def leg_count(y: int) -> int:
        return len(_runs(mask[y], min_w))
    rows = [round(belly + (bottom - belly) * f) for f in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    probe = min(rows, key=lambda y: (abs(min(leg_count(y), 6) - 4), y))
    leg_runs = _legs(_runs(mask[probe], min_w), W)
    centers = [(a + b) / 2 for a, b in leg_runs]
    # 다리 둘도 찾은 것으로 — 옆모습은 반대편 다리를 그리지 않는 아이가 많다 (소 · 기린, 운영 11/4). 앞뒤로 떨어져 있고
    # 둘 다 다리 하나 굵기일 때만 (붙여 그린 다리 한 쌍이 넓적한 한 덩어리면 지금처럼 어른이 맞춘다). 각 다리는 한 쌍을 겹쳐 놓는다
    two = (len(leg_runs) == 2 and centers[1] - centers[0] >= W * 0.3
           and all(b - a <= W * 0.22 for a, b in leg_runs))
    found = (len(centers) >= 3 or two) and (bottom - belly) >= H * 0.08
    if len(centers) < 2:
        return template(mask, facing), False
    # 그래도 넷이 넘으면 가장 가까운 것끼리 합쳐 넷 이하로
    while len(centers) > 4:
        k = int(np.argmin(np.diff(centers)))
        centers[k:k + 2] = [(centers[k] + centers[k + 1]) / 2]
    # 앞뒤 나누기: 넷이면 둘 · 둘. 가장 넓은 틈으로 나누면 간격이 고른 다리(호랑이)가 하나 | 셋으로 갈려
    # 앞다리 하나를 두 번 쓰고 가운데 다리가 빠졌다 (실제 그림, 11/4). 셋 이하일 때만 넓은 틈으로
    cut = 2 if len(centers) == 4 else int(np.argmax(np.diff(centers))) + 1
    pair = lambda c: [c[0], c[0]] if len(c) == 1 else [c[0], c[-1]]  # noqa: E731
    side_a, side_b = pair(centers[:cut]), pair(centers[cut:])

    def thickness(x0: int, x1: int) -> float:
        cols = mask[:, max(left, x0):max(left, x0) + max(1, x1 - x0)]
        return float(cols.sum(axis=0).mean()) if cols.size else 0.0
    # 머리 쪽: 어른이 「얼굴이 반대쪽이에요」로 정해 주면 그대로 (11/4 — 굵은 쪽 추측은 악어처럼 꼬리가 굵으면 틀린다.
    # 눈 · 입 선으로 가리는 팀원 실험 방법도 크레파스 그림에선 못 가렸다: 실제 그림 7장 중 맞음 2)
    if facing in ("left", "right"):
        head_right = facing == "right"
    else:
        head_right = thickness(int(max(side_b)), right + 1) >= thickness(left, int(min(side_a)))
    back, front = (side_a, side_b) if head_right else (side_b, side_a)

    def foot(x: float) -> tuple[float, float]:
        col, y = int(round(x)), probe
        while y + 1 <= bottom and mask[y + 1, max(0, col - 2):col + 3].any():
            y += 1
            seg = np.where(mask[y, max(0, col - 15):col + 16])[0]
            if seg.size:
                col = max(0, col - 15) + int(seg.mean())
        return float(col), float(y)

    def body_top(x: float) -> float:
        col = mask[:, int(round(x))]
        return float(np.argmax(col)) if col.any() else float(top)

    j: dict[str, tuple[float, float]] = {}
    for side, legs in (("b", sorted(back, key=lambda c: c if head_right else -c)),
                       ("f", sorted(front, key=lambda c: -c if head_right else c))):
        for name, x in (("right", legs[1]), ("left", legs[0])):
            fx, fy = foot(x)
            j[f"{name}_hip_{side}"] = (x, float(belly))
            j[f"{name}_knee_{side}"] = ((x + fx) / 2, (belly + fy) / 2)
            j[f"{name}_foot_{side}"] = (fx, fy)
    toward_head = 1 if head_right else -1
    bx, fx_ = float(np.mean(back)), float(np.mean(front))
    span = abs(fx_ - bx)
    hip_x = bx + toward_head * span * 0.25
    j["hip"] = (hip_x, belly - (belly - body_top(hip_x)) * 0.15)
    j["torso"] = (fx_, belly - (belly - body_top(fx_)) * 0.55)
    neck_x = fx_ + toward_head * span * 0.08
    j["neck"] = (neck_x, belly - (belly - body_top(neck_x)) * 0.65)
    j.update(find_tail(mask, belly, leg_runs, head_right, j["hip"]))
    return j, found


def find_tail(mask: np.ndarray, belly: int, leg_runs: list[tuple[int, int]], head_right: bool,
              hip: tuple[float, float]) -> dict[str, tuple[float, float]]:
    """꼬리 점 셋(뿌리 · 가운데 · 끝). 꼬리가 없으면 셋 다 엉덩이에 겹친다 — 겹친 점은 움직여도 아무것도 안 끈다.

    팀원 실험(hankan-experiments animal-monster/quadruped.py find_head_tail, 10/20)의 방법:
    몸통 가운데에서 몸 한가운데 능선을 따라 꼬리 쪽 가장 먼 끝까지 가고, 끝에서 거슬러 오다 굵기가 끝 굵기의 2배를 넘는 곳이 뿌리.
    끝에서 몸 폭 12% 들어가도 몸통 굵기의 45%보다 가늘어야 꼬리다 (둥근 엉덩이는 금방 굵어져서 꼬리가 아니다).
    팀원 실험은 35% — 실제 그림에서 굵게 칠한 꼬리(선 고양이 0.42)를 놓쳤다. 꼬리 없는 그림은 0.53 · 0.67 (11/4)"""
    from skimage.graph import MCP_Geometric

    none = {n: hip for n in TAIL}
    body = mask.astype(np.uint8)
    # 다리를 뺀 몸 — 배 선 아래, 다리 사이 구간만 지운다 (뒷다리 바깥으로 늘어진 꼬리는 남긴다)
    lo, hi = min(a for a, _ in leg_runs), max(b for _, b in leg_runs)
    body[belly + 1:, lo:hi + 1] = 0
    dt = cv2.distanceTransform(body, cv2.DIST_L2, 5)
    band = np.zeros(dt.shape, bool)
    band[:, lo:hi + 1] = True
    c = np.unravel_index(int(np.argmax(np.where(band, dt, 0))), dt.shape)
    if dt[c] <= 0:
        return none
    D, _ = MCP_Geometric(np.where(body > 0, 1.0, np.inf)).find_costs([c])
    D = np.where(np.isfinite(D), D, -1)
    xx = np.arange(body.shape[1])[None, :]
    side = -1 if head_right else 1                           # 꼬리 쪽
    ref = lo if side < 0 else hi
    cand = np.where(((xx - ref) * side > 0) & (body > 0), D, -1)
    if cand.max() <= 0:
        return none
    tip = np.unravel_index(int(np.argmax(cand)), cand.shape)
    ridge = MCP_Geometric(np.where(body > 0, 1.0 / (dt + 1.0), np.inf))
    ridge.find_costs([c])
    path = np.array(ridge.traceback(tip), np.float32)[:, ::-1]     # (x, y), 몸통 가운데 → 끝
    w = np.array([dt[int(round(y)), int(round(x))] for x, y in path])
    tip_w = float(np.median(w[-max(3, len(w) // 4):]))
    root_i = 0
    for i in range(len(w) - 1, -1, -1):
        if w[i] > 2.0 * max(tip_w, 2.0):
            root_i = i
            break
    # 뿌리는 뒷다리 바깥에서 — 굵게 칠한 꼬리는 「굵기 2배」가 몸 한가운데까지 밀려, 흔들 때 엉덩이가 같이 휘었다 (선 고양이 11/4)
    outside = np.nonzero((path[:, 0] - ref) * side > 0)[0]
    if outside.size:
        root_i = max(root_i, int(outside[0]))
    xs = np.nonzero(body.any(0))[0]
    L = int(0.12 * (xs[-1] - xs[0]))
    if not (len(w) - root_i > L and len(w) > L and w[-L] < 0.45 * dt[c]):
        return none
    pts = (path[root_i], path[(root_i + len(w) - 1) // 2], path[-1])
    return {n: (float(x), float(y)) for n, (x, y) in zip(TAIL, pts)}


def template(mask: np.ndarray, facing: str | None = None) -> dict[str, tuple[float, float]]:
    """다리를 못 찾았을 때의 기본 뼈대 — 마스크 상자에 맞춘 옆모습 네 발 (머리는 오른쪽). 사람이 옮겨 맞춘다."""
    ys, xs = np.where(mask)
    l, t, r, b = xs.min(), ys.min(), xs.max(), ys.max()
    X = lambda f: float(l + (r - l) * f)  # noqa: E731
    Y = lambda f: float(t + (b - t) * f)  # noqa: E731
    j = {"hip": (X(0.32), Y(0.5)), "torso": (X(0.68), Y(0.4)), "neck": (X(0.74), Y(0.25))}
    for side, xs_ in (("b", (0.2, 0.32)), ("f", (0.66, 0.8))):
        for name, x in (("left", xs_[0]), ("right", xs_[1])):
            j[f"{name}_hip_{side}"] = (X(x), Y(0.6))
            j[f"{name}_knee_{side}"] = (X(x), Y(0.8))
            j[f"{name}_foot_{side}"] = (X(x), Y(0.98))
    j.update({n: j["hip"] for n in TAIL})                    # 꼬리 없음 — 있으면 사람이 끌어다 놓는다
    if facing == "left":                                     # 머리가 왼쪽이라고 정해 주면 좌우로 뒤집는다
        j = {n: (float(l + r - x), y) for n, (x, y) in j.items()}
    return j


def analyze(image_path: str, mask_path: str | None, facing: str | None = None) -> dict:
    """analyzer.analyze_image 와 같은 모양으로 돌려준다 (+ form · joints_found)."""
    started = time.perf_counter()
    original = cv2.imread(image_path, cv2.IMREAD_COLOR)            # 휴대폰 방향 정보를 반영한다
    if original is None:
        return {"success": False, "bbox": None, "joints": [], "message": "INVALID_IMAGE: could not read image"}
    scale = min(1.0, A.MAX_ANALYSIS_SIZE / max(original.shape[:2]))
    img = original if scale == 1.0 else cv2.resize(
        original, (round(original.shape[1] * scale), round(original.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    box = _place(img)
    if box["right"] <= box["left"] or box["bottom"] <= box["top"]:
        return {"success": False, "bbox": None, "joints": [], "message": "INVALID_BBOX: Character bounding box is invalid"}
    small = build_mask(img, box)
    before = cv2.resize(small, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    after = A.refine_mask(original, before)
    big_box = {k: round(v / scale) for k, v in box.items() if k != "score"}
    mask = restore_lines(original, before, after, big_box)
    if not (mask > 127).any():
        return {"success": False, "bbox": None, "joints": [], "message": "NO_CHARACTER_DETECTED: empty mask"}

    joints, found = find_joints(mask > 127, facing)
    saved = False
    if mask_path is not None:
        Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
        saved = bool(cv2.imwrite(mask_path, mask))
    return {
        "success": True,
        "form": "quadruped",
        "joints_found": found,
        "bbox": big_box,
        "joints": [{"name": n, "x": round(joints[n][0]), "y": round(joints[n][1]), "score": None} for n in ORDER],
        # 사람 모델 점수가 아니다 — 다리를 찾았으면 상자 점수, 못 찾았으면 낮게(화면이 어른 확인을 권한다)
        "confidence": round(box["score"] if found else min(box["score"], 0.2), 3),
        "mask": {"width": original.shape[1], "height": original.shape[0], "saved": saved, "covers": "full_image"},
        "message": "Analysis complete" if found else "JOINTS_NOT_FOUND: legs not found, template skeleton",
        "model_version": A.MODEL_VERSION,
        "pipeline_version": A.PIPELINE_VERSION,
        "coordinate_space": A.COORDINATE_SPACE,
        "processing_time_ms": round((time.perf_counter() - started) * 1000),
    }
