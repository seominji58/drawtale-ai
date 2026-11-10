"""눈 찾기: 사람 아닌 그림(날개형 · 사슬형 · 지느러미형)의 머리 쪽을 정할 때 같이 쓴다 (2026-11-10).

몸짓 연출을 실제 그림으로 시험하니 새 · 뱀 · 물고기의 머리가 꼬리 · 날개 끝 · 등지느러미에 잡혔다 (관절은 「찾음」이라
보호자에게 고치라고 하지도 않았다). 원인은 눈을 넓이로만 골라서였다. 꼬리 세모의 테두리처럼 작은 선 고리도 눈이 됐고,
가는 선으로 그린 뱀은 몸을 깎아 보다가 눈까지 지워졌다.
아이들이 그리는 눈은 「작고 · 속이 찬 · 동그란 · 아주 어두운 점」이다. 선 고리(속이 빔) · 선 조각(길쭉함) · 칠한 면(큼)은 눈이 아니다.
시험: experiments/head-check (그림마다 머리 상자를 적어 두고 head 가 그 안인지)
"""

import cv2
import numpy as np

# 아주 어두운 곳 (Lab L, 0~255). 파랑 · 초록 테두리(L 50~70)는 빼고 검정 눈만
EYE_DARK = 60
# 눈 크기 상한 (그림 상자 긴 변 대비) · 속이 찬 정도 하한 (넓이 / 둘러싼 네모) · 길쭉함 상한
EYE_MAX_SIDE = 0.06
EYE_MIN_FILL = 0.5
EYE_MAX_ASPECT = 2.0


def find_eyes(img: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """그림 안 눈 자리 (x, y). 큰 것부터. img 는 BGR, mask 는 그림 영역(참 · 거짓 또는 0 · 255)."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() < 50:
        return []
    ys, xs = np.nonzero(m)
    side = max(int(np.ptp(xs)), int(np.ptp(ys)), 1)
    # 눈 흰자 같은 구멍도 그림 안으로 (테두리 안쪽에 떨어진 점)
    flood = m.copy()
    cv2.floodFill(flood, np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8), (0, 0), 1)
    inside = cv2.dilate(m | (1 - flood), np.ones((5, 5), np.uint8)) > 0
    light = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    dark = ((light < EYE_DARK) & inside).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(dark, 8)
    out = []
    for i in range(1, n):
        w, h, a = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA]
        if a < 6 or max(w, h) > EYE_MAX_SIDE * side:
            continue
        if a / float(w * h) < EYE_MIN_FILL or max(w, h) > EYE_MAX_ASPECT * max(1, min(w, h)):
            continue
        out.append((a, (float(cen[i][0]), float(cen[i][1]))))
    return [c for _, c in sorted(out, key=lambda e: -e[0])]


def eye_centre(eyes: list[tuple[float, float]], near: float) -> tuple[float, float] | None:
    """가장 큰 눈과 그 둘레(near 안)의 눈들 가운데, 두 눈이면 그 사이. 눈이 없으면 None"""
    if not eyes:
        return None
    x0, y0 = eyes[0]
    group = [e for e in eyes if np.hypot(e[0] - x0, e[1] - y0) <= near]
    return float(np.mean([e[0] for e in group])), float(np.mean([e[1] for e in group]))
