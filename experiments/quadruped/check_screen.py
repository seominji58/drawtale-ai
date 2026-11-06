"""화면에 그리기(800×1000 흰 바탕) 그림 영역 검사: 4족형 · 사슬형 · 지느러미형이 쓰는 quadruped.cut_out (11/6).

옆으로 길게 그린 강아지를 사람 탐지 상자가 앞 절반만 잡아 몸 안이 지워지고 테두리만 움직였다. 그림을 만들어
cut_out 마스크가 정답 실루엣과 IoU 0.9 이상 겹치는지 본다. AI 컨테이너 안에서:

    docker cp experiments/quadruped/check_screen.py drawtale-ai-ai-1:/app/check_screen.py
    docker exec -w /app -e PYTHONPATH=/app drawtale-ai-ai-1 python check_screen.py
"""
import os
import sys
import tempfile

import cv2
import numpy as np

from ai_server import quadruped as Q

W, H = 800, 1000
INK = (51, 55, 58)          # 앱의 진한 선 #3A3733 (BGR)
MIN_IOU = 0.9


def P(pts):
    return np.array([[int(x * W), int(y * H)] for x, y in pts], np.int32)


def draw(pts, fill=None, extra=(), thick=14):
    img = np.full((H, W, 3), 255, np.uint8)
    gt = np.zeros((H, W), np.uint8)
    p = P(pts)
    cv2.fillPoly(gt, [p], 255)
    if fill is not None:
        cv2.fillPoly(img, [p], fill)
    cv2.polylines(img, [p], True, INK, thick, cv2.LINE_AA)
    cv2.polylines(gt, [p], True, 255, thick)
    for e in extra:
        q = P(e)
        cv2.polylines(img, [q], False, INK, thick, cv2.LINE_AA)
        cv2.polylines(gt, [q], False, 255, thick)
    return img, gt > 0


ORANGE, BROWN, BLUE, PINK, YELLOW = (32, 128, 240), (40, 90, 140), (220, 160, 60), (170, 140, 230), (60, 190, 230)
DOG = [(0.16, 0.42), (0.70, 0.42), (0.72, 0.34), (0.76, 0.28), (0.86, 0.28), (0.92, 0.34), (0.90, 0.42), (0.82, 0.46),
       (0.80, 0.62), (0.79, 0.84), (0.71, 0.84), (0.70, 0.64), (0.66, 0.64), (0.65, 0.84), (0.57, 0.84), (0.56, 0.64),
       (0.38, 0.64), (0.37, 0.84), (0.29, 0.84), (0.28, 0.64), (0.25, 0.64), (0.24, 0.84), (0.16, 0.84), (0.15, 0.62)]
GIRAFFE = [(0.10, 0.55), (0.62, 0.55), (0.70, 0.30), (0.74, 0.12), (0.86, 0.10), (0.88, 0.18), (0.80, 0.22), (0.76, 0.55),
           (0.75, 0.70), (0.74, 0.90), (0.67, 0.90), (0.66, 0.72), (0.60, 0.72), (0.59, 0.90), (0.52, 0.90), (0.51, 0.72),
           (0.24, 0.72), (0.23, 0.90), (0.16, 0.90), (0.15, 0.72), (0.12, 0.72), (0.11, 0.90), (0.04, 0.90), (0.05, 0.70)]
s = np.linspace(0, 1, 60)
WORM = ([(0.08 + 0.84 * t, 0.45 + 0.08 * np.sin(t * 3 * np.pi)) for t in s]
        + [(0.08 + 0.84 * t, 0.55 + 0.08 * np.sin(t * 3 * np.pi)) for t in s[::-1]])
FISH = [(0.50 + 0.36 * np.cos(a), 0.50 + 0.14 * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 50)[:-1]]
FIN = [[(0.14, 0.50), (0.04, 0.40), (0.04, 0.60), (0.14, 0.50)]]

CASES = {
    "넓은 강아지 칠함": draw(DOG, ORANGE, [[(0.16, 0.46), (0.08, 0.36)]]),
    "넓은 강아지 선만": draw(DOG, None, [[(0.16, 0.46), (0.08, 0.36)]]),
    "왼쪽 보는 넓은 강아지": draw([(1 - x, y) for x, y in DOG], BROWN, [[(0.84, 0.46), (0.92, 0.36)]]),
    "작은 강아지": draw([(0.35 + (x - 0.5) * 0.45, 0.55 + (y - 0.55) * 0.45) for x, y in DOG], ORANGE),
    "목 긴 기린": draw(GIRAFFE, YELLOW),
    "긴 지렁이": draw(WORM, PINK),
    "넓은 물고기 선만": draw(FISH, None, FIN),
    "넓은 물고기 칠함": draw(FISH, BLUE, FIN),
}

bad = 0
with tempfile.TemporaryDirectory() as d:
    for name, (img, gt) in CASES.items():
        path = os.path.join(d, "x.png")
        cv2.imwrite(path, img)
        cut = Q.cut_out(path)
        if "error" in cut:
            print(f"✗ {name}: {cut['error']}")
            bad += 1
            continue
        m = cut["mask"] > 127
        iou = float((m & gt).sum()) / max(1, (m | gt).sum())
        ok = iou >= MIN_IOU
        bad += not ok
        print(f"{'✓' if ok else '✗'} {name}: IoU {iou:.2f}")
print(f"{len(CASES) - bad} / {len(CASES)} 통과")
sys.exit(1 if bad else 0)
