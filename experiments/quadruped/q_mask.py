"""4족형 마스크 — 사람 탐지에 기대지 않고 (시험용, 2026-11-04, pt_docs 19 9-1).

지금 분석은 사람 탐지 모델이 사람으로 봐야 마스크를 만든다. 실제 아이 그림 4족형 8장 중 악어는 LOW_CONFIDENCE,
기린 + 작은 새는 MULTIPLE 로 거절됐고, 검은 선으로만 그린 다리는 마스크에서 빠졌다.

  1. 자리: 탐지 상자를 점수와 상관없이 받는다. 여럿이면 가장 큰 상자(주인공). 없으면 종이 위 선의 범위
  2. 마스크: 서비스와 같은 선 찾기(create_mask) — 틈 메우기 강도를 바꿔 가며, 사람 관절 대신
     「상자 안 선이 마스크에 얼마나 들어오나」로 고른다 (상자를 통째로 덮으면 안팎이 뒤집힌 것)
  3. 선만 그린 부분: 마스크에 닿아 있는데 빠진 가는 선은 굵혀서 더한다
  4. 종이색 깎기는 서비스와 같게 (refine_mask)

    python q_mask.py real/02.orig.jpg real/04.orig.jpg …   → real/{번호}.qmask.png, out/qmask_sheet.png
AI 서버 이미지 안에서, 모델 서버(TORCHSERVE_URL)가 떠 있을 때.
"""

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, "/app")
from ai_server import analyzer as A  # noqa: E402

HERE = Path(__file__).parent


def detections(img: np.ndarray) -> list[dict]:
    r = requests.post(f"{os.environ['TORCHSERVE_URL']}/predictions/drawn_humanoid_detector",
                      files={"data": cv2.imencode(".png", img)[1].tobytes()}, timeout=60)
    out = json.loads(r.content) if r.status_code < 300 else []
    return out if isinstance(out, list) else []


def ink(img: np.ndarray) -> np.ndarray:
    """선 · 칠한 곳 (create_mask 의 첫 단계와 같은 문턱)."""
    g = np.min(img, axis=2)
    return cv2.bitwise_not(cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 8)) > 0


def place(img: np.ndarray) -> tuple[dict, str]:
    dets = [d for d in detections(img) if d.get("score", 0) >= 0.05]
    if dets:
        best = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
        l, t, r, b = [round(v) for v in best["bbox"]]
        return {"left": l, "top": t, "right": r, "bottom": b}, f"탐지 {best['score']:.2f} (후보 {len(dets)})"
    ys, xs = np.where(ink(img))
    return {"left": int(np.percentile(xs, 2)), "top": int(np.percentile(ys, 2)),
            "right": int(np.percentile(xs, 98)), "bottom": int(np.percentile(ys, 98))}, "선 범위"


def mask_of(img: np.ndarray, box: dict) -> tuple[np.ndarray, str]:
    l, t, r, b = box["left"], box["top"], box["right"], box["bottom"]
    m = A.MASK_CROP_MARGIN
    y0, y1, x0, x1 = max(0, t - m), min(img.shape[0], b + m), max(0, l - m), min(img.shape[1], r + m)
    crop = img[y0:y1, x0:x1]
    pad = A.MASK_PADDING
    padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    lines = ink(padded)
    inside = np.zeros(lines.shape, bool); inside[pad + (t - y0):pad + (b - y0), pad + (l - x0):pad + (r - x0)] = True
    box_area = inside.sum()
    best, best_s, note = None, -1.0, ""
    for it in A.MASK_CLOSE_ATTEMPTS:
        try:
            cm = A.create_mask(padded, close_iterations=it) > 0
        except RuntimeError:
            continue
        cover = (cm & lines & inside).sum() / max(1, (lines & inside).sum())
        fill = (cm & inside).sum() / box_area
        s = cover - (1.0 if fill > 0.92 else 0.0)        # 상자를 통째로 덮으면 안팎이 뒤집힌 것
        if s > best_s:
            best, best_s, note = cm, s, f"틈메우기 {it} · 선 {cover:.0%} · 상자 {fill:.0%}"
    if best is None:
        raise RuntimeError("마스크를 못 만들었다")
    # 3. 마스크에 닿은 가는 선(선만 그린 다리 · 꼬리)
    k = max(3, round(min(crop.shape[:2]) * 0.012))
    near = cv2.dilate(best.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    extra = lines & ~best & inside
    n, lab = cv2.connectedComponents(extra.astype(np.uint8))
    touch = np.unique(lab[near & extra])
    add = np.isin(lab, touch[touch > 0])
    add = cv2.dilate(add.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    merged = best | (add & inside)
    full = np.zeros(img.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = (merged[pad:pad + crop.shape[0], pad:pad + crop.shape[1]] * 255).astype(np.uint8)
    before = full > 127
    full = A.refine_mask(img, full)
    # 종이색 깎기(사람 그림 기준)는 다리 사이 종이를 지우면서 가는 선 다리까지 지운다 (검은 선 다리 · 꼬리).
    # 깎기 전 마스크 안의 선 중 남은 몸에 닿은 것을 조금 굵혀 되살린다
    kept = full > 127
    stroke = np.zeros(img.shape[:2], bool)
    stroke[y0:y1, x0:x1] = lines[pad:pad + crop.shape[0], pad:pad + crop.shape[1]]
    lost = stroke & before & ~kept
    n, lab = cv2.connectedComponents(lost.astype(np.uint8))
    # 아이는 다리를 몸통에서 조금 떨어뜨려 긋기도 한다 — 몸 크기의 4% 안이면 닿은 것으로 본다
    reach = max(k, round(min(crop.shape[:2]) * 0.04))
    near = cv2.dilate(kept.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach))) > 0
    lost = lost | (stroke & ~kept & near)     # 처음 마스크에 없던 가는 선도 몸 가까이면
    n, lab = cv2.connectedComponents(lost.astype(np.uint8))
    hit = np.unique(lab[near & lost]); back = np.isin(lab, hit[hit > 0])
    back = cv2.dilate(back.astype(np.uint8), np.ones((max(3, k // 2),) * 2, np.uint8)) > 0
    # 몸과 떨어진 틈은 한 번 이어 준다 (메시가 한 덩어리여야 움직인다 — 렌더의 뼈대 띠가 마저 잇는다)
    box_only = np.zeros(img.shape[:2], bool); box_only[t:b, l:r] = True
    joined = cv2.morphologyEx(((kept | back) & box_only).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((reach, reach), np.uint8)) > 0
    full = np.where(kept | (back & box_only) | (joined & ~kept & cv2.dilate(back.astype(np.uint8), np.ones((reach, reach), np.uint8)).astype(bool)), 255, 0).astype(np.uint8)
    note_back = f" · 되살림 {int((back & box_only & ~kept).sum())}px"
    n, lab, st, _ = cv2.connectedComponentsWithStats((full > 127).astype(np.uint8))
    if n > 2:
        full = np.where(lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
    return full, note + f" · 선 더함 {int(add.sum())}px" + note_back


def main() -> None:
    tiles = []
    for path in sys.argv[1:]:
        name = Path(path).name.split(".")[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)                # 휴대폰 방향 정보를 반영한다 (서비스와 같다)
        s = A.MAX_ANALYSIS_SIZE / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
        box, how = place(img)
        try:
            m, note = mask_of(img, box)
        except RuntimeError as e:
            print(name, how, "실패", e); continue
        cv2.imwrite(str(HERE / "real" / f"{name}.qmask.png"), m)
        view = img.copy(); red = np.zeros_like(img); red[..., 2] = 255
        view = np.where((m > 127)[..., None], (0.6 * view + 0.4 * red).astype(np.uint8), view)
        cv2.rectangle(view, (box["left"], box["top"]), (box["right"], box["bottom"]), (255, 120, 0), 2)
        tile = cv2.resize(view, (round(view.shape[1] * 400 / max(view.shape[:2])), round(view.shape[0] * 400 / max(view.shape[:2]))))
        canvas = np.full((420, 410, 3), 255, np.uint8); canvas[20:20 + tile.shape[0], 5:5 + tile.shape[1]] = tile
        cv2.putText(canvas, name, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tiles.append(canvas)
        print(name, how, "|", note)
    rows = [np.hstack(tiles[i:i + 4] + [np.full((420, 410, 3), 255, np.uint8)] * (4 - len(tiles[i:i + 4]))) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(HERE / "out" / "qmask_sheet.png"), np.vstack(rows))


if __name__ == "__main__":
    main()
