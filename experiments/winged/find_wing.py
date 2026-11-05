"""날개형 부위 찾기 시험 (11/5). 선 그림의 닫힌 칸으로 나눈다 — 머리(눈이 든 칸) · 몸통(가장 아래 큰 칸) · 날개(몸통에 붙은 나머지 큰 칸).
몸통 칸에 완전히 둘러싸인 날개 칸은 몸 안에 그린 날개(퍼덕이지 않음).  python find_wing.py b1.png … → 그림.parts.png"""
import sys
import cv2
import numpy as np

def parts(path):
    img = cv2.imread(path)
    ink = (img.min(axis=2) < 160).astype(np.uint8)
    # 선 굵기 — 날개 둘레를 볼 때 선 너머까지 (선만 보면 바깥 종이에 닿지 못해 솟은 날개를 몸 안 날개로 봤다)
    lw = max(3.0, 2 * float(cv2.distanceTransform(ink, cv2.DIST_L2, 3).max()))
    # 날개 선과 몸 선이 만나는 곳의 작은 틈을 막는다 — 틈으로 날개 칸이 바깥 종이와 이어져 칸이 안 잡혔다
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8))
    h, w = ink.shape
    paper = (1 - ink).astype(np.uint8)
    n, lab, st, cen = cv2.connectedComponentsWithStats(paper, 4)
    outside = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])).tolist())
    total = float(sum(st[i, cv2.CC_STAT_AREA] for i in range(1, n) if i not in outside))
    cells = [i for i in range(1, n) if i not in outside and st[i, cv2.CC_STAT_AREA] > 0.03 * total]
    # 눈 = 칸 안에 갇힌 작은 잉크 덩어리
    m_ink, lab_ink, st_ink, cen_ink = cv2.connectedComponentsWithStats(ink, 8)
    big_ink = 1 + int(np.argmax(st_ink[1:, cv2.CC_STAT_AREA]))
    eyes = [cen_ink[j] for j in range(1, m_ink) if j != big_ink and st_ink[j, cv2.CC_STAT_AREA] < 0.01 * total]
    head = None
    for e in eyes:
        c = lab[int(e[1]), int(e[0])]
        near = [i for i in cells if (lab[max(0, int(e[1]) - 25):int(e[1]) + 25, max(0, int(e[0]) - 25):int(e[0]) + 25] == i).any()]
        if near:
            head = max(near, key=lambda i: st[i, cv2.CC_STAT_AREA]); break
    rest = [i for i in cells if i != head]
    body = max(rest, key=lambda i: st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] + 0.0001 * st[i, cv2.CC_STAT_AREA])
    def ring(i, r=None):
        r = r or int(2.5 * lw)
        return (cv2.dilate((lab == i).astype(np.uint8), np.ones((r, r), np.uint8)) > 0) & (lab != i)
    wings, inner = [], []
    for i in rest:
        if i == body:
            continue
        around = lab[ring(i)]
        around = around[around > 0]
        touches_body = (around == body).any()
        touches_out = np.isin(around, list(outside)).any()
        if not touches_body:
            continue
        (inner if not touches_out else wings).append(i)
    ov = img.copy()
    col = {head: (120, 220, 120), body: (255, 200, 120)}
    for i in wings: col[i] = (60, 60, 255)
    for i in inner: col[i] = (200, 120, 255)
    for i, c in col.items():
        if i is not None:
            ov[lab == i] = (ov[lab == i] * 0.4 + np.array(c) * 0.6).astype(np.uint8)
    cv2.imwrite(path.replace(".png", ".parts.png"), ov)
    # 관절: 몸통 · 머리 가운데, 날개마다 뿌리(몸통에 붙은 곳) · 끝(뿌리에서 가장 먼 곳) · 가운데
    import json
    def centre(i):
        ys, xs = np.nonzero(lab == i); return float(xs.mean()), float(ys.mean())
    bx, by = centre(body)
    J = {"body": (bx, by), "head": centre(head) if head is not None else (bx, by)}
    found = []
    for i in wings[:2]:
        ys, xs = np.nonzero(lab == i)
        touch = ring(i) & (lab == body)
        ty, tx = np.nonzero(cv2.dilate(touch.astype(np.uint8), np.ones((int(3 * lw), int(3 * lw)), np.uint8)) & (lab == i))
        root = (float(tx.mean()), float(ty.mean())) if len(tx) else (float(xs.mean()), float(ys.mean()))
        d = np.hypot(xs - root[0], ys - root[1]); k = int(np.argmax(d))
        tip = (float(xs[k]), float(ys[k]))
        found.append((root, ((root[0] + tip[0]) / 2, (root[1] + tip[1]) / 2), tip))
    if not found:
        found = [((bx, by),) * 3]
    if len(found) == 1:
        found.append(found[0])
    for n_, (r, m_, t) in zip((1, 2), found):
        J[f"wing{n_}_root"], J[f"wing{n_}_mid"], J[f"wing{n_}_tip"] = r, m_, t
    joints = [{"name": k_, "x": round(v[0]), "y": round(v[1])} for k_, v in J.items()]
    json.dump(joints, open(path.replace(".png", ".joints.json"), "w"))
    filled = ((lab > 0) & ~np.isin(lab, list(outside))) | (ink > 0)
    filled = filled.astype(np.uint8)
    hh, ww = filled.shape
    fl = filled.copy(); cv2.floodFill(fl, np.zeros((hh + 2, ww + 2), np.uint8), (0, 0), 1)
    filled = filled | (1 - fl)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(filled, 8)
    big = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
    cv2.imwrite(path.replace(".png", ".mask.png"), np.where(lab2 == big, 255, 0).astype(np.uint8))
    for j in joints:
        cv2.circle(ov, (j["x"], j["y"]), 10, (0, 0, 0), -1)
    cv2.imwrite(path.replace(".png", ".parts.png"), ov)
    print(path, "칸", len(cells), "머리", head is not None, "날개(밖)", len(wings), "날개(몸 안)", len(inner))

for p in sys.argv[1:]:
    parts(p)
