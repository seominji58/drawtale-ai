"""네 발 그림의 관절 15개를 마스크만 보고 찾는다 (시험용, 2026-11-04).

사람 관절 모델은 강아지 · 공룡도 사람으로 통과시킨다(신뢰도 0.93 · 0.95, 꼬리에 팔이 찍힘). 네 발은 모델 없이
그림 모양으로 찾는다 — pt_docs 15 8장 「중심선으로 하는 길」의 네 발 판. 어른은 미리 찍힌 점을 확인만 한다.

  1. 다리: 아래에서 위로 올라가며 가로로 끊긴 덩어리를 센다. 둘 이상인 구간이 다리, 그 맨 위가 배 선
  2. 앞뒤: 다리 바깥 양쪽 중 두꺼운 쪽이 머리(꼬리는 가늘다)
  3. 다리마다 위(배 선) · 무릎(가운데) · 발끝. 앞뒤로 둘씩 — 다리가 셋 이하로 보이면 같은 자리를 둘로 쓴다
  4. 엉덩이 = 뒷다리 위에서 몸 안쪽으로 조금, 몸통 = 앞다리 위에서 등 쪽으로(「목 바로 밑」 — README 2장),
     목 = 머리 쪽으로 더 위

    python auto_joints.py              → 시험 그림 셋의 자동 · 정답 비교 (out/auto_*.png, 오차 표)
    python auto_joints.py <그림.png>    → 좌표 문자열 (make_char.py 에 그대로)
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

HERE = Path(__file__).parent
ORDER = ["hip", "torso", "neck",
         "right_hip_b", "right_knee_b", "right_foot_b", "left_hip_b", "left_knee_b", "left_foot_b",
         "right_hip_f", "right_knee_f", "right_foot_f", "left_hip_f", "left_knee_f", "left_foot_f"]


def mask_of(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("RGBA"))
    if (a[..., 3] < 128).mean() > 0.05:              # 이미 잘라 낸 그림(투명 바탕)
        m = a[..., 3] >= 128
    else:                                              # 흰 종이
        m = (255 - a[..., :3].astype(np.int16)).max(axis=2) > 40
        m = ndimage.binary_closing(m, np.ones((5, 5)))
    lab, n = ndimage.label(m)
    m = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    return ndimage.binary_fill_holes(m)


def runs(row: np.ndarray, min_w: int) -> list[tuple[int, int]]:
    """한 줄에서 마스크가 이어진 구간들 (너무 좁은 것은 뺀다)."""
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    out = list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
    return [(a, b) for a, b in out if b - a >= min_w]


def legs_score(mask: np.ndarray) -> float:
    """이 방향이 「다리가 아래」인지 — 아래쪽에 다리로 보이는 덩어리가 넷에 가깝고, 다리 구간이 적당히 길수록 높다."""
    ys, xs = np.where(mask)
    top, bottom = ys.min(), ys.max()
    H, W = bottom - top + 1, xs.max() - xs.min() + 1
    min_w = max(2, round(W * 0.012))
    belly = bottom
    for y in range(bottom, top, -1):
        if len(runs(mask[y], min_w)) >= 2:
            belly = y
        elif y < bottom - H * 0.08:
            break
    leg_h = (bottom - belly) / H
    n = len(runs(mask[round(belly + (bottom - belly) * 0.35)], min_w))
    if n < 2 or leg_h < 0.08:
        return -10.0
    return -abs(min(n, 6) - 4) + (1.0 if 0.15 <= leg_h <= 0.7 else 0.0) + (0.5 if W >= H * 0.8 else 0.0)


def find_any_way(mask: np.ndarray) -> tuple[dict[str, tuple[float, float]], int]:
    """사진이 옆으로 돌아가 있어도 (종이를 세로로 놓고 찍은 가로 그림) — 네 방향 중 다리가 가장 잘 보이는 방향으로 찾고
    좌표를 원래 사진 기준으로 돌려 준다. 돌린 횟수(90도 단위, 반시계)도 돌려준다."""
    best = max(range(4), key=lambda k: legs_score(np.rot90(mask, k)))
    m = np.rot90(mask, best)
    j = find(m)
    # np.rot90(k=1): 원래 (r, c) → 새 (W-1-c, r). 거꾸로 새 (r', c') → 원래 (c', W-1-r')
    out = {}
    for name, (x, y) in j.items():
        r, c = y, x
        shape = m.shape
        for _ in range(best):
            H2, W2 = shape
            r, c = c, H2 - 1 - r       # 새 → 이전 단계
            shape = (W2, H2)
        out[name] = (c, r)
    return out, best


def find(mask: np.ndarray) -> dict[str, tuple[float, float]]:
    ys, xs = np.where(mask)
    top, bottom, left, right = ys.min(), ys.max(), xs.min(), xs.max()
    H, W = bottom - top + 1, right - left + 1
    min_w = max(2, round(W * 0.012))

    # 1. 배 선: 아래에서 올라가며 덩어리가 둘 이상인 줄이 끊기는 곳
    belly = bottom
    for y in range(bottom, top, -1):
        if len(runs(mask[y], min_w)) >= 2:
            belly = y
        elif y < bottom - H * 0.08:      # 맨 아래 몇 줄(발끝이 하나로 붙은 곳)은 지나간다
            break
    leg_h = bottom - belly
    probe = round(belly + leg_h * 0.35)
    legs = sorted(runs(mask[probe], min_w), key=lambda r: r[0])
    centers = [(a + b) / 2 for a, b in legs]
    if len(centers) < 2:
        centers = [left + W * 0.25, left + W * 0.75]

    # 앞뒤 둘로 나누기: 가장 큰 틈에서
    if len(centers) >= 2:
        gaps = np.diff(centers)
        cut = int(np.argmax(gaps)) + 1
        side_a, side_b = centers[:cut], centers[cut:]
    pick = lambda c: (c + c)[:2] if len(c) == 1 else [c[0], c[-1]]  # noqa: E731 — 하나면 둘로
    side_a, side_b = pick(side_a), pick(side_b)

    # 2. 머리 쪽: 다리 바깥 양쪽의 평균 두께 (세로로 마스크가 몇 칸인지)
    def thickness(x0: int, x1: int) -> float:
        cols = mask[:, max(left, x0):max(left, x0) + max(1, x1 - x0)]
        return float(cols.sum(axis=0).mean()) if cols.size else 0.0
    t_left = thickness(left, int(min(side_a)))
    t_right = thickness(int(max(side_b)), right + 1)
    head_right = t_right >= t_left
    back, front = (side_a, side_b) if head_right else (side_b, side_a)

    def foot(x: float) -> tuple[float, float]:
        """그 다리를 따라 내려간 맨 아래 점."""
        col = int(round(x))
        y = probe
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
    for side, (outer, inner) in (("b", sorted(back, key=lambda c: c if head_right else -c)),
                                  ("f", sorted(front, key=lambda c: -c if head_right else c))):
        for name, x in (("right", inner), ("left", outer)):
            fx, fy = foot(x)
            j[f"{name}_hip_{side}"] = (x, float(belly))
            j[f"{name}_knee_{side}"] = ((x + fx) / 2, (belly + fy) / 2)
            j[f"{name}_foot_{side}"] = (fx, fy)

    toward_head = 1 if head_right else -1
    bx, fx_ = float(np.mean(back)), float(np.mean(front))
    span = abs(fx_ - bx)
    hip_x = bx + toward_head * span * 0.25
    j["hip"] = (hip_x, belly - (belly - body_top(hip_x)) * 0.15)
    tor_x = fx_
    j["torso"] = (tor_x, belly - (belly - body_top(tor_x)) * 0.55)
    neck_x = fx_ + toward_head * span * 0.08
    j["neck"] = (neck_x, belly - (belly - body_top(neck_x)) * 0.65)   # 머리 꼭대기가 아니라 머리 아래쪽
    return j


SAMPLES = {
    "dog": (HERE / "samples/dog.png", "330,450 660,350 670,270 380,440 378,515 376,578 225,430 223,490 221,540 800,430 800,490 800,535 625,440 623,520 621,590"),
    "gong": (HERE / "samples/gong.png", "400,570 700,500 715,450 460,580 462,670 465,775 310,520 305,650 302,790 765,600 775,700 790,820 650,630 650,740 650,865"),
    "char6": (Path("/opt/AnimatedDrawings/examples/characters/char6/texture.png")
              if Path("/opt/AnimatedDrawings").exists()
              else HERE.parents[1] / "third_party/AnimatedDrawings/examples/characters/char6/texture.png",
              "238,264 453,181 455,121 139,266 138,290 132,324 82,249 65,279 47,308 442,245 455,284 458,320 364,266 364,305 360,336"),
}


def as_string(j: dict) -> str:
    return " ".join(f"{round(j[n][0])},{round(j[n][1])}" for n in ORDER)


def compare() -> None:
    (HERE / "out").mkdir(exist_ok=True)
    print(f"{'그림':6s} {'평균 오차(몸 길이 %)':>18s}  가장 틀린 관절")
    for name, (path, truth) in SAMPLES.items():
        img = Image.open(path).convert("RGBA")
        # 선만 그린 그림은 밝기로는 몸 안이 안 채워진다 — 서비스는 AI 마스크(몸 안까지)를 쓰니 있으면 그것으로
        given = path.with_name("mask.png")
        m = (np.asarray(Image.open(given).convert("L")) > 127) if given.exists() else mask_of(img)
        auto = find(m)
        gt = dict(zip(ORDER, [tuple(map(float, p.split(","))) for p in truth.split()]))
        ys, xs = np.where(m)
        size = float(max(np.ptp(xs), np.ptp(ys)))
        # 왼쪽 · 오른쪽 다리는 그림에서 구별이 안 되니 둘을 바꿔 더 가까운 쪽으로 잰다
        def err(n: str) -> float:
            a, g = np.array(auto[n]), np.array(gt[n])
            other = n.replace("right", "@").replace("left", "right").replace("@", "left")
            alt = np.linalg.norm(np.array(auto[other]) - g) if other != n else 1e9
            return min(float(np.linalg.norm(a - g)), alt) / size * 100
        errs = {n: err(n) for n in ORDER}
        worst = max(errs, key=errs.get)
        print(f"{name:6s} {np.mean(list(errs.values())):17.1f}%  {worst} {errs[worst]:.0f}%")
        bg = Image.new("RGBA", img.size, "white"); bg.alpha_composite(img); d = ImageDraw.Draw(bg)
        for n in ORDER:
            for (x, y), c in ((gt[n], "blue"), (auto[n], "red")):
                d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=c)
        bg.convert("RGB").save(HERE / "out" / f"auto_{name}.png")
        print("   자동:", as_string(auto))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(as_string(find(mask_of(Image.open(sys.argv[1])))))
    else:
        compare()
