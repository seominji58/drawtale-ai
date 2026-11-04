"""큰 얼굴을 원본 그대로 옮겨 붙이기 (2026-11-04).

머리가 몸보다 크고 목 없이 몸에 붙은 그림은 Meta 가 얼굴 전체를 목 관절 한 점으로만 붙잡아, 움직일 때 얼굴이
콩팥 모양으로 휘었다 (사용자 「애니메이션까지 적용되니까 얼굴도 늘어지고」). 관절 자리를 옮기는 방법(목을 턱으로 ·
몸통 위로, 몸통 움직임 줄이기)은 셋 다 효과가 없거나 머리가 기울어 더 나빴다 — AI docs/진행상황.md.

그래서 그런 그림만:
1. 얼굴 떼기 — 목(코) 관절에서 몸통 쪽으로 가며 마스크 폭이 가장 좁은 곳을 자르고, 목이 있는 덩어리를 얼굴로 본다
   (머리카락 · 리본 포함). 몸통 · 엉덩이 관절이 그 덩어리에 들어 있으면 자르기 실패로 보고 쓰지 않는다.
2. 적용 조건 — 얼굴 넓이 / 그림 넓이 >= HEAD_MIN, 자른 곳 폭 / 얼굴 폭 <= NECK_MAX (목을 가늘게 그린 보통 그림은 제외).
3. 찾기 렌더 — 텍스처를 얼굴만 표시색, 나머지 투명으로 바꿔 한 번 그려 프레임마다 얼굴 자리(중심 · 기울기)를 잰다.
   팔이 얼굴 앞을 지나도 가려지지 않는다. 앞뒤 프레임으로 고르게 한다(동작은 되풀이되니 처음과 끝을 잇는다).
4. 그리기 렌더 — 얼굴 자리만 표시색인 텍스처로 그린 뒤, 프레임마다 표시색(휜 얼굴)을 지우고 원본 얼굴을 옮겨 붙인다.
   기울기는 ±MAX_TILT 안에서만 따른다. 얼굴은 늘 팔보다 앞에 그려진다.

시험(저장소 밖): 사용자 사진 9장 + Meta 예제 6장 중 머리가 크고 목 없는 3장에만 적용, 나머지는 그대로.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

HEAD_MIN = 0.35     # 얼굴 덩어리가 그림 넓이의 이만큼 이상
NECK_MAX = 0.5      # 얼굴과 몸이 이어지는 곳 폭이 얼굴 폭의 이만큼 이하
MAX_TILT = 15.0     # 얼굴 기울기 상한(도)
MARKER_NEAR = 120   # 표시색으로 볼 RGB 거리
FRINGE_NEAR = 210   # 표시색 둘레(7px)에서 지울 섞인 색 거리
MARKERS = np.array([[0, 255, 255], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 0]], float)


@dataclass
class FacePlan:
    texture: np.ndarray      # 원래 텍스처 (BGRA)
    head: np.ndarray         # 얼굴 자리 (bool, 텍스처 크기)
    piece: Image.Image       # 얼굴만 잘라 낸 RGBA
    rest: tuple              # piece 안 얼굴 중심 x, y · 각도 · 길쭉함 · 넓이(텍스처 px)
    marker: np.ndarray       # 표시색 RGB
    why: str


def _run_width(m: np.ndarray, p: np.ndarray, side: np.ndarray):
    h, w = m.shape
    x, y = round(p[0]), round(p[1])
    if not (0 <= x < w and 0 <= y < h and m[y, x]):
        return 0, None
    ends, total, limit = [], 1, max(m.shape)
    for sign in (1, -1):
        k = 1
        while k < limit:
            x, y = round(p[0] + side[0] * k * sign), round(p[1] + side[1] * k * sign)
            if not (0 <= x < w and 0 <= y < h and m[y, x]):
                break
            k += 1
        total += k - 1
        ends.append(p + side * (k - 1) * sign)
    return total, ends


def split_head(alpha: np.ndarray, skeleton: list[dict]) -> tuple[np.ndarray | None, str]:
    """텍스처 알파(crop 좌표)와 char_cfg 뼈대로 얼굴 덩어리. 쓰지 않을 그림이면 (None, 이유)."""

    m = alpha > 127
    at = {j["name"]: np.array(j["loc"], float) for j in skeleton}
    if not {"neck", "torso", "hip"} <= at.keys() or not m.any():
        return None, "no-joints"
    neck, torso = at["neck"], at["torso"]
    length = float(np.linalg.norm(torso - neck))
    if length < 4:
        return None, "short"
    down = (torso - neck) / length
    side = np.array([-down[1], down[0]])
    ts = np.linspace(0, 1.0, max(10, round(length / 2)))
    runs = [_run_width(m, neck + down * length * t, side) for t in ts]
    widths = np.array([r[0] for r in runs])
    face = widths[0]
    if face == 0:
        return None, "neck-off-mask"
    i = int(np.argmin(widths))
    if widths[i] > face * NECK_MAX:
        return None, f"neck-wide {widths[i] / face:.2f}"
    cut = m.astype(np.uint8)
    if runs[i][1] is not None:
        a, b = runs[i][1]
        cv2.line(cut, tuple(np.round(a + side * 3).astype(int)), tuple(np.round(b - side * 3).astype(int)), 0, 5)
    _, lab = cv2.connectedComponents(cut, connectivity=4)
    head_id = lab[round(neck[1]), round(neck[0])]
    if head_id == 0:
        return None, "neck-label0"
    head = lab == head_id
    for j in ("torso", "hip"):
        x, y = np.round(at[j]).astype(int)
        if 0 <= y < head.shape[0] and 0 <= x < head.shape[1] and head[y, x]:
            return None, f"leak-{j}"
    ratio = head.sum() / m.sum()
    if ratio < HEAD_MIN:
        return None, f"head-small {ratio:.2f}"
    return head, f"ok head {ratio:.2f} neck {widths[i] / face:.2f}"


def _moments(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    ev, evec = np.linalg.eigh(np.cov(np.stack([xs - cx, ys - cy])))
    ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
    return cx, cy, ang, ev[1] / max(ev[0], 1e-6), len(xs)


def plan(char_dir: Path) -> FacePlan | None:
    """Meta 캐릭터 폴더(texture.png · char_cfg.yaml)를 보고 얼굴을 옮겨 붙일 그림이면 그 준비를 돌려준다."""

    tex = cv2.imread(str(char_dir / "texture.png"), cv2.IMREAD_UNCHANGED)
    if tex is None or tex.ndim != 3 or tex.shape[2] != 4:
        return None
    cfg = yaml.safe_load(open(char_dir / "char_cfg.yaml"))
    head, why = split_head(tex[:, :, 3], cfg["skeleton"])
    if head is None:
        return None
    rgb = cv2.cvtColor(tex[:, :, :3], cv2.COLOR_BGR2RGB).reshape(-1, 3)[::50].astype(float)
    marker = max(MARKERS, key=lambda c: np.percentile(np.linalg.norm(rgb - c, axis=1), 1))
    rgba = cv2.cvtColor(tex, cv2.COLOR_BGRA2RGBA)
    piece = np.zeros_like(rgba)
    piece[head] = rgba[head]
    ys, xs = np.nonzero(head)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cx, cy, ang, elong, area = _moments(head)
    return FacePlan(tex, head, Image.fromarray(piece[y0:y1, x0:x1]), (cx - x0, cy - y0, ang, elong, area), marker, why)


def write_texture(char_dir: Path, p: FacePlan, mode: str) -> None:
    """mode: "locate"(얼굴만 표시색 · 나머지 투명) · "paint"(얼굴 자리만 표시색) · "original"."""

    tex = p.texture.copy()
    if mode != "original":
        if mode == "locate":
            tex[:, :, 3] = 0
        tex[p.head, :3] = p.marker[::-1]
        tex[p.head, 3] = 255
    cv2.imwrite(str(char_dir / "texture.png"), tex)


def _marker_mask(frame: np.ndarray, marker: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(frame[:, :, :3].astype(np.float32) - marker.astype(np.float32), axis=2)
    return (frame[:, :, 3] > 0) & (d < MARKER_NEAR)


def track(frames: list[np.ndarray], p: FacePlan) -> list | None:
    """찾기 렌더 프레임마다 (중심 x, y, 기울기) — 고르게 한 값. 얼굴이 한 장도 안 보이면 None."""

    raw = []
    for f in frames:
        mk = _marker_mask(f, p.marker)
        raw.append(_moments(mk) if mk.sum() > 50 else None)
    idx = [i for i, t in enumerate(raw) if t is not None]
    if not idx:
        return None
    _, _, rang, relong, _ = p.rest
    pts, tilts, areas = [], [], []
    for i in range(len(raw)):   # 안 보인 장은 가까운 보인 장 값으로
        t = raw[min(idx, key=lambda j: abs(j - i))]
        cx, cy, ang, elong, area = t
        tilt = 0.0
        if relong > 1.25 and elong > 1.25:   # 동그란 얼굴은 각도가 의미 없어 기울이지 않는다
            tilt = float(np.clip(((ang - rang + 90) % 180) - 90, -MAX_TILT, MAX_TILT))
        pts.append((cx, cy))
        tilts.append(tilt)
        areas.append(area)
    pts, tilts = np.array(pts), np.array(tilts)
    if len(pts) >= 3:
        pts = (np.roll(pts, 1, 0) + 2 * pts + np.roll(pts, -1, 0)) / 4
        tilts = (np.roll(tilts, 1) + 2 * tilts + np.roll(tilts, -1)) / 4
    scale = float(np.sqrt(np.median([areas[i] for i in idx]) / p.rest[4]))
    return [(pts[i, 0], pts[i, 1], tilts[i], scale) for i in range(len(raw))]


def apply(frames: list[np.ndarray], where: list, p: FacePlan) -> list[np.ndarray]:
    """그리기 렌더 프레임마다 표시색(휜 얼굴)을 지우고 원본 얼굴을 옮겨 붙인다."""

    rcx, rcy = p.rest[0], p.rest[1]
    scale = where[0][3]
    small = p.piece.resize((max(1, round(p.piece.width * scale)), max(1, round(p.piece.height * scale))), Image.LANCZOS)
    out = []
    for f, (cx, cy, tilt, _) in zip(frames, where):
        base = f.copy()
        mk = _marker_mask(base, p.marker)
        fringe = cv2.dilate(mk.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        d = np.linalg.norm(base[:, :, :3].astype(np.float32) - p.marker.astype(np.float32), axis=2)
        base[mk | (fringe & (d < FRINGE_NEAR)), 3] = 0
        im = Image.fromarray(base, "RGBA")
        hd = small.rotate(-tilt, resample=Image.BICUBIC, expand=True)
        r = np.radians(tilt)
        vx, vy = rcx * scale - small.width / 2, rcy * scale - small.height / 2
        px = hd.width / 2 + vx * np.cos(r) - vy * np.sin(r)
        py = hd.height / 2 + vx * np.sin(r) + vy * np.cos(r)
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        layer.paste(hd, (round(cx - px), round(cy - py)), hd)
        im.alpha_composite(layer)
        out.append(np.array(im))
    return out
