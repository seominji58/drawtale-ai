"""네 발 캐릭터 설정을 손으로 만든다 (시험용).

AI 가 동물 관절을 찍어 주지 않으므로 좌표를 사람이 넣는다.
이번 시험의 목적은 자동화가 아니라 **움직임 품질**이라 이렇게 한다.

  python make_char.py <그림.png> <이름> "x1,y1 x2,y2 ... (15개)"

좌표는 pick_joints.html 로 찍으면 그대로 나옵니다.

만드는 것: <이름>/texture.png, <이름>/mask.png, <이름>/char_cfg.yaml
좌표는 원본 그림 기준으로 넣고, 잘라낸 만큼은 스크립트가 빼 준다.
"""

import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage

# Meta 네 발 골격 (examples/characters/char6). b = 뒤, f = 앞
#
# root 는 받지 않는다. **엉덩이(hip)와 같은 자리**에 두는 것이 규칙이라 여기서 자동으로 넣는다.
# 서비스 코드(runtime/ai_server/renderer.py)도 사람 관절 15개에 같은 방식으로 root 를 붙인다.
# 떼어 놓으면 없던 뼈가 하나 생겨 걸을 때 몸통이 통째로 끌려다닌다 (2026-10-20 시험).
JOINTS = [
    ("root", None),
    ("hip", "root"),
    ("torso", "hip"),
    ("neck", "torso"),
    ("right_hip_b", "root"),
    ("right_knee_b", "right_hip_b"),
    ("right_foot_b", "right_knee_b"),
    ("left_hip_b", "root"),
    ("left_knee_b", "left_hip_b"),
    ("left_foot_b", "left_knee_b"),
    ("right_hip_f", "root"),
    ("right_knee_f", "right_hip_f"),
    ("right_foot_f", "right_knee_f"),
    ("left_hip_f", "root"),
    ("left_knee_f", "left_hip_f"),
    ("left_foot_f", "left_knee_f"),
]


def build_mask(img: Image.Image) -> np.ndarray:
    """흰 배경에서 그림만 남긴다. 아이 그림은 흰 종이에 그린다고 보고 밝기로 가른다."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.int16)
    # 흰색에서 얼마나 떨어져 있나
    far = (255 - rgb).max(axis=2) > 40
    far = ndimage.binary_closing(far, np.ones((5, 5)))
    # 가장 큰 덩어리만 — 떨어진 점은 버린다 (AI 서버 마스크와 같은 규칙)
    lab, n = ndimage.label(far)
    if n == 0:
        raise SystemExit("그림을 찾지 못했습니다. 배경이 흰색인지 보세요.")
    big = (lab == (np.bincount(lab.ravel())[1:].argmax() + 1))
    return ndimage.binary_fill_holes(big)


def main() -> None:
    src, name, coords = sys.argv[1], sys.argv[2], sys.argv[3]
    pts = [tuple(int(v) for v in p.split(",")) for p in coords.split()]
    if len(pts) != 15:
        raise SystemExit(f"관절은 15개여야 합니다 (받은 개수: {len(pts)}). root 는 넣지 않습니다")
    pts = [pts[0]] + pts  # root = hip (같은 자리)

    img = Image.open(src)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGBA")

    mask = build_mask(img)
    ys, xs = np.where(mask)
    pad = 12
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, img.width)
    y1 = min(int(ys.max()) + pad + 1, img.height)

    out = Path(name)
    out.mkdir(exist_ok=True)
    img.crop((x0, y0, x1, y1)).save(out / "texture.png")
    Image.fromarray((mask[y0:y1, x0:x1] * 255).astype(np.uint8), mode="L").save(out / "mask.png")

    w, h = x1 - x0, y1 - y0
    skeleton = [
        {"name": n, "parent": p, "loc": [int(x) - x0, int(y) - y0]}
        for (n, p), (x, y) in zip(JOINTS, pts)
    ]
    with open(out / "char_cfg.yaml", "w") as f:
        yaml.dump({"width": w, "height": h, "skeleton": skeleton}, f, sort_keys=False)

    outside = [j["name"] for j in skeleton
               if not (0 <= j["loc"][0] < w and 0 <= j["loc"][1] < h)]
    print(f"{name}: 잘라낸 크기 {w}x{h} (원본 {img.width}x{img.height})")
    if outside:
        print("  그림 밖으로 나간 관절:", ", ".join(outside))
    else:
        print("  관절 16개 모두 그림 안에 있습니다")


if __name__ == "__main__":
    main()
