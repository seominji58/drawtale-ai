"""만든 설정이 맞는지 눈으로 본다.

    python check.py <이름>

마스크는 빨강, 관절은 노란 점, 뼈대는 파란 선으로 그려 out/check_<이름>.png 에 저장한다.
**마스크가 그림을 제대로 덮었는지 먼저 보라.** 여기가 틀리면 결과가 전부 깨진다.
"""

import io
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

name = sys.argv[1] if len(sys.argv) > 1 else "dog"

tex = Image.open(f"{name}/texture.png").convert("RGB")
msk = Image.open(f"{name}/mask.png").convert("L")
cfg = yaml.safe_load(io.open(f"{name}/char_cfg.yaml", encoding="utf8"))

over = Image.new("RGB", tex.size, (255, 0, 0))
vis = Image.composite(Image.blend(tex, over, 0.35), tex, msk)
d = ImageDraw.Draw(vis)

loc = {j["name"]: j["loc"] for j in cfg["skeleton"]}
for j in cfg["skeleton"]:
    if j["parent"]:
        d.line([*loc[j["parent"]], *j["loc"]], fill=(0, 0, 255), width=4)
for j in cfg["skeleton"]:
    x, y = j["loc"]
    d.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 255, 0), outline=(0, 0, 0), width=2)

Path("out").mkdir(exist_ok=True)
vis.save(f"out/check_{name}.png")
print(f"out/check_{name}.png 를 열어 보세요 ({cfg['width']}x{cfg['height']})")
