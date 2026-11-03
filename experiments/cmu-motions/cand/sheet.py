"""렌더한 GIF 를 동작마다 한 줄(6장면)로 모아 비교표를 만든다 (시험용).

    python sheet.py <변형:soft|live> [한 장에 몇 줄=8]  →  out/sheet_{변형}_{n}.png
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
variant = sys.argv[1]
per = int(sys.argv[2]) if len(sys.argv) > 2 else 8
FRAMES, CELL, LABEL = 6, 160, 130

gifs = sorted((HERE / "out").glob(f"char1_*_{variant}.gif"))
for n in range(0, len(gifs), per):
    rows = gifs[n:n + per]
    sheet = Image.new("RGB", (LABEL + CELL * FRAMES, CELL * len(rows)), "white")
    d = ImageDraw.Draw(sheet)
    for r, g in enumerate(rows):
        im = Image.open(g)
        d.text((6, r * CELL + CELL // 2), g.stem.split("_", 1)[1].rsplit("_", 1)[0], fill="black")
        for k in range(FRAMES):
            im.seek(k * (im.n_frames - 1) // (FRAMES - 1))
            f = im.convert("RGBA").resize((CELL, CELL))
            bg = Image.new("RGBA", f.size, "white")
            bg.alpha_composite(f)
            sheet.paste(bg.convert("RGB"), (LABEL + k * CELL, r * CELL))
        d.line([(0, r * CELL), (sheet.width, r * CELL)], fill="#ddd")
    sheet.save(HERE / "out" / f"sheet_{variant}_{n // per + 1}.png")
    print(f"sheet_{variant}_{n // per + 1}.png  {len(rows)}줄")
