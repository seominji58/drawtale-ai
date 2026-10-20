# -*- coding: utf-8 -*-
"""AI 가 찍은 관절 15개를 그림 위에 그린다 (사람 그림용).

어디를 짚었는지 눈으로 보려는 것이다. 네 발은 사람이 직접 찍으므로 check.py 를 쓴다.

    python overlay.py <그림.png> <분석결과.json> <결과.png>
"""
import io, json, sys
from PIL import Image, ImageDraw

img_path, js_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(io.open(js_path, encoding="utf8"))
loc = {j["name"]: (j["x"], j["y"]) for j in d["joints"]}
# 서비스 코드와 같은 계층 (root 는 hip 자리라 생략)
BONES = [("hip","torso"),("torso","neck"),
         ("torso","right_shoulder"),("right_shoulder","right_elbow"),("right_elbow","right_hand"),
         ("torso","left_shoulder"),("left_shoulder","left_elbow"),("left_elbow","left_hand"),
         ("hip","right_hip"),("right_hip","right_knee"),("right_knee","right_foot"),
         ("hip","left_hip"),("left_hip","left_knee"),("left_knee","left_foot")]
im = Image.open(img_path).convert("RGB")
dr = ImageDraw.Draw(im)
b = d["bbox"]
dr.rectangle([b["left"], b["top"], b["right"], b["bottom"]], outline=(255,0,0), width=3)
for a, c in BONES:
    dr.line([*loc[a], *loc[c]], fill=(0,0,255), width=5)
for n, (x, y) in loc.items():
    dr.ellipse([x-9, y-9, x+9, y+9], fill=(255,255,0), outline=(0,0,0), width=3)
im.save(out_path)
print("저장:", out_path)
