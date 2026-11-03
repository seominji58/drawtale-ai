from PIL import Image, ImageDraw
from pathlib import Path
import json

from ai_server.analyzer import build_skeleton_edges

BASE = Path(
    r"D:\capston\A1-AI-Runtime\configs\skeletons\quadruped"
)

IMAGE_PATH = BASE / "dog.png"
JOINTS_PATH = BASE / "quadruped_manual_joints.json"
OUTPUT_PATH = BASE / "dog_quadruped_skeleton.png"

with open(
    JOINTS_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

joints = data["joints"]
skeleton_type = data["skeleton_type"]

edges = build_skeleton_edges(
    joints,
    skeleton_type
)

img = Image.open(
    IMAGE_PATH
).convert("RGB")

draw = ImageDraw.Draw(img)

# Skeleton 연결선
for edge in edges:
    draw.line(
        [
            (
                edge["parent_x"],
                edge["parent_y"]
            ),
            (
                edge["child_x"],
                edge["child_y"]
            )
        ],
        fill="blue",
        width=5
    )

# Joint 점
for joint in joints:
    x = joint["x"]
    y = joint["y"]

    r = 7

    draw.ellipse(
        (
            x - r,
            y - r,
            x + r,
            y + r
        ),
        fill="red",
        outline="white",
        width=2
    )

    draw.text(
        (
            x + 8,
            y - 8
        ),
        joint["name"],
        fill="black"
    )

img.save(
    OUTPUT_PATH
)

print(
    f"[OK] 생성됨: {OUTPUT_PATH}"
)