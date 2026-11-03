from PIL import Image, ImageDraw
from pathlib import Path

BASE = Path(r"D:\capston\nonhuman_test")

RESULTS = {
    "human": {
        "bbox": (139, 94, 613, 646),
        "joints": {
            "hip": (382,459), "torso": (385,339), "neck": (379,225),
            "right_shoulder": (354,336), "right_elbow": (274,342), "right_hand": (194,342),
            "left_shoulder": (416,342), "left_elbow": (496,348), "left_hand": (583,355),
            "right_hip": (361,459), "right_knee": (336,540), "right_foot": (317,626),
            "left_hip": (404,459), "left_knee": (410,540), "left_foot": (422,620)
        }
    },

    "snake": {
        "bbox": (6, 266, 754, 525),
        "joints": {
            "hip": (575,430), "torso": (132,327), "neck": (141,322),
            "right_shoulder": (102,322), "right_elbow": (73,332), "right_hand": (34,352),
            "left_shoulder": (161,332), "left_elbow": (570,400), "left_hand": (570,410),
            "right_hip": (570,439), "right_knee": (628,469), "right_foot": (706,478),
            "left_hip": (580,420), "left_knee": (619,430), "left_foot": (706,459)
        }
    },

    "starfish": {
        "bbox": (48, 64, 732, 685),
        "joints": {
            "hip": (399,499), "torso": (403,374), "neck": (403,370),
            "right_shoulder": (296,361), "right_elbow": (216,334), "right_hand": (118,299),
            "left_shoulder": (510,388), "left_elbow": (590,352), "left_hand": (688,326),
            "right_hip": (332,495), "right_knee": (279,557), "right_foot": (234,628),
            "left_hip": (466,504), "left_knee": (519,566), "left_foot": (581,655)
        }
    },

    "snail": {
        "bbox": (99, 227, 681, 624),
        "joints": {
            "hip": (610,361), "torso": (576,350), "neck": (295,407),
            "right_shoulder": (568,354), "right_elbow": (553,316), "right_hand": (538,255),
            "left_shoulder": (583,346), "left_elbow": (553,308), "left_hand": (538,255),
            "right_hip": (606,361), "right_knee": (629,323), "right_foot": (651,278),
            "left_hip": (614,361), "left_knee": (629,316), "left_foot": (651,285)
        }
    }
}

EDGES = [
    ("hip", "torso"),
    ("torso", "neck"),

    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_hand"),

    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_hand"),

    ("hip", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_foot"),

    ("hip", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_foot"),
]

for name, result in RESULTS.items():

    image_path = BASE / f"{name}.png"

    if not image_path.exists():
        print(f"[ERROR] 파일 없음: {image_path}")
        continue

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    bbox = result["bbox"]
    joints = result["joints"]

    # Bounding Box
    draw.rectangle(
        bbox,
        outline="red",
        width=4
    )

    # Skeleton lines
    for start, end in EDGES:
        draw.line(
            [joints[start], joints[end]],
            fill="blue",
            width=5
        )

    # Joint points
    for joint_name, (x, y) in joints.items():

        radius = 7

        draw.ellipse(
            [
                x - radius,
                y - radius,
                x + radius,
                y + radius
            ],
            fill="red",
            outline="white",
            width=2
        )

        draw.text(
            (x + 8, y - 8),
            joint_name,
            fill="black"
        )

    output_path = BASE / f"{name}_pose_result.png"

    img.save(output_path)

    print(f"[OK] 생성됨: {output_path}")

print("모든 시각화 완료")