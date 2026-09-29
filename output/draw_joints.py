"""그림 위에 관절과 뼈대를 그려 눈으로 확인한다."""
import json
import sys

from PIL import Image, ImageDraw

LINKS = [
    ('neck', 'torso'), ('torso', 'hip'),
    ('neck', 'right_shoulder'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_hand'),
    ('neck', 'left_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_hand'),
    ('hip', 'right_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_foot'),
    ('hip', 'left_hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_foot'),
]


def draw(image_path: str, joints_path: str, out_path: str) -> None:
    im = Image.open(image_path).convert('RGB')
    d = ImageDraw.Draw(im)
    by = {j['name']: (j['x'], j['y']) for j in json.load(open(joints_path))}

    for a, b in LINKS:
        if a in by and b in by:
            d.line([by[a], by[b]], fill=(0, 160, 255), width=5)
    for name, (x, y) in by.items():
        color = (255, 0, 0) if 'hand' in name or 'foot' in name else (0, 90, 220)
        d.ellipse([x - 9, y - 9, x + 9, y + 9], fill=color)

    im.save(out_path)
    print('저장:', out_path)


if __name__ == '__main__':
    draw(sys.argv[1], sys.argv[2], sys.argv[3])
