"""Meta AnimatedDrawings 렌더링 래퍼.

Backend가 보낸 관절 좌표(사용자 보정 반영)로 Meta 캐릭터 설정을 만들고
애니메이션 MP4를 생성한다.

렌더링은 화면 없는 환경에서 동작해야 하므로 OSMesa(USE_MESA)를 사용한다.
OpenCV에 H.264 인코더가 없어 MP4를 직접 만들 수 없으므로,
GIF로 렌더링한 뒤 ffmpeg으로 MP4로 변환한다.
"""

import hashlib
import logging
from collections import OrderedDict
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from ai_server import face_lock, forms

ANIMATED_DRAWINGS_DIR = Path(
    os.getenv("ANIMATED_DRAWINGS_DIR", "/opt/AnimatedDrawings")
)

# Meta skeleton 계층. 프로젝트 표준 15개 관절에 root(=hip 위치)를 더해 16개가 된다.
JOINT_PARENTS: dict[str, str | None] = {
    "root": None,
    "hip": "root",
    "torso": "hip",
    "neck": "torso",
    "right_shoulder": "torso",
    "right_elbow": "right_shoulder",
    "right_hand": "right_elbow",
    "left_shoulder": "torso",
    "left_elbow": "left_shoulder",
    "left_hand": "left_elbow",
    "right_hip": "root",
    "right_knee": "right_hip",
    "right_foot": "right_knee",
    "left_hip": "root",
    "left_knee": "left_hip",
    "left_foot": "left_knee",
}

_META_MOTIONS = ANIMATED_DRAWINGS_DIR / "examples/config/motion"

# 팔을 몸에 붙여 그린 그림은 팔이 몸 쪽으로 내려오거나 팔꿈치가 접힐 때 뭉개진다.
# *_gentle 은 팔 회전을 줄이고 팔꿈치를 거의 편 채로 둔 판이다.
# runtime/tools/make_gentle_motions.py 가 Meta BVH 에서 만든다 (third_party 는 고치지 않는다).
_GENTLE_MOTIONS = Path(__file__).parent / "motions"

_META_RETARGET = ANIMATED_DRAWINGS_DIR / "examples/config/retarget"

# 사용 가능한 동작. Backend가 이 이름으로 요청한다. 이름 -> (motion 설정 파일, retarget 설정 파일)
# walk · run 은 CMU Mocap 이라 뼈 이름이 달라 우리 retarget(cmu1_human)을 쓴다 (출처 motions/CMU-NOTICE.md)
MOTIONS: dict[str, tuple[Path, Path]] = {
    "wave_hello": (_META_MOTIONS / "wave_hello.yaml", _META_RETARGET / "fair1_ppf.yaml"),
    "jumping": (_META_MOTIONS / "jumping.yaml", _META_RETARGET / "fair1_ppf.yaml"),
    "jumping_jacks": (_META_MOTIONS / "jumping_jacks.yaml", _META_RETARGET / "cmu1_pfp.yaml"),
    "dab": (_META_MOTIONS / "dab.yaml", _META_RETARGET / "fair1_ppf.yaml"),
    "zombie": (_META_MOTIONS / "zombie.yaml", _META_RETARGET / "fair1_spf.yaml"),
    "wave_hello_gentle": (_GENTLE_MOTIONS / "wave_hello_gentle.yaml", _META_RETARGET / "fair1_ppf.yaml"),
    # 같은 이름 그대로 고친 판을 가리킨다 (runtime/tools/make_soft_motions.py, 2026-11-03) —
    # 점프는 웅크릴 때 몸 · 머리가 83% 로 눌려서, 걷기 · 뛰기는 다리 PCA 투영이 프레임마다 뒤집혀 다리가 꼬여서.
    # 고치기 전 판은 *_orig 로 남겨 둔다 (비교용, 백엔드는 쓰지 않음)
    "jumping_gentle": (_GENTLE_MOTIONS / "jumping_soft.yaml", _META_RETARGET / "fair1_ppf.yaml"),
    # 걷기 · 뛰기는 정면 제자리 걸음 (11/9 사용자 결정 — 정면 그림을 옆으로 걷게 하면 다리가 엇갈려 꼬이고 휘었다.
    # 옆면 걸음 cmu1_side 는 실험으로 남김). 장면이 그림을 옆으로 옮기고 가는 쪽으로 기울인다
    # 사람형 주인공 걷기 · 달리기는 장면 등장인물과 같은 걸음 (2026-11-10 사용자 「등장인물처럼 무릎 굽히며 걷는 걸음을 주인공에도」).
    # 전에는 walk_steps(무릎을 앞으로 드는 CMU 걸음)라 앞에서 보면 다리가 옆으로 꺾였다. 달리기는 같은 걸음을 화면이 빠르게 돌린다
    "walk": (_GENTLE_MOTIONS / "scene_walk.yaml", _GENTLE_MOTIONS / "scene_front.yaml"),
    "run": (_GENTLE_MOTIONS / "scene_walk.yaml", _GENTLE_MOTIONS / "scene_front.yaml"),
    "walk_side": (_GENTLE_MOTIONS / "walk.yaml", _GENTLE_MOTIONS / "cmu1_side.yaml"),
    "walk_front": (_GENTLE_MOTIONS / "walk_steps.yaml", _GENTLE_MOTIONS / "cmu1_front.yaml"),
    "jumping_gentle_orig": (_GENTLE_MOTIONS / "jumping_gentle.yaml", _META_RETARGET / "fair1_ppf.yaml"),
    "walk_orig": (_GENTLE_MOTIONS / "walk.yaml", _GENTLE_MOTIONS / "cmu1_human.yaml"),
    "run_orig": (_GENTLE_MOTIONS / "run.yaml", _GENTLE_MOTIONS / "cmu1_human.yaml"),
    "dab_gentle": (_GENTLE_MOTIONS / "dab_gentle.yaml", _META_RETARGET / "fair1_ppf.yaml"),
    # 장면 등장인물(친구 · 엄마 · 직업인 …, 팔을 몸에 붙여 그린 일러스트)이 걸어 들어올 때 — 걷기와 같은 동작에 팔만 그린
    # 자세 그대로 (forms/biped.json waves.walk_still_arms). 팔까지 움직이면 몸에 붙은 팔이 날개처럼 찢어졌다 (11/9)
    "walk_still_arms": (_GENTLE_MOTIONS / "walk_steps.yaml", _GENTLE_MOTIONS / "cmu1_front.yaml"),
    # 장면 등장인물 그림띠용 (프론트 tools/people-moves, 11/9): 한 팔만 들어 흔드는 인사(흔들기 구간을 프론트가 되풀이) · 일자걸음.
    # 관절 각도로 만든다 (runtime/tools/make_scene_motions.py). 첫 장이 팔을 조금 벌리고 선 그린 자세
    "scene_wave": (_GENTLE_MOTIONS / "scene_wave.yaml", _GENTLE_MOTIONS / "scene_front.yaml"),
    "scene_walk": (_GENTLE_MOTIONS / "scene_walk.yaml", _GENTLE_MOTIONS / "scene_front.yaml"),
    # 이야기 쪽 동작 9개 (2026-11-04, pt_docs 17 — 쪽 45%가 가만히였다). CMU 후보 32개 중 평면 그림에서 몸이
    # 뒤집히거나 팔이 뭉개지지 않고 서로 달라 보이는 것만 골랐고, 만세 · 떨기는 관절 각도로 만들었다.
    # 이름은 백엔드 쪽 동작 이름과 같다. 만드는 법은 runtime/tools/install_more_motions.py
    **{
        name: (_GENTLE_MOTIONS / f"{name}.yaml", _GENTLE_MOTIONS / "cmu1_front.yaml")
        for name in (
            "breathe", "shiver", "hooray", "point", "fly", "swim", "march", "kick", "search",
        )
    },
}
DEFAULT_MOTION = "wave_hello"

# 렌더링 시 캐릭터 영역 둘레에 두는 여유 (bbox 크기 대비 비율).
# 캐릭터가 잘리지 않을 때까지 앞에서부터 차례로 시도한다.
CROP_MARGINS = (0.12, 0.25, 0.40, 0.60)

# 텍스처 마스크를 3x3 로 몇 번 깎나. 분석이 종이색을 이미 깎아(analyzer.refine_mask) 1px 만 —
# 2px 면 가는 팔다리 선이 사라진다
MASK_ERODE_ITERATIONS = 1

# 렌더에 넘기는 그림의 긴 변 상한 (px). 영상이 500px 라 이보다 크면 시간 · 메모리만 든다
RENDER_MAX_SIDE = 800
# 움직이는 틀 윤곽을 매끈하게 넘기는 흐림 정도 (px)
MESH_SMOOTH = 2.0
# 가장자리에 닿은 그림에 덧대는 투명 여백 (긴 변 대비)
EDGE_PAD = 0.04

# 움직이는 틀(메시)을 이을 뼈대 띠 굵기 (자른 그림 대각선 대비)
BONE_BAND = 0.012
BONES = (
    ("hip", "torso"), ("torso", "neck"),
    ("neck", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
    ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
    ("hip", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_foot"),
    ("hip", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_foot"),
)


def mesh_mask(mask: np.ndarray, joints: list[dict], tight: bool = False) -> np.ndarray:
    """Meta 는 마스크에서 가장 큰 덩어리 하나만 움직인다 (나머지는 버림).
    종이색을 깎으면 가는 팔다리 선이 몸통과 떨어진 조각이 되므로, 관절 사이 뼈대 띠로 이어 붙인다.
    보이는 부분은 텍스처 투명도(깎은 마스크)가 정하니 띠 자리의 종이는 보이지 않는다 (2026-11-03).

    tight: 오린 그림(투명도 마스크, alpha_mask)이면 부풀려 매끈하게 한 틀을 원래 모양 + 뼈대 띠로 다시 깎는다 (2026-11-09).
    부풀리기가 좁은 다리 사이 틈을 메워 두 다리가 한 덩어리 틀이 되면, 렌더에서 한쪽 다리가 무릎에서 꺾여 파였다(장면 할머니).
    오린 그림은 윤곽이 원래 매끈해 점 수가 늘지 않는다. 아이 그림은 그대로"""

    out = mask.copy()
    bands = np.zeros_like(mask)
    at = {j["name"]: (int(j["x"]), int(j["y"])) for j in joints}
    form = forms.form_or_default(set(at))
    bones = BONES if form == forms.DEFAULT else forms.bones(form)   # 사람은 목 → 어깨 (부모 관계와 다르다)
    width = max(3, round(BONE_BAND * float(np.hypot(*mask.shape[:2]))))
    for a, b in bones:
        if a in at and b in at:
            cv2.line(out, at[a], at[b], 255, width)
            cv2.line(bands, at[a], at[b], 255, width)
    # 윤곽을 매끈하게 — Meta 는 윤곽 점마다 계산이 커져, 깎은 마스크의 들쭉날쭉한 윤곽 그대로면
    # 렌더 한 번에 메모리가 1.6GB 를 넘어 컨테이너(2GB)가 죽었다. 틀이 조금 커져도 텍스처가 투명이라 안 보인다
    k = max(3, round(BONE_BAND * float(np.hypot(*mask.shape[:2]))))
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    out = np.where(cv2.GaussianBlur(out, (0, 0), k) > 127, 255, 0).astype(np.uint8)
    if tight:
        out = np.where((out > 127) & (np.maximum(mask, bands) > 127), 255, 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((out > 127).astype(np.uint8), connectivity=8)
    if count > 2:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.where(labels == biggest, 255, 0).astype(np.uint8)
    out[[0, -1], :] = 0
    out[:, [0, -1]] = 0
    return out


class RenderError(RuntimeError):
    """message는 "CODE: 설명" 형식으로 둬 Backend가 코드를 읽을 수 있게 한다."""


def build_char_cfg(joints: list[dict], width: int, height: int) -> dict:
    """관절로 Meta char_cfg.yaml 내용을 만든다. 좌표는 crop 이미지 기준.
    뼈대(관절 이름 · 부모)는 관절 이름으로 알아본 형태의 정의에서 (forms/*.json, pt_docs 19)."""

    by_name = {j["name"]: j for j in joints}
    try:
        form = forms.form_of(set(by_name))
        parents, root_at = forms.parents(form), forms.get(form).get("root_at", "hip")
    except KeyError:
        parents, root_at = JOINT_PARENTS, "hip"   # 이름이 모자라면 아래에서 사람 관절 기준으로 무엇이 빠졌는지 알린다

    missing = [n for n in parents if n != "root" and n not in by_name]
    if missing:
        raise RenderError(f"INVALID_JOINTS: missing joints {missing}")

    # root는 hip과 같은 위치에 둔다 (Meta 예시와 동일). 엉덩이가 없는 형태(사슬형 …)는 형태 정의의 root_at
    at = by_name[root_at]
    by_name["root"] = {"name": "root", "x": at["x"], "y": at["y"]}

    skeleton = []
    for name, parent in parents.items():
        joint = by_name[name]
        skeleton.append(
            {
                "loc": [round(joint["x"]), round(joint["y"])],
                "name": name,
                "parent": parent,
            }
        )

    return {"height": height, "width": width, "skeleton": skeleton}


def expand_crop(
    mask: np.ndarray, bbox: dict, image_width: int, image_height: int
) -> tuple[int, int, int, int]:
    """캐릭터가 잘리지 않는 crop 범위를 찾는다.

    mask는 원본 이미지 전체 크기다. crop 경계에 캐릭터가 닿아 있으면
    그만큼 잘린 것이므로 여유를 넓힌다. 이미지 범위를 넘을 수는 없다.
    """

    bbox_left, bbox_top = round(bbox["left"]), round(bbox["top"])
    bbox_w = round(bbox["right"]) - bbox_left
    bbox_h = round(bbox["bottom"]) - bbox_top

    result = (bbox_left, bbox_top, bbox_left + bbox_w, bbox_top + bbox_h)

    for ratio in CROP_MARGINS:
        margin_x = round(ratio * bbox_w)
        margin_y = round(ratio * bbox_h)

        left = max(0, bbox_left - margin_x)
        top = max(0, bbox_top - margin_y)
        right = min(image_width, bbox_left + bbox_w + margin_x)
        bottom = min(image_height, bbox_top + bbox_h + margin_y)
        result = (left, top, right, bottom)

        # 이 범위로 잘랐을 때 캐릭터가 경계에 닿는지 본다
        part = mask[top:bottom, left:right] > 127
        touching = (
            part[0].any() or part[-1].any()
            or part[:, 0].any() or part[:, -1].any()
        )

        if not touching:
            break

    return result


def prepare_character(
    image_path: str,
    mask_path: str,
    joints: list[dict],
    bbox: dict,
    work_dir: Path,
) -> Path:
    """Meta가 요구하는 캐릭터 폴더(char_cfg.yaml, texture.png, mask.png)를 만든다.

    Meta는 캐릭터 영역만 잘라낸 이미지를 기준으로 하므로, 원본 좌표를 crop 기준으로 옮긴다.
    """

    char_dir = work_dir / "character"
    char_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise RenderError("INVALID_IMAGE: could not read source image")

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RenderError("MASK_NOT_FOUND: could not read mask image")

    if mask.shape[:2] != image.shape[:2]:
        raise RenderError("MASK_SIZE_MISMATCH: mask must cover the whole image")

    # 큰 사진은 줄여서 렌더한다. 영상은 어차피 500px 인데, 1600px 사진은 렌더 한 번에 30초가 걸려
    # (한 번에 하나씩 도니) 셋째 사람이 100초를 기다렸다 (2026-11-03 VM)
    # 가장자리 깎기(아래 설명)는 줄이기 전 원본 크기에서 — 줄인 뒤 깎으면 가는 다리 선이 점으로 끊겼다
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=MASK_ERODE_ITERATIONS)
    scale = RENDER_MAX_SIDE / max(image.shape[:2])
    if scale < 1:
        size = (round(image.shape[1] * scale), round(image.shape[0] * scale))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        # 가는 팔다리 선이 끊기지 않게 평균으로 줄인 뒤 조금이라도 걸친 칸은 남긴다
        mask = np.where(cv2.resize(mask, size, interpolation=cv2.INTER_AREA) >= 100, 255, 0).astype(np.uint8)
        joints = [{**j, "x": j["x"] * scale, "y": j["y"] * scale} for j in joints]
        bbox = {k: (v * scale if isinstance(v, (int, float)) else v) for k, v in bbox.items()}

    # bbox는 캐릭터에 딱 맞게(때로는 팔다리 끝을 빼고) 잡히므로, 그대로 자르면 끝이 잘린다.
    # 마스크가 가장자리에 닿지 않을 때까지 여유를 넓혀 가며 자른다.
    left, top, right, bottom = expand_crop(mask, bbox, image.shape[1], image.shape[0])

    if right <= left or bottom <= top:
        raise RenderError("INVALID_BBOX: character bounding box is invalid")

    cropped = image[top:bottom, left:right]

    # 마스크는 원본 전체 크기이므로 같은 범위로 자른다
    mask = mask[top:bottom, left:right].copy()

    # 사진 가장자리에 닿은 그림(잘라 낼 여유가 없어 그림 칸 = 사진 전체)은 둘레에 투명 여백을 덧댄다.
    # 아니면 아래 윤곽 흐리기(MESH_SMOOTH)가 가장자리 쪽 윤곽을 끊어 Meta 가 틀을 절반만 만들었다 —
    # S-05 미리보기에서 캐릭터 왼쪽 절반이 잘림 (2026-11-04 사용자, Meta 예제 char1: 손끝이 사진 왼쪽 끝에 닿음).
    # 닿지 않는 그림은 그대로 둔다 (Meta 는 그림 칸 크기로 캐릭터 크기를 잡아, 늘 덧대면 다른 그림 크기도 바뀐다)
    # 정확히 닿은 것만 보면, 가장자리에서 몇 픽셀 떨어진 그림(그림판 끝까지 길게 그린 뱀)은 덧대지 않아 같은 이유로
    # 몸 대부분이 빠지고 아래쪽 두 조각만 움직였다 (11/9). 덧댈 폭 안에 들어오면 덧댄다
    pad = max(8, round(EDGE_PAD * max(mask.shape[:2])))
    near = max(2, pad // 2)
    if mask[:near].any() or mask[-near:].any() or mask[:, :near].any() or mask[:, -near:].any():
        cropped = cv2.copyMakeBorder(cropped, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        mask = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        left, top = left - pad, top - pad   # 관절을 옮길 때 같은 만큼 더해지게
    height, width = cropped.shape[:2]

    # 마스크가 가장자리까지 차면 Meta 메시가 윤곽선을 찾지 못해 렌더가 실패한다
    mask[[0, -1], :] = 0
    mask[:, [0, -1]] = 0

    # 마스크 경계가 바깥으로 1~2px 넓어 종이 · 배경색이 캐릭터 둘레에 얇게 남는다 (2026-11-03 사용자 확인,
    # 빨간 · 노란 바탕). 굵은 테두리 선 안쪽으로 조금 깎아 낸다
    # (깎기는 위에서 줄이기 전에 했다)

    # texture는 RGBA여야 한다. 마스크 밖은 투명 처리.
    texture = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    texture[:, :, 3] = mask

    cv2.imwrite(str(char_dir / "texture.png"), texture)

    # 관절 좌표를 원본 기준 -> crop 기준으로 옮긴다.
    # 사용자가 관절을 캐릭터 영역 밖으로 끌 수 있는데 Meta는 음수 좌표를 거부하므로,
    # crop 영역 안으로 제한한다.
    crop_joints = [
        {
            "name": j["name"],
            "x": min(max(round(j["x"]) - left, 0), width - 1),
            "y": min(max(round(j["y"]) - top, 0), height - 1),
        }
        for j in joints
    ]

    # 움직이는 틀은 팔다리 조각을 뼈대로 이어 붙인 마스크 (보이는 부분은 위 텍스처).
    # 흑백 두 값 그대로면 Meta 가 찾는 윤곽(128 등고선)이 픽셀 계단을 따라가 점이 수천 개가 되고,
    # 렌더 시간 · 메모리가 점 수에 비례해 는다. 살짝 흐려 등고선을 매끈한 선으로 넘긴다
    tight = (Path(mask_path).parent / "mask_alpha").exists()   # 오린 그림 (main.analyze mask_from=alpha)
    mesh = mesh_mask(np.where(mask > 127, 255, 0).astype(np.uint8), crop_joints, tight)
    cv2.imwrite(str(char_dir / "mask.png"), cv2.GaussianBlur(mesh, (0, 0), MESH_SMOOTH))

    char_cfg = build_char_cfg(crop_joints, width=width, height=height)
    with open(char_dir / "char_cfg.yaml", "w") as f:
        yaml.dump(char_cfg, f)

    return char_dir


# 투명 프레임 묶음 (S-09 장면 층). 칸은 정사각 SPRITE_CELL, 가로는 늘 SPRITE_COLS 칸(남는 칸은 투명)으로 줄을 바꾼다.
# 렌더 화면을 자르지 않아 같은 캐릭터면 동작이 달라도 크기 · 발끝 자리가 같다.
# 칸 크기 그대로 그린다(11/3 저녁 — 전에는 640 으로 그려 512 로 줄였다. 그리는 픽셀이 36% 적고 줄이는 단계가 없다)
# 큰 얼굴 원본 그대로 옮겨 붙이기 (face_lock.py). 끄면 늘 지금처럼 Meta 가 얼굴까지 변형한다
FACE_LOCK = os.environ.get("DRAWTALE_FACE_LOCK", "1") != "0"
SPRITE_WINDOW = 512
SPRITE_CELL = 512
SPRITE_COLS = 8
SPRITE_MAX_FRAMES = 48
# 같은 캐릭터의 준비 계산 다시 쓰기 (_reuse_setup). 끄면 늘 새로 계산 (검사 · 문제 생길 때)
REUSE_SETUP = os.environ.get("DRAWTALE_REUSE_SETUP", "1") != "0"
# 꼬리 흔들기 각 (4족형 꼬리 첫 마디 기준, 끝 마디는 1.5배 — _tail_wag)
TAIL_WAG_DEG = 14.0
# 기본(Meta mvc_base_cfg)은 [0, 0.7, 2.0] · [0, 0.5, 2.0]. 기울기는 그대로 두고 가까이 · 조금 아래로
SPRITE_CAMERA_POS = (0.0, 0.6, 1.35)   # 11/3 실측: 캐릭터가 칸 높이의 4할(기본은 3할), 뛰기 · 점프도 안 잘림
SPRITE_CAMERA_FWD = (0.0, 0.5, 2.0)


class _capture_frames:
    """Meta 렌더가 GIF 로 넘기는 RGBA 프레임을 그대로 받아 둔다.

    GIF(256색 · 1비트 투명)로 저장된 뒤에는 색과 테두리가 뭉개지고, 초록 바탕 크로마키는 아이가 칠한
    초록까지 지운다 — 그래서 저장 전 프레임을 가로챈다."""

    def __init__(self) -> None:
        # Meta 렌더 모듈을 먼저 불러오면 OpenGL 이 화면 없는 렌더(osmesa)로 설정되기 전에 올라온다
        # ('GLXPlatform' object has no attribute 'OSMesa') — 설정을 먼저 둔다
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
        import animated_drawings.controller.video_render_controller as vrc

        self._cls = vrc.GIFWriter
        self._orig = vrc.GIFWriter.process_frame
        self._orig_cleanup = vrc.GIFWriter.cleanup
        self.frames: list[np.ndarray] = []
        self.duration_ms = 33
        frames, orig, me = self.frames, self._orig, self

        def process_frame(writer, frame):  # noqa: ANN001 — Meta 쪽 서명 그대로
            orig(writer, frame)
            frames.append(writer.frames[-1])
            me.duration_ms = writer.duration

        # GIF 는 쓰지 않는다 — 받아 둔 프레임으로 MP4 를 바로 만든다(_frames_to_mp4). 장마다 256색으로 줄여
        # 압축하고 다시 푸는 단계가 없어진다 (11/3 저녁 「렌더링 속도」)
        def cleanup(writer):  # noqa: ANN001
            return None

        self._cls.process_frame = process_frame
        self._cls.cleanup = cleanup

    def release(self) -> None:
        self._cls.process_frame = self._orig
        self._cls.cleanup = self._orig_cleanup


# 같은 캐릭터의 다른 동작은 Meta 의 준비 계산(관절 → 삼각형 · 변형 계산기)이 똑같다 — 그림 · 틀 · 처음 관절 자리로만
# 정해지고 동작과는 상관없다. 캐릭터 하나에 렌더가 5번(미리보기 + 동작 4)이라 매번 4초쯤 다시 계산했다 (11/3 저녁 실측:
# 큰 사진 손 흔들기 6초 중 관절 → 삼각형 2.5초 · 변형 계산기 1.4초, 실제 그리기 0.4초). 입력이 같을 때만(해시) 다시 쓴다
_SETUP_CACHE_SIZE = 2
_jt_cache: "OrderedDict[str, dict]" = OrderedDict()
_arap_cache: "OrderedDict[str, object]" = OrderedDict()


def _digest(*parts) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(part.tobytes() if isinstance(part, np.ndarray) else repr(part).encode())
    return h.hexdigest()


def _remember(cache: "OrderedDict", key: str, value) -> None:
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _SETUP_CACHE_SIZE:
        cache.popitem(last=False)


# 그물(mesh) 안쪽 점 간격 — 메타 기본은 그림 칸 전체를 40×40 으로만 나눠, 다리 · 팔처럼 가는 곳은 안쪽 점이 거의 없었다.
# 큰 삼각형이 통째로 꺾여 걸을 때 테두리가 울퉁불퉁했다 (11/9 사용자 「관절이랑 몸통 전체가 울퉁불퉁」). 더 촘촘하게
MESH_GRID = int(os.environ.get("MESH_GRID", "40"))   # 90 이면 렌더가 3초 → 35초 (11/9 잼) — 메타 기본 40 으로 두고 실험만


class _dense_mesh:
    """Meta AnimatedDrawing._generate_mesh 를 안쪽 점만 촘촘하게(MESH_GRID) 바꾼 판 — 나머지는 원래와 같다."""

    def __init__(self) -> None:
        import animated_drawings.model.animated_drawing as ad

        self._cls = ad.AnimatedDrawing
        self._orig = ad.AnimatedDrawing._generate_mesh

        def gen(drawing) -> None:  # noqa: ANN001
            from scipy.spatial import Delaunay
            from shapely import geometry
            from skimage import measure

            contours = measure.find_contours(drawing.mask, 128)
            contours.sort(key=len, reverse=True)
            outside = measure.approximate_polygon(contours[0], tolerance=0.25)
            outline = geometry.Polygon(contours[0])
            from shapely.prepared import prep
            inside_test = prep(outline)
            g = np.linspace(0, drawing.img_dim, MESH_GRID)
            xv, yv = np.meshgrid(g, g)
            pts = [(x, y) for x, y in zip(xv.ravel(), yv.ravel()) if inside_test.contains(geometry.Point(x, y))]
            vertices = np.concatenate([outside, np.array(pts).reshape(-1, 2)]).astype(np.float32)
            tris = []
            for t in Delaunay(vertices).simplices:
                if inside_test.contains(geometry.Point(np.mean(vertices[t], 0))):
                    tris.append(t)
            vertices /= drawing.img_dim
            drawing.mesh = {"vertices": vertices, "triangles": tris}

        ad.AnimatedDrawing._generate_mesh = gen

    def release(self) -> None:
        self._cls._generate_mesh = self._orig


class _reuse_setup:
    """Meta AnimatedDrawing 의 준비 계산 두 가지를 입력이 같으면 다시 쓴다 (결과 · 부작용 모두 원래와 같게).

    - _initialize_joint_to_triangles_dict: 결과(관절별 삼각형)를 기억하고, 원래 함수가 하는 부작용(관절 y 좌표 뒤집기)도 똑같이 한다
    - ARAP(변형 계산기): 만든 뒤 바뀌지 않는다 — solve 는 임시 칸(b1)만 덮어쓴다. 같은 객체를 다시 넘긴다"""

    def __init__(self) -> None:
        import animated_drawings.model.animated_drawing as ad

        self._mod, self._cls = ad, ad.AnimatedDrawing
        self._orig_jt = ad.AnimatedDrawing._initialize_joint_to_triangles_dict
        self._orig_arap = ad.ARAP
        orig_jt, orig_arap = self._orig_jt, self._orig_arap

        def jt(drawing):  # noqa: ANN001
            skel = [(j["name"], j["parent"], tuple(j["loc"])) for j in drawing.char_cfg.skeleton]
            tris = np.stack(drawing.mesh["triangles"]) if len(drawing.mesh["triangles"]) else np.zeros(0)
            key = _digest(drawing.mask, drawing.img_dim, skel, np.asarray(drawing.mesh["vertices"]), tris)
            hit = _jt_cache.get(key)
            if hit is None:
                orig_jt(drawing)
                _remember(_jt_cache, key, {k: v.copy() for k, v in drawing.joint_to_tri_v_idx.items()})
                return
            for joint in drawing.char_cfg.skeleton:   # 원래 함수의 부작용 그대로
                joint["loc"][1] = 1 - joint["loc"][1]
            drawing.joint_to_tri_v_idx = {k: v.copy() for k, v in hit.items()}
            _jt_cache.move_to_end(key)

        def arap(pins_xy, triangles, vertices, w: int = 1000):  # noqa: ANN001
            tris = np.stack(triangles) if len(triangles) else np.zeros(0)
            key = _digest(np.asarray(pins_xy), tris, np.asarray(vertices), w)
            hit = _arap_cache.get(key)
            if hit is None:
                hit = orig_arap(pins_xy, triangles, vertices, w)
                # 준비할 때만 쓰는 큰 행렬(A1 · A2, 큰 사진이면 30MB) — 그릴 때(solve)는 희소 행렬만 쓴다. 메모리를 아낀다
                hit.A1 = hit.A2 = None
                _remember(_arap_cache, key, hit)
            else:
                _arap_cache.move_to_end(key)
            return hit

        ad.AnimatedDrawing._initialize_joint_to_triangles_dict = jt
        ad.ARAP = arap

    def release(self) -> None:
        self._cls._initialize_joint_to_triangles_dict = self._orig_jt
        self._mod.ARAP = self._orig_arap


def _center_body(drawing) -> None:  # noqa: ANN001
    """몸 가운데가 렌더 칸 가운데에 오게 몸 위치를 옮긴다 (물결로 움직이는 형태 — 사슬 · 지느러미 · 날개 · 촉수).

    렌더 카메라는 뿌리 관절 쪽을 본다. 사슬형은 뿌리가 머리(몸 한쪽 끝)라, 그림판 끝까지 길게 그리거나 머리가 오른쪽 ·
    비스듬하게 그린 뱀은 몸이 반대쪽으로 칸 밖까지 뻗어 잘렸다 (11/9 「화면에 그리기로 그린 뱀이 잘림」).
    관절 · 움직임은 그대로 두고 모든 장의 몸 위치를 (뿌리 − 관절 가운데)만큼 옮긴다. 장면은 sprite_box 로 자리를 잡아 그대로다"""
    root, rt = drawing.rig.root_joint, drawing.retargeter
    pts = np.asarray(drawing.rig.get_joints_2D_positions(), float)
    if not len(pts):
        return
    middle = (pts.min(axis=0) + pts.max(axis=0)) / 2
    shift = np.asarray(root.get_world_position()[:2], float) - middle
    # 대응표의 시작 자리(char_starting_location, x −0.3)만큼 왼쪽에 놓여 몸 가운데가 칸의 1/3 에 왔다 — 가로는 가운데로
    shift[0] -= float(drawing.retarget_cfg.char_start_loc[0])
    rt.char_root_positions[:, :2] += shift.astype(np.float32)


def _apply_wave(drawing, wave: dict) -> None:  # noqa: ANN001
    """마디(뼈)마다 방향을 직접 넣는 움직임 — 동작 파일에 없는 움직임 (사슬형 꿈틀 · 지느러미형 헤엄 · 형태별 대표 동작, 11/5).

    꼬리 흔들기와 같은 방법이다. wave 칸 (모두 마디 목록 bones 순서, 머리 → 꼬리):
    - amp · cycles · lag: 물결 「폭(도) × sin(박자 − 마디 지연)」. 늦음이 있어 물결이 머리에서 꼬리로 흐른다.
      물결 흐르는 빠르기(물결 길이 × 박자)를 장면이 옮기는 빠르기에 맞춘다 — 물결이 6배 빠르면 제자리에서 꿈틀대며
      미끄러지는 것처럼 보였다. 장면 걷기는 1초에 몸길이 0.08 ~ 0.14, 뛰기 0.12 ~ 0.22 (프론트 SceneStage STRIDE)
    - lower: 마디를 땅 쪽으로 숙이는 각(도, 음수면 든다). 뼈가 오른쪽을 보는지 왼쪽을 보는지에 따라 회전 방향을 바꿔 —
      머리 방향과 상관없이 「숙이기」가 된다 (4족 냄새 맡기 · 뱀 고개 들기)
    - curl: 말기 — 꼬리 쪽 마디일수록 이만큼(도)씩 더 휜다. 한 동작에 한 번 말았다 편다 (뱀 몸 말기)
    - spin: 한 동작에 몇 바퀴 — 몸 가운데를 축으로 돈다 (물고기 빙글 돌기)
    - bob: 몸 전체 오르내림 (몸길이 대비, 지느러미형 헤엄). bob_by: 폭을 잴 두 관절 (없으면 bones 처음 · 끝)
    - keep_root: 참이면 몸 위치를 동작 파일 그대로(4족 — 숨쉬기 몸 들썩임). 아니면 제자리
    장 수의 정수배 주기라 반복해도 끊기지 않는다"""
    root, rt = drawing.rig.root_joint, drawing.retargeter
    n = rt.joint_positions.shape[0]
    phase = np.arange(n) / n
    t = 2 * np.pi * float(wave.get("cycles", 2)) * phase
    bones = wave["bones"]
    amps = wave.get("amp") or [0] * len(bones)
    lowers = wave.get("lower") or [0] * len(bones)
    curl = float(wave.get("curl", 0)) * (0.5 - 0.5 * np.cos(2 * np.pi * phase))   # 0 → 최대 → 0
    turn = 360.0 * float(wave.get("spin", 0)) * phase
    for i, name in enumerate(bones):
        joint = root.get_transform_by_name(name)
        parent = joint.get_parent() if joint is not None else None
        if joint is None or parent is None:
            continue
        dx = float(joint.get_world_position()[0] - parent.get_world_position()[0])
        # 땅 쪽으로 도는 쪽 — 오른쪽을 보는 뼈는 − 가 아래, 왼쪽을 보는 뼈는 + 가 아래 (11/5 뱀 목 마디 ±40° 렌더로 확인).
        # 냄새 맡기에서 얼굴이 돌아간 것은 부호가 아니라 각(70°)이 커서였다
        down = -1.0 if dx >= 0 else 1.0
        # curl_mid: 가운데 마디를 두고 양끝이 위로 말려 올라간다(∪) — 머리를 축으로 말면 몸 전체가 일어서듯 돌았다 (11/9)
        bend = (-down * curl * (i - (len(bones) - 1) / 2)) if wave.get("curl_mid") else curl * (i + 1)
        angle = (joint.starting_theta + down * float(lowers[i]) + float(amps[i]) * np.sin(t - float(wave.get("lag", 0.7)) * i)
                 + bend + turn)
        rt.char_joint_to_orientation[name] = (angle % 360.0).astype(np.float32)
    if not wave.get("keep_root"):
        # 몸 위치는 사람 동작 데이터(박자용)의 오르내림 · 흔들림을 따르지 않는다 — 제자리에 둔다
        rt.char_root_positions[:] = rt.char_root_positions[0]
    if wave.get("keep_ground"):
        _keep_ground(drawing, bones)
    ends = [root.get_transform_by_name(m) for m in (bones[0], bones[-1])]
    if any(e is None for e in ends):
        return
    head, tail = (np.asarray(e.get_world_position()[:2], float) for e in ends)
    m = rt.char_root_positions.shape[0]
    if turn.any():
        # 뿌리(머리) 둘레로 돌면 몸이 크게 원을 그린다 — 몸 가운데가 제자리에 있게 뿌리를 옮긴다
        root_xy = np.asarray(root.get_world_position()[:2], float)
        v = (head + tail) / 2 - root_xy
        a = np.radians(360.0 * float(wave.get("spin", 0)) * np.arange(m) / m)
        rv = np.stack([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)], axis=1)
        rt.char_root_positions[:, :2] += (v - rv).astype(np.float32)
    bob = float(wave.get("bob", 0))
    if bob:
        # 몸 전체가 천천히 오르내린다 (지느러미형 헤엄, 11/5 사용자 「헤엄치는 느낌」) — 몸길이 대비 폭, 한 동작에 한 번
        length = float(np.linalg.norm(head - tail))
        # bob_by: 폭을 잴 두 관절. 날개형 기쁨은 날개 끝 사이(그림마다 크게 다름)로 재면 날개를 크게 그린 새는
        # 칸 밖으로 날아가 잘렸다, 몸통 · 머리 사이로 잰다 (11/6 새 그림 3장)
        by = [root.get_transform_by_name(n) for n in wave.get("bob_by") or ()]
        if len(by) == 2 and all(b is not None for b in by):
            length = float(np.linalg.norm(np.asarray(by[0].get_world_position()[:2], float)
                                          - np.asarray(by[1].get_world_position()[:2], float)))
        rt.char_root_positions[:, 1] += (bob * length * np.sin(2 * np.pi * np.arange(m) / m)).astype(np.float32)


def _keep_ground(drawing, bones: list[str]) -> None:  # noqa: ANN001
    """몸이 땅(그린 자세의 가장 낮은 곳) 밑으로 내려가지 않게 장마다 몸을 올린다 (11/9 「뱀이 몸을 말 때 땅 밑으로
    내려갔다가 돌며 말림」). 말기는 머리(뿌리)를 축으로 꼬리 쪽 마디를 돌려서 꼬리가 땅 밑으로 휘었다.
    마디 방향(그린 방향 + 바뀐 각)으로 장마다 마디 끝 자리를 이어 계산해, 가장 낮은 점이 처음보다 내려간 만큼 올린다"""
    root, rt = drawing.rig.root_joint, drawing.retargeter
    n = rt.char_root_positions.shape[0]
    joints = [root.get_transform_by_name(b) for b in bones]
    if any(j is None or j.get_parent() is None for j in joints):
        return
    start = np.asarray(joints[0].get_parent().get_world_position()[:2], float)
    rest = [np.asarray(j.get_world_position()[:2], float) - np.asarray(j.get_parent().get_world_position()[:2], float)
            for j in joints]
    low0 = min([start[1]] + [float(np.asarray(j.get_world_position()[:2], float)[1]) for j in joints])
    xs0 = [start[0]] + [float(np.asarray(j.get_world_position()[:2], float)[0]) for j in joints]
    mid0 = (min(xs0) + max(xs0)) / 2
    lift = np.zeros((n, 2), np.float32)
    for f in range(n):
        p, pts = start.copy(), [start.copy()]
        for j, v, b in zip(joints, rest, bones):
            a = np.radians(float(rt.char_joint_to_orientation[b][f]) - j.starting_theta)
            p = p + np.array([v[0] * np.cos(a) - v[1] * np.sin(a), v[0] * np.sin(a) + v[1] * np.cos(a)])
            pts.append(p)
        q = np.array(pts)
        # 가로는 몸 가운데를 제자리에 (말면서 옆으로 미끄러지지 않게)
        lift[f] = (mid0 - (q[:, 0].min() + q[:, 0].max()) / 2, max(0.0, low0 - q[:, 1].min()))
    rt.char_root_positions[:, :2] += lift


def _apply_rest(drawing, wave: dict) -> None:  # noqa: ANN001
    """가만히(숨쉬기) — 그린 자세 그대로 제자리에서 위아래로만 살짝 (11/9 형태 버그 4 「가만히에서 몸이 좌우로 흔들림」).

    사람 숨쉬기 동작(CMU 140_06)은 서 있는 사람이 몸무게를 옮기며 상체를 좌우로 기울인다 — 붙이면 사람형은 좌우로 12px,
    4족형은 몸통 · 다리가 2px 떨리고, 사슬형 숨쉬기 물결은 머리에서 꼬리로 흘러 제자리에서 기어가듯 좌우로 움직였다.
    모든 뼈를 그림의 원래 방향(starting_theta)에 두고, 몸 위치는 첫 장 그대로 + 위로 살짝 (rest_bob, 몸 키 대비).
    한 동작에 rest_cycles 번 들숨 · 날숨 — 첫 장이 쉬는 자세라 멈춘 그림과 같다. center 면 몸 가운데를 칸 가운데로(사슬형, _center_body). 꼬리 흔들기(4족)는 이 뒤에 그대로 넣는다"""
    root, rt = drawing.rig.root_joint, drawing.retargeter
    n = rt.char_root_positions.shape[0]
    for name in list(rt.char_joint_to_orientation):
        j = root.get_transform_by_name(name)
        if j is not None:
            rt.char_joint_to_orientation[name] = np.full(n, j.starting_theta % 360.0, np.float32)
    rt.char_root_positions[:] = rt.char_root_positions[0]
    pts = np.asarray(drawing.rig.get_joints_2D_positions(), float)
    tall = float(np.ptp(pts[:, 1])) if len(pts) else 0.0
    breath = 0.5 - 0.5 * np.cos(2 * np.pi * float(wave.get("rest_cycles", 1)) * np.arange(n) / n)
    rt.char_root_positions[:, 1] += (float(wave.get("rest_bob", 0.012)) * tall * breath).astype(np.float32)
    jitter = float(wave.get("rest_jitter", 0))
    if jitter:
        # 떨기 (4족, 11/9) — 몸 전체를 잘게 덜덜(위아래). 다리만 후들거리면 다리가 짧은 그림에서 거의 안 보였다.
        # 좌우로 흔들면 렌더 칸 「제자리」 손질(_in_place)이 몸 가로 이동을 지워 아무것도 안 보였다
        k = np.arange(n)
        rt.char_root_positions[:, 1] += (jitter * tall * np.abs(np.sin(np.pi * k / 2.0))).astype(np.float32)
    hops = int(wave.get("rest_hops", 0))
    if hops:
        # 깡충 (4족 기쁨 · 점프, 11/9) — 한 동작에 hops 번, 땅에서 뛰어올랐다 내려온다. 사람 점프 동작은 몸통이 돌아 얼굴이 뒤틀렸다
        up = np.clip(np.sin(2 * np.pi * hops * np.arange(n) / n), 0, None) ** 1.5
        rt.char_root_positions[:, 1] += (float(wave.get("rest_hop", 0.25)) * tall * up).astype(np.float32)


def _thick_legs(drawing, pos) -> float:  # noqa: ANN001
    """다리 굵기: 무릎 높이에서 다리와 이어진 그물 점들의 가로 폭 / 다리 길이 (다리 넷 중 가장 큰 값).
    붙여 그린 한 쌍은 둘이 한 덩어리라 둘을 합친 폭이 나온다 — 따로 흔들면 찢기는 그림이라 그것도 굵은 것으로 본다"""
    try:
        v = np.asarray(drawing.mesh["vertices"], float)
        step = 1.0 / 40 * 1.2   # Meta 안쪽 점 간격(그림 칸 / 40)보다 조금 넓은 틈이면 떨어진 것
        worst = 0.0
        for e in ("f", "b"):
            for sd in ("left", "right"):
                h, k, f = pos(f"{sd}_hip_{e}"), pos(f"{sd}_knee_{e}"), pos(f"{sd}_foot_{e}")
                length = float(np.linalg.norm(f - h))
                if length <= 1e-6:
                    continue
                d = (f - h) / length
                nrm = np.array([-d[1], d[0]])
                # 무릎 높이에서 잰다 — 넓적다리 위쪽은 배와 이어져 몸통 폭까지 잡혔다 (가는 다리 고양이가 3.3)
                rel = v - k
                band = (np.abs(rel @ d) < 0.06 * length) & (np.abs(rel @ np.array([-d[1], d[0]])) < 0.8 * length)
                side = np.sort(rel[band] @ nrm)
                if side.size == 0:
                    continue
                c = int(np.argmin(np.abs(side)))
                lo = hi = c
                while lo > 0 and side[lo] - side[lo - 1] < step:
                    lo -= 1
                while hi < side.size - 1 and side[hi + 1] - side[hi] < step:
                    hi += 1
                worst = max(worst, float(side[hi] - side[lo]) / length)
        return worst
    except Exception:  # noqa: BLE001
        return 0.0


def _apply_gait(drawing, gait: dict) -> None:  # noqa: ANN001
    """4족 걷기 · 뛰기 — 다리 방향을 옆모습 걸음으로 직접 넣는다 (11/9 형태 버그 3 「4족 걷기가 정면처럼 보임」).

    사람 걷기 동작(CMU)을 네 다리에 나눠 붙이면 같은 쪽 앞 · 뒷다리가 똑같이 움직이고, 사람 다리가 몸 앞뒤로 엇갈리는 모양이
    그대로 들어와 다리가 X 자로 모였다 벌어졌다 — 옆으로 걸어가는 것이 아니라 제자리에서 다리를 꼬는 정면 걸음처럼 보였다.
    다리마다 넓적다리(엉덩이 → 무릎)를 머리 쪽 · 꼬리 쪽으로 흔들고, 앞으로 내딛는 동안 아랫다리를 굽혀 발을 든다.
    대각선 두 다리(앞 1 · 뒷 2, 앞 2 · 뒷 1)가 같이 움직인다 (네 발 동물 걸음). 몸통 · 목은 그림 방향 그대로, 몸은 제자리에서
    걸음마다 살짝 오르내린다 (옮겨 가기는 장면이 한다). gait 칸: swing · bend (도), cycles (한 동작에 걸음 주기), bob (몸 키 대비)
    장 수의 정수배 주기라 반복해도 끊기지 않는다"""
    root, rt = drawing.rig.root_joint, drawing.retargeter
    n = rt.joint_positions.shape[0]
    head = 1.0 if root.get_transform_by_name("neck").get_world_position()[0] >=         root.get_transform_by_name("hip").get_world_position()[0] else -1.0
    t = 2 * np.pi * float(gait.get("cycles", 2)) * np.arange(n) / n
    swing, bend = float(gait.get("swing", 22)), float(gait.get("bend", 30))
    bob = float(gait.get("bob", 0.02))
    # 다리가 짧고 굵은 그림(다리 길이 < 몸길이(엉덩이~목)의 SHORT_LEG 배)은 다리를 크게 흔들면 그물이 늘어나 다리 끝이
    # 말리며 후들거렸다 (11/9 갈색 강아지 · 악어 0.42, 사용자 「게다리」). 다리는 작게 · 무릎은 펴고, 몸을 더 통통 띄운다.
    # 다리가 긴 그림(호랑이 0.93 · 고양이 0.76 · 화면 강아지 0.59)은 그대로
    def _pos(n: str) -> np.ndarray:
        return np.asarray(root.get_transform_by_name(n).get_world_position()[:2], float)
    try:
        legs = [float(np.linalg.norm(_pos(f"{s}_hip_{e}") - _pos(f"{s}_foot_{e}")))
                for s in ("left", "right") for e in ("f", "b")]
        body_len = float(np.linalg.norm(_pos("neck") - _pos("hip")))
        short = body_len > 0 and np.mean(legs) / body_len < float(gait.get("short_leg", 0.5))
    except Exception:  # noqa: BLE001
        short = False
    thick = _thick_legs(drawing, _pos)
    logging.getLogger(__name__).info("4족 다리 굵기 %.2f 짧은 다리 %s", thick, short)
    if short:
        swing, bend, bob = swing * 0.45, 0.0, bob * 2.5
    elif thick > float(gait.get("thick_leg", 0.6)):
        # 다리가 굵거나 두 다리를 붙여 그린 그림(무릎 높이 다리 폭 / 다리 길이 > 0.6 — 굵게 붙은 갈색 개 0.73, 가는 다리 고양이 0.38 · 호랑이 0.44)은 크게 흔들면 넓은 다리 그물이 비틀려
        # 벌어질 때 가늘고 길게 늘어나고 모일 때 뭉쳤다 (11/12 사용자 갈색 개 「펴졌다 얇아졌다」). 흔들기 · 무릎 굽힘을 줄인다
        swing, bend = swing * 0.55, bend * 0.35
    for name in ("hip", "torso", "neck"):
        j = root.get_transform_by_name(name)
        if j is not None and name in rt.char_joint_to_orientation:
            rt.char_joint_to_orientation[name] = np.full(n, j.starting_theta % 360.0, np.float32)
    for leg, off in (("right_{}_f", 0.0), ("left_{}_b", 0.0), ("left_{}_f", np.pi), ("right_{}_b", np.pi)):
        kn, fn = leg.format("knee"), leg.format("foot")
        knee, foot = root.get_transform_by_name(kn), root.get_transform_by_name(fn)
        if knee is None or foot is None:
            continue
        # 한 걸음 = 딛기(duty, 발이 땅에 닿은 채 다리가 머리 쪽 → 꼬리 쪽으로 고르게 밀린다) + 내딛기(나머지, 발을 들고
        # 빠르게 머리 쪽으로). 시계추처럼 앞뒤로 같은 빠르기로 흔들면 발이 땅을 딛는 때가 없어 다리가 팔랑거렸다 (11/9 호랑이)
        u = ((t + off) / (2 * np.pi)) % 1.0
        duty = float(gait.get("duty", 0.65))
        stance = u < duty
        s_st = u / duty
        s_sw = (u - duty) / (1 - duty)
        # 각은 반시계가 + — 머리가 오른쪽이면 + 가 머리 쪽으로 내딛기
        thigh = head * swing * np.where(stance, 1 - 2 * s_st, -1 + 2 * (0.5 - 0.5 * np.cos(np.pi * s_sw)))
        lift = bend * np.where(stance, 0.0, np.sin(np.pi * np.clip(s_sw, 0, 1)))   # 내딛는 동안만 발을 든다
        rt.char_joint_to_orientation[kn] = ((knee.starting_theta + thigh) % 360.0).astype(np.float32)
        rt.char_joint_to_orientation[fn] = ((foot.starting_theta + thigh - head * lift) % 360.0).astype(np.float32)
    rt.char_root_positions[:] = rt.char_root_positions[0]
    pts = np.asarray(drawing.rig.get_joints_2D_positions(), float)
    tall = float(np.ptp(pts[:, 1])) if len(pts) else 0.0
    rt.char_root_positions[:, 1] += (bob * tall * np.abs(np.sin(t))).astype(np.float32)


def _folded_arms(drawing) -> list[str]:  # noqa: ANN001
    """몸 앞에 모아 그린 팔(팔짱 · 배 앞에 모은 손) — 그린 그림에서 손이 두 어깨 사이 안쪽, 어깨와 엉덩이 사이 높이에 있는 쪽 (사람형만).
    그 팔은 몸통과 한 덩어리로 그려져, 동작이 팔을 몸 옆으로 내리면 팔과 몸통이 함께 찢어졌다 (11/12 사용자 「걸음을 멈췄어요」 쪽).
    자리는 그린 관절(char_cfg) — 뼈대 자리는 이미 동작 첫 장으로 움직여 있다"""
    try:
        at = {j["name"]: np.asarray(j["loc"], float) for j in drawing.char_cfg.skeleton}
    except Exception:  # noqa: BLE001
        return []
    rs, ls, hip = at.get("right_shoulder"), at.get("left_shoulder"), at.get("hip")
    if rs is None or ls is None or hip is None:
        return []
    x0, x1 = sorted((rs[0], ls[0]))
    width = x1 - x0
    top, bottom = sorted(((rs[1] + ls[1]) / 2, hip[1]))
    out = []
    for side in ("right", "left"):
        h = at.get(f"{side}_hand")
        if h is None or width <= 0:
            continue
        if x0 + 0.15 * width < h[0] < x1 - 0.15 * width and top < h[1] < bottom:
            out.append(side)
    return out


def _hold_folded_arms(drawing, root, rt) -> None:  # noqa: ANN001
    """몸 앞에 모은 팔은 어느 동작에서든 그린 자세 그대로 — 다리 · 몸만 움직인다 (팔짱 낀 채 걷기). 손 흔들기 · 만세는 약해지지만 찢어지지 않는다"""
    folded = _folded_arms(drawing)
    if folded:
        logging.getLogger(__name__).warning("몸 앞에 모은 팔 %s: 그린 자세 그대로", folded)
    for side in folded:
        for part in ("shoulder", "elbow", "hand"):
            name = f"{side}_{part}"
            j = root.get_transform_by_name(name)
            if j is not None and name in rt.char_joint_to_orientation:
                rt.char_joint_to_orientation[name] = np.full(rt.char_root_positions.shape[0], j.starting_theta % 360.0, np.float32)


class _tail_wag:
    """옆모습 형태(4족형)의 동작 손질 — 머리 쪽으로 걷게 뒤집기 · 꼬리 흔들기.

    뒤집기 (11/4 사용자 「얼굴 방향에 따라 < 인지 > 인지」): 동작 파일은 화면 오른쪽으로 걷는다. 머리가 왼쪽인 그림
    (실제 그림 8장 모두)은 꼬리 쪽으로 걸음을 내디뎌 뒤로 걸었다 — 몸이 오른쪽으로 흘렀다(재 보니 +0.011 · +0.015).
    목이 엉덩이보다 왼쪽이면 모든 뼈 방향을 세로축에 대해 뒤집고(각 → 360 − 각) 몸통 가로 이동도 뒤집는다.

    겹친 다리 (11/4 운영 소 그림): 다리를 둘만 그린 그림은 분석이 한 다리에 한 쌍을 겹쳐 놓는다. 동작 파일의 두 다리는
    번갈아 움직여 그림 다리 하나가 양쪽으로 찢어졌다 — 한 쌍이 겹쳐 있으면 왼쪽도 오른쪽 다리 움직임을 그대로 따른다.

    꼬리 흔들기 (11/4 사용자 「꼬리도 적용시켜줘」).

    동작 파일은 사람 것이라 꼬리 움직임이 없다. Meta 는 대응표에 있는 뼈만 사람 뼈 방향을 「그대로」 따라가서
    꼬리를 팔에 대응시키면 팔처럼 아래로 늘어진다 — 그래서 대응표에 넣지 않고, 동작 계산이 끝난 뒤
    꼬리 두 마디의 방향을 「그림의 원래 방향 ± 흔들기」로 직접 넣는다. 한 동작에 두 번, 끝 마디는 반만큼 더 휜다
    (팀원 실험 hankan-experiments animal-monster/quadruped.py 의 TAIL_WAG 14° · sin(4πt) 와 같다).
    장 수의 정수배 주기라 반복해도 끊기지 않는다. 꼬리가 없는 그림은 점이 엉덩이에 겹쳐 길이가 0 — 돌려도 아무것도 안 끈다"""

    def __init__(self, wave: dict | None = None) -> None:
        import animated_drawings.model.animated_drawing as ad

        self._cls = ad.AnimatedDrawing
        self._orig = ad.AnimatedDrawing._initialize_retargeter_bvh
        orig = self._orig

        def init(drawing, *a, **kw):  # noqa: ANN001 — Meta 쪽 서명 그대로
            orig(drawing, *a, **kw)
            root = drawing.rig.root_joint
            rt = drawing.retargeter
            mid, tip = root.get_transform_by_name("tail_mid"), root.get_transform_by_name("tail_tip")
            if mid is None or tip is None:      # 꼬리 관절이 없는 형태(2족형 · 사슬형 · 지느러미형)는 물결만
                _hold_folded_arms(drawing, root, rt)
                if wave and wave.get("mirror"):
                    # 옆면 걷기(cmu1_side)는 다리가 화면 왼쪽으로 내딛는다 — 이 뼈들만 세로축에 대해 뒤집어 오른쪽으로 (11/9)
                    for name in wave["mirror"]:
                        if name in rt.char_joint_to_orientation:
                            rt.char_joint_to_orientation[name] = ((360.0 - rt.char_joint_to_orientation[name]) % 360.0).astype(np.float32)
                if wave and wave.get("straight_legs"):
                    # 무릎을 굽히지 않는다 — 아랫다리를 넓적다리 방향 그대로 (11/9 사용자 「옷 때문에 깨진다」). 긴 바지 · 치마가 무릎에서
                    # 한 덩어리로 꺾여 고무처럼 휘었다. 엉덩이에서만 흔드는 종이 인형 걸음
                    for side in ("right", "left"):
                        k, f = f"{side}_knee", f"{side}_foot"
                        if k in rt.char_joint_to_orientation and f in rt.char_joint_to_orientation:
                            j_k, j_f = root.get_transform_by_name(k), root.get_transform_by_name(f)
                            bent = (j_f.starting_theta - j_k.starting_theta) if j_k is not None and j_f is not None else 0.0
                            rt.char_joint_to_orientation[f] = ((rt.char_joint_to_orientation[k] + bent) % 360.0).astype(np.float32)
                if wave and (wave.get("hold") or wave.get("mirror") or wave.get("straight_legs")):   # hold: 이 뼈들만 그린 방향 그대로
                    for name in wave.get("hold") or ():
                        j = root.get_transform_by_name(name)
                        if j is not None and name in rt.char_joint_to_orientation:
                            rt.char_joint_to_orientation[name] = np.full(rt.char_root_positions.shape[0], j.starting_theta % 360.0, np.float32)
                    return
                if wave and wave.get("rest"):
                    _apply_rest(drawing, wave)
                    if wave.get("bones"):               # 그린 자세에서 이 마디만 (사람 두리번 — 고개 · 몸 갸웃)
                        _apply_wave(drawing, {**wave, "keep_root": True})
                    if wave.get("center"):
                        _center_body(drawing)
                    return
                if wave:
                    _apply_wave(drawing, wave)
                    _center_body(drawing)
                return
            neck, hip = root.get_transform_by_name("neck"), root.get_transform_by_name("hip")
            if neck is not None and hip is not None and neck.get_world_position()[0] < hip.get_world_position()[0]:
                for k, v in rt.char_joint_to_orientation.items():
                    rt.char_joint_to_orientation[k] = ((360.0 - v) % 360.0).astype(np.float32)
                x = rt.char_root_positions[:, 0]
                rt.char_root_positions[:, 0] = 2 * x[0] - x
            if wave and wave.get("rest"):
                _apply_rest(drawing, wave)
            if wave and wave.get("gait") and neck is not None and hip is not None:
                _apply_gait(drawing, wave["gait"])
            # 몸 가운데를 렌더 칸 가운데로 — 대응표 시작 자리(x −0.3) 때문에 몸이 칸 왼쪽 1/4 에 놓여 옆으로 긴 그림(호랑이)은
            # 머리가 칸 왼쪽 끝에 닿았다 (11/9). 장면은 sprite_box 로 자리를 잡아 화면 위치는 그대로다
            _center_body(drawing)
            if neck is not None and hip is not None:
                pos = lambda n: np.asarray(root.get_transform_by_name(n).get_world_position()[:2])  # noqa: E731
                tol = 0.03 * float(np.linalg.norm(pos("neck") - pos("hip")))
                for side in ("b", "f"):
                    parts = ("hip", "knee", "foot")
                    if all(float(np.linalg.norm(pos(f"left_{k}_{side}") - pos(f"right_{k}_{side}"))) <= tol for k in parts):
                        for k in ("knee", "foot"):
                            src = rt.char_joint_to_orientation.get(f"right_{k}_{side}")
                            if src is not None:
                                rt.char_joint_to_orientation[f"left_{k}_{side}"] = src.copy()
            # 직접 넣는 각도(냄새 맡기 …)는 뒤집기 뒤에 — 그림 자신의 방향 기준이라 뒤집으면 반대로 숙였다
            if wave and wave.get("bones"):
                _apply_wave(drawing, wave)
            n = rt.joint_positions.shape[0]
            wag = TAIL_WAG_DEG * np.sin(4 * np.pi * np.arange(n) / n)
            rt.char_joint_to_orientation["tail_mid"] = (mid.starting_theta + wag).astype(np.float32)
            rt.char_joint_to_orientation["tail_tip"] = (tip.starting_theta + 1.5 * wag).astype(np.float32)

        ad.AnimatedDrawing._initialize_retargeter_bvh = init

    def release(self) -> None:
        self._cls._initialize_retargeter_bvh = self._orig


class _thin_frames:
    """묶음에 담을 장만 렌더한다 (11/3 저녁 「렌더링 속도」 — 걷기 78장 · 춤 120장 중 48장만 묶음에 들어가고 나머지는 버렸다).

    Meta 가 동작 파일(BVH)을 읽은 직후 SPRITE_MAX_FRAMES 장을 고르게 골라 남기고, 장 사이 시간을 그만큼 늘린다 —
    묶음에 들어가던 장과 같은 자세 · 같은 길이라 화면(장면 층, 12장/초) 움직임은 그대로다.
    몸통 이동은 장 사이 차이를 더해 가므로 장을 건너뛰어도 같은 자리로 간다."""

    def __init__(self) -> None:
        from animated_drawings.model.bvh import BVH

        self._cls = BVH
        self._orig = BVH.__dict__["from_file"]
        orig = self._orig.__func__

        def from_file(cls, *a, **kw):  # noqa: ANN001
            bvh = orig(cls, *a, **kw)
            n = bvh.frame_max_num
            if n > SPRITE_MAX_FRAMES:
                idx = np.linspace(0, n - 1, SPRITE_MAX_FRAMES).round().astype(int)
                bvh.pos_data, bvh.rot_data = bvh.pos_data[idx], bvh.rot_data[idx]
                bvh.frame_time = bvh.frame_time * (n - 1) / (SPRITE_MAX_FRAMES - 1)
                bvh.frame_max_num = SPRITE_MAX_FRAMES
                bvh.cur_frame = 0
                bvh.apply_frame(0)
            return bvh

        BVH.from_file = classmethod(from_file)

    def release(self) -> None:
        self._cls.from_file = self._orig


class _in_place:
    """걷기 · 뛰기에서 몸 전체가 옆으로 가는 것을 뺀다 (프레임 묶음일 때만, 11/3 사용자 「움직일 때 잘려나가」).

    Meta 는 BVH 의 몸통 이동을 캐릭터에도 그대로 옮겨서 뛰기 · 걷기는 렌더 칸 안에서 오른쪽으로 흘러가
    칸 끝(512px)을 넘는다 — 넘은 부분은 묶음에 없다. 장면 층은 화면이 주인공을 옮기므로(걸어 들어오기 ·
    목표로 가기) 렌더는 제자리면 된다. 가로 이동의 직선 성분만 빼서 몸이 살짝 흔들리는 것과 위아래(점프)는 남긴다."""

    def __init__(self) -> None:
        from animated_drawings.model.retargeter import Retargeter

        self._cls = Retargeter
        self._orig = Retargeter.scale_root_positions_for_character
        orig = self._orig

        def scale(rt, *a, **kw):  # noqa: ANN001 — Meta 쪽 서명 그대로
            orig(rt, *a, **kw)
            x = rt.char_root_positions[:, 0]
            if len(x) > 1:
                t = np.arange(len(x), dtype=np.float32)
                slope, icpt = np.polyfit(t, x, 1)
                rt.char_root_positions[:, 0] = x - (slope * t + icpt) + x[0]

        Retargeter.scale_root_positions_for_character = scale

    def release(self) -> None:
        self._cls.scale_root_positions_for_character = self._orig


def _write_sprite(frames: list[np.ndarray], path: Path) -> dict:
    """RGBA 프레임들 → 투명 WebP 한 장. 프레임이 많으면 고르게 골라 SPRITE_MAX_FRAMES 장까지.

    {"frames": 장 수, "box": 첫 장에서 그림이 차지하는 자리 [left, top, right, bottom] (칸 대비 0~1)} —
    화면이 장면에 세울 때 키 · 발끝 · 가운데를 맞춘다. 화면은 다른 주소의 그림 픽셀을 읽을 수 없어서(CORS) 여기서 잰다"""

    from PIL import Image

    if not frames:
        raise RenderError("RENDER_FAILED: no frames for sprite")
    if len(frames) > SPRITE_MAX_FRAMES:
        idx = np.linspace(0, len(frames) - 1, SPRITE_MAX_FRAMES).round().astype(int)
        frames = [frames[i] for i in idx]
    n = len(frames)
    cols = SPRITE_COLS   # 늘 8칸 — 장이 적어도 빈칸(투명)을 둔다. 화면이 「가로 ÷ 8」로 칸 크기를 안다
    rows = (n + SPRITE_COLS - 1) // SPRITE_COLS
    sheet = Image.new("RGBA", (cols * SPRITE_CELL, rows * SPRITE_CELL), (0, 0, 0, 0))
    for i, f in enumerate(frames):
        cell = Image.fromarray(f, "RGBA")
        if cell.size != (SPRITE_CELL, SPRITE_CELL):   # 칸 크기로 그렸으면 줄이지 않는다
            cell = cell.resize((SPRITE_CELL, SPRITE_CELL), Image.LANCZOS)
        sheet.paste(cell, ((i % SPRITE_COLS) * SPRITE_CELL, (i // SPRITE_COLS) * SPRITE_CELL))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "WEBP", quality=86, method=4)
    alpha = frames[0][..., 3]
    ys, xs = np.nonzero(alpha > 24)
    h, w = alpha.shape
    box = [0.0, 0.0, 1.0, 1.0] if not len(xs) else [
        round(xs.min() / w, 4), round(ys.min() / h, 4), round((xs.max() + 1) / w, 4), round((ys.max() + 1) / h, 4)]
    return {"frames": n, "box": box}


def _gif_to_mp4(gif_path: Path, mp4_path: Path) -> None:
    """OpenCV가 H.264를 지원하지 않으므로 ffmpeg으로 변환한다.

    GIF 는 바탕이 투명한데 MP4 는 투명을 담지 못한다. 투명 자리를 자홍(#FF00FF)으로 채워 두면
    화면이 그 색만 지워 장소 배경 위에 캐릭터를 세운다 (frontend features/story/key.ts, 2026-11-03).
    흰색으로 두면 색칠하지 않은 몸 · 눈의 흰색과 구별되지 않는다.
    """

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(gif_path),
            "-movflags", "faststart",
            "-pix_fmt", "yuv420p",
            # 자홍 판 위에 GIF 를 얹는다. H.264는 가로·세로가 짝수여야 한다
            "-filter_complex",
            "[0:v]format=rgba,split[fg][b];[b]drawbox=c=0xFF00FF@1:t=fill[bg];"
            "[bg][fg]overlay=format=auto,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(mp4_path),
        ],
        capture_output=True,
        timeout=300,
    )

    if result.returncode != 0 or not mp4_path.exists():
        detail = result.stderr.decode("utf-8", "replace")[-500:]
        raise RenderError(f"ENCODING_FAILED: ffmpeg failed: {detail}")


def _frames_to_mp4(frames: list[np.ndarray], duration_ms: int, mp4_path: Path) -> None:
    """받아 둔 RGBA 프레임 → 자홍(#FF00FF) 바탕 MP4 (_gif_to_mp4 와 같은 결과, GIF 를 거치지 않는다 — 11/3 저녁)."""

    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2   # H.264 는 가로 · 세로가 짝수
    bg = np.array([255, 0, 255], np.float32)
    raw = bytearray()
    for f in frames:
        a = f[:h2, :w2, 3:4].astype(np.float32) / 255.0
        raw += (f[:h2, :w2, :3].astype(np.float32) * a + bg * (1 - a)).round().astype(np.uint8).tobytes()
    fps = max(1.0, 1000.0 / max(1, duration_ms))
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w2}x{h2}", "-r", f"{fps:.4f}", "-i", "-",
         "-movflags", "faststart", "-pix_fmt", "yuv420p", str(mp4_path)],
        input=bytes(raw), capture_output=True, timeout=300,
    )
    if result.returncode != 0 or not mp4_path.exists():
        detail = result.stderr.decode("utf-8", "replace")[-500:]
        raise RenderError(f"ENCODING_FAILED: ffmpeg failed: {detail}")


def _motion_cfg(cfg_path: Path, work_dir: Path) -> Path:
    """motion 설정을 렌더에 넘길 경로로 돌려준다.

    Meta 설정의 BVH 경로는 Meta 저장소 기준이라 그대로 쓴다. 우리 설정(*_gentle)은
    설정 파일 옆의 BVH 를 가리키므로 절대 경로로 바꾼 사본을 작업 폴더에 둔다.
    """

    cfg = yaml.safe_load(cfg_path.read_text())
    local_bvh = cfg_path.parent / cfg["filepath"]
    if not local_bvh.exists():
        return cfg_path
    cfg["filepath"] = str(local_bvh)
    out = work_dir / "motion_cfg.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def render_animation(
    image_path: str,
    mask_path: str,
    joints: list[dict],
    bbox: dict,
    motion: str,
    output_path: str,
    sprite_path: str | None = None,
) -> dict:
    """관절 좌표로 애니메이션 MP4를 만든다.

    sprite_path 가 있으면 같은 렌더의 프레임을 **바탕이 투명한** 프레임 묶음(WebP 한 장)으로도 저장한다 —
    화면이 장면 배경 · 물건과 층으로 쌓아 움직이기 위해서다 (프론트 S-09, 2026-11-03).
    MP4(흰 바탕)는 S-05 미리보기 · 예전 화면이 그대로 쓴다.
    """

    started_at = time.perf_counter()

    form = forms.form_or_default({j["name"] for j in joints})
    if form != forms.DEFAULT:
        # 2족형이 아닌 형태는 형태 정의의 동작 · 대응표 (pt_docs 19 6장). 없는 동작은 backend 가 대신 동작으로 바꿔 부른다
        spec = forms.get(form)
        if motion not in spec.get("motions", {}):
            raise RenderError(f"UNKNOWN_MOTION: {motion} for {form} (available: {', '.join(spec.get('motions', {}))})")
        motion_cfg_path = _GENTLE_MOTIONS / spec["motions"][motion]
        retarget_path = _GENTLE_MOTIONS / spec["retarget"]
    elif motion not in MOTIONS:
        raise RenderError(
            f"UNKNOWN_MOTION: {motion} (available: {', '.join(MOTIONS)})"
        )

    else:
        motion_cfg_path, retarget_path = MOTIONS[motion]

    work_dir = Path(tempfile.mkdtemp(prefix="drawtale-render-"))

    try:
        char_dir = prepare_character(image_path, mask_path, joints, bbox, work_dir)

        gif_path = work_dir / "video.gif"
        mvc_cfg = {
            "scene": {
                "ANIMATED_CHARACTERS": [
                    {
                        "character_cfg": str(char_dir / "char_cfg.yaml"),
                        "motion_cfg": str(_motion_cfg(motion_cfg_path, work_dir)),
                        "retarget_cfg": str(retarget_path),
                    }
                ]
            },
            "controller": {
                "MODE": "video_render",
                "OUTPUT_VIDEO_PATH": str(gif_path),
            },
            # 화면 없는 환경에서 렌더링하려면 반드시 필요하다
            "view": {"USE_MESA": True},
        }

        if sprite_path:
            # 프레임 묶음이 장면에서 작아 보이지 않게 조금 크게 그린다 (MP4 도 같은 크기가 된다).
            # 기본 카메라는 캐릭터가 화면 높이의 3할 남짓이라 장면에서 늘리면 흐려진다 — 가까이 당긴다.
            # 모든 동작이 같은 카메라라 같은 캐릭터면 크기 · 발끝 자리가 같다 (팔 · 점프 여유는 남긴다)
            mvc_cfg["view"]["WINDOW_DIMENSIONS"] = [SPRITE_WINDOW, SPRITE_WINDOW]
            # 옆으로 긴 형태(4족형)는 사람 기준 카메라면 칸 양끝이 잘린다 — 형태 정의에서 카메라를 물린다
            cam = forms.get(form).get("sprite_camera", {})
            mvc_cfg["view"]["CAMERA_POS"] = list(cam.get("pos", SPRITE_CAMERA_POS))
            mvc_cfg["view"]["CAMERA_FWD"] = list(cam.get("fwd", SPRITE_CAMERA_FWD))

        mvc_path = work_dir / "mvc_cfg.yaml"
        with open(mvc_path, "w") as f:
            yaml.dump(mvc_cfg, f)

        # 큰 얼굴을 원본 그대로 옮겨 붙일 그림인지 (face_lock — 머리가 크고 목 없는 그림만). 그런 그림은
        # 프레임을 늘 가로챈다(MP4 만 만들 때도) — 붙인 얼굴이 MP4 · 프레임 묶음 둘 다에 들어가게
        face = face_lock.plan(char_dir) if FACE_LOCK and form == forms.DEFAULT else None   # 큰 얼굴 붙이기는 사람 그림만
        capture = bool(sprite_path) or face is not None

        def run_meta():
            """Meta 렌더 한 번. 가로챈 프레임(_capture_frames) 또는 None(GIF 로 저장)."""
            # Meta 코드는 상대 경로를 자기 저장소 기준으로 해석한다
            cwd = os.getcwd()
            try:
                os.chdir(ANIMATED_DRAWINGS_DIR)
                import animated_drawings.render

                frames = _capture_frames() if capture else None
                still = _in_place() if sprite_path else None
                thin = _thin_frames() if sprite_path else None
                reuse = _reuse_setup() if REUSE_SETUP else None
                dense = _dense_mesh() if MESH_GRID != 40 else None
                # 물결로 움직이는 형태(사슬형 · 지느러미형)는 형태 정의의 그 동작 물결
                wag = _tail_wag(forms.get(form).get("waves", {}).get(motion))
                try:
                    animated_drawings.render.start(str(mvc_path))
                finally:
                    for patch in (frames, still, thin, reuse, wag, dense):
                        if patch is not None:
                            patch.release()
                return frames
            except RenderError:
                raise
            except Exception as e:
                raise RenderError(f"RENDER_FAILED: {e}") from e
            finally:
                os.chdir(cwd)

        face_used = False
        frames = None
        if face is not None:
            # 찾기 렌더(얼굴만) → 그리기 렌더(얼굴 자리 표시색) → 덮어 붙이기. 준비 계산은 틀 · 관절만 보므로
            # 두 번째부터는 _reuse_setup 이 다시 쓴다 (늘어나는 시간은 그리기 한 번). 어디서든 어긋나면 원래 텍스처로 다시 그린다
            try:
                face_lock.write_texture(char_dir, face, "locate")
                located = run_meta()
                where = face_lock.track(located.frames, face) if located is not None else None
                if where is not None:
                    face_lock.write_texture(char_dir, face, "paint")
                    frames = run_meta()
                    if frames is not None and len(frames.frames) == len(where):
                        frames.frames[:] = face_lock.apply(frames.frames, where, face)
                        face_used = True
            except RenderError:
                face_used = False
            if not face_used:
                face_lock.write_texture(char_dir, face, "original")
                frames = None
        if frames is None:
            frames = run_meta()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if frames is not None:
            if not frames.frames:
                raise RenderError("RENDER_FAILED: no output produced")
            _frames_to_mp4(frames.frames, frames.duration_ms, output)
        else:
            if not gif_path.exists():
                raise RenderError("RENDER_FAILED: no output produced")
            _gif_to_mp4(gif_path, output)
        sprite = _write_sprite(frames.frames, Path(sprite_path)) if frames is not None and sprite_path else None

        return {
            "format": "mp4",
            "motion": motion,
            "size_bytes": output.stat().st_size,
            "sprite": sprite,
            "face_lock": face_used,
            "processing_time_ms": round((time.perf_counter() - started_at) * 1000),
        }

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
