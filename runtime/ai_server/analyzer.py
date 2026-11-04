from pathlib import Path
import json
import os
import time
import requests
import cv2
import numpy as np
from skimage import measure
from scipy import ndimage


# Docker: http://torchserve:8080 (docker-compose.yml) / Local: http://127.0.0.1:8080
TORCHSERVE_URL = os.getenv("TORCHSERVE_URL", "http://127.0.0.1:8080")

MODEL_VERSION = "meta-animated-drawings-pretrained"
PIPELINE_VERSION = "0.1.0"

# 좌표계: 업로드된 원본 이미지의 픽셀. 원점 왼쪽 위, x 오른쪽, y 아래.
COORDINATE_SPACE = "image_px"

# Meta 원본 코드와 동일하게 분석 시 최대 dimension을 이 값으로 제한한다.
MAX_ANALYSIS_SIZE = 1000

# 마스크 생성 시 crop 이미지 둘레에 붙이는 흰 여백 (px)
MASK_PADDING = 20

# 선이 끊긴 그림을 위해 틈 메우기 강도를 단계적으로 올리며 재시도한다.
MASK_CLOSE_ATTEMPTS = (2, 6, 12, 20)

# 관절이 캐릭터 영역 위에 있어야 정상. 이 비율 미만이면 마스크가 뒤집힌 것으로 본다.
MASK_JOINT_HIT_RATIO = 0.6

# 마스크를 만들 때 bbox 바깥으로 더 잘라내는 여유 (px).
# 캐릭터가 bbox 경계에 닿으면 선이 잘려 배경이 새어 들어간다.
MASK_CROP_MARGIN = 30

# 바탕이 한 가지 색인지 볼 때 그림 둘레(px)와, 그 색들이 바탕색에서 떨어진 정도의 상한.
# 색종이 · 단색 바탕은 둘레가 고르다. 그늘진 사진은 이 값을 넘어 기존 방식(create_mask)으로 간다
BG_BORDER = 4
BG_UNIFORM_MAX = 30
# 바탕색에서 이만큼 떨어진 색만 캐릭터로 본다 (RGB 거리). 둘레가 고르지 않을수록 높인다
BG_DISTANCE_MIN = 60
# 바탕색 방식에서 덩어리를 셀 때 메우는 틈 (그림 긴 변 대비) · 함께 남기는 덩어리 크기 (가장 큰 것 대비)
BG_JOIN_GAP = 0.012
BG_KEEP_PART = 0.03
# 흰 종이로 보는 바탕: 가장 어두운 성분이 이 이상, 성분 차이가 이 이하 (그늘진 흰 종이 포함)
WHITE_PAPER_MIN = 200
WHITE_PAPER_TINT = 30

# 마스크 둘레에서 종이색으로 볼 거리 (RGB). 연한 색연필 선이 남을 만큼 좁게 (2026-11-03 실제 사진으로 정함:
# 24 는 연분홍 팔다리 선이 일부 깎였다)
PAPER_NEAR = 16
# 종이색을 깎은 뒤 원래 마스크의 이만큼도 안 남으면 깎기가 잘못된 것으로 보고 원래 마스크를 쓴다
PAPER_KEEP_MIN = 0.3

# 점수 기준. 실제 그림으로 잰 값은 docs/진행상황.md 「점수 기준」.
#
# 검출 점수는 낙서·동그라미에도 0.97 이상이 나와 사람인지 가리지 못한다.
# 사람 모양인지는 관절 점수 평균으로 본다 (낙서 0.10, 동그라미 0.23, 실제 그림 0.56 이상).
LOW_CONFIDENCE_SCORE = 0.3
LOW_CONFIDENCE_JOINT_MEAN = 0.35

# 가장 높은 것 말고도 이 점수 이상인 검출이 따로 떨어져 있으면 사람이 여럿인 그림이다.
# 두 사람을 나란히 그린 그림은 0.99, 한 사람 옆 라디오·조명은 0.54 였다.
MULTIPLE_MIN_SCORE = 0.8

# 두 번째 검출이 첫 번째 상자와 이만큼 이상 겹치면 같은 사람을 두 번 잡은 것으로 본다.
# 작은 상자의 넓이 중 겹친 비율이다.
SAME_CHARACTER_OVERLAP = 0.5

# Meta 17개 keypoint 중 15개 관절의 근거. 두 개면 그 중간점이다.
JOINT_SOURCES = {
    "hip": (11, 12),
    "torso": (5, 6),
    "neck": (0,),
    "right_shoulder": (6,),
    "right_elbow": (8,),
    "right_hand": (10,),
    "left_shoulder": (5,),
    "left_elbow": (7,),
    "left_hand": (9,),
    "right_hip": (12,),
    "right_knee": (14,),
    "right_foot": (16,),
    "left_hip": (11,),
    "left_knee": (13,),
    "left_foot": (15,),
}


def check_torchserve() -> bool:
    try:
        response = requests.get(
            f"{TORCHSERVE_URL}/ping",
            timeout=3
        )
        return response.status_code == 200

    except requests.RequestException:
        return False


def detect_character(img: np.ndarray) -> dict:
    """
    Meta Animated Drawings detector를 호출해서
    가장 높은 score의 character bbox를 반환한다.

    점수가 너무 낮으면 LOW_CONFIDENCE, 떨어진 자리에 다른 사람이 또 있으면
    MULTIPLE_CHARACTERS 로 멈춘다. 둘 다 아이에게 다른 그림을 부탁하는 경우다.

    bbox 좌표는 현재 분석 이미지
    (최대 1000px로 resize된 이미지) 기준이다.
    """

    img_bytes = cv2.imencode(".png", img)[1].tobytes()

    response = requests.post(
        f"{TORCHSERVE_URL}/predictions/drawn_humanoid_detector",
        files={"data": img_bytes},
        verify=False,
        timeout=30
    )

    if response.status_code >= 300:
        raise RuntimeError(
            f"Detector request failed: HTTP {response.status_code}"
        )

    detection_results = json.loads(response.content)

    if (
        isinstance(detection_results, dict)
        and detection_results.get("code") == 404
    ):
        raise RuntimeError(
            "drawn_humanoid_detector model is not available"
        )

    if not detection_results:
        raise RuntimeError(
            "NO_CHARACTER_DETECTED: No drawn humanoid detected"
        )

    detection_results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = detection_results[0]

    if best["score"] < LOW_CONFIDENCE_SCORE:
        raise RuntimeError(
            f"LOW_CONFIDENCE: Best detection score {best['score']:.2f}"
        )

    for other in detection_results[1:]:
        if other["score"] < MULTIPLE_MIN_SCORE:
            break
        if overlap_ratio(best["bbox"], other["bbox"]) < SAME_CHARACTER_OVERLAP:
            raise RuntimeError(
                f"MULTIPLE_CHARACTERS: Another character scored {other['score']:.2f}"
            )

    bbox = best["bbox"]

    left, top, right, bottom = [
        round(value) for value in bbox
    ]

    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "score": float(best["score"]),
    }


def overlap_ratio(a: list, b: list) -> float:
    """두 상자 [left, top, right, bottom] 이 겹친 넓이 / 작은 상자 넓이."""

    width = min(a[2], b[2]) - max(a[0], b[0])
    height = min(a[3], b[3]) - max(a[1], b[1])

    if width <= 0 or height <= 0:
        return 0.0

    smaller = min(
        (a[2] - a[0]) * (a[3] - a[1]),
        (b[2] - b[0]) * (b[3] - b[1]),
    )

    return width * height / smaller if smaller > 0 else 0.0

def create_mask(img: np.ndarray, close_iterations: int = 2) -> np.ndarray:
    """
    Meta Animated Drawings의 원본 segment() 방식을 사용해서
    캐릭터 영역의 binary mask를 생성한다.

    반환값:
        0   = 배경
        255 = 캐릭터
    """

    # 1. RGB 3채널을 grayscale처럼 변환
    img = np.min(img, axis=2)

    # 2. Adaptive Threshold
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        115,
        8
    )

    # 검정/흰색 반전
    img = cv2.bitwise_not(img)

    # 3. Morphological operation
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )

    # 선이 끊긴 그림은 이 단계에서 틈을 메워야 배경이 캐릭터 안으로 새지 않는다.
    img = cv2.morphologyEx(
        img,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=close_iterations
    )

    img = cv2.morphologyEx(
        img,
        cv2.MORPH_DILATE,
        kernel,
        iterations=2
    )

    # 4. Flood Fill을 위한 mask
    flood_mask = np.zeros(
        [img.shape[0] + 2, img.shape[1] + 2],
        np.uint8
    )

    flood_mask[1:-1, 1:-1] = img.copy()

    # Flood Fill 결과
    im_floodfill = np.full(
        img.shape,
        255,
        np.uint8
    )

    # 5. 이미지 가장자리에서 Flood Fill
    h, w = img.shape[:2]

    for x in range(0, w - 1, 10):
        cv2.floodFill(
            im_floodfill,
            flood_mask,
            (x, 0),
            0
        )

        cv2.floodFill(
            im_floodfill,
            flood_mask,
            (x, h - 1),
            0
        )

    for y in range(0, h - 1, 10):
        cv2.floodFill(
            im_floodfill,
            flood_mask,
            (0, y),
            0
        )

        cv2.floodFill(
            im_floodfill,
            flood_mask,
            (w - 1, y),
            0
        )

    # 이미지 테두리는 캐릭터가 아니라고 처리
    im_floodfill[0, :] = 0
    im_floodfill[-1, :] = 0
    im_floodfill[:, 0] = 0
    im_floodfill[:, -1] = 0

    # 6. 가장 큰 contour 찾기
    mask2 = cv2.bitwise_not(im_floodfill)

    mask = None
    biggest = 0

    contours = measure.find_contours(
        mask2,
        0.0
    )

    for contour in contours:
        x = np.zeros(
            mask2.T.shape,
            np.uint8
        )

        cv2.fillPoly(
            x,
            [np.int32(contour)],
            1
        )

        size = len(
            np.where(x == 1)[0]
        )

        if size > biggest:
            mask = x
            biggest = size

    # contour가 하나도 없으면 실패
    if mask is None:
        raise RuntimeError(
            "NO_CHARACTER_DETECTED: Mask contour not found"
        )

    # 7. 내부 구멍 채우기
    mask = ndimage.binary_fill_holes(
        mask
    ).astype(int)

    # 0 / 255 binary mask로 변환
    mask = 255 * mask.astype(np.uint8)

    return mask.T

def estimate_pose(cropped: np.ndarray, bbox: dict) -> list:
    """
    BBox로 잘라낸 이미지를 Pose Estimator에 전달하고
    Meta의 17개 keypoint를 15개 joint로 변환한다.

    Pose Estimator가 반환하는 좌표는 crop 좌표이므로
    bbox의 left/top을 더해서 분석 이미지 좌표로 변환한다.
    """

    img_bytes = cv2.imencode(".png", cropped)[1].tobytes()

    response = requests.post(
        f"{TORCHSERVE_URL}/predictions/drawn_humanoid_pose_estimator",
        files={"data": img_bytes},
        verify=False,
        timeout=60
    )

    if response.status_code >= 300:
        raise RuntimeError(
            f"Pose estimator request failed: HTTP {response.status_code}"
        )

    pose_results = json.loads(response.content)

    if (
        isinstance(pose_results, dict)
        and pose_results.get("code") == 404
    ):
        raise RuntimeError(
            "drawn_humanoid_pose_estimator model is not available"
        )

    if not pose_results:
        raise RuntimeError(
            "NO_CHARACTER_DETECTED: Pose estimation returned no result"
        )

    if len(pose_results) != 1:
        raise RuntimeError(
            "Pose estimation returned an unexpected number of skeletons"
        )

    keypoints = pose_results[0].get("keypoints", [])

    if len(keypoints) < 17:
        raise RuntimeError(
            f"Pose estimation returned only {len(keypoints)} keypoints"
        )

    # Meta Animated Drawings의 17개 keypoint
    kpts = np.array(keypoints)

    # crop 좌표 -> 현재 분석 이미지 좌표
    def point(index: int) -> tuple:
        x = round(float(kpts[index][0]) + bbox["left"])
        y = round(float(kpts[index][1]) + bbox["top"])
        return x, y

    # 두 관절의 중간점
    def midpoint(index1: int, index2: int) -> tuple:
        x = round(
            (float(kpts[index1][0]) + float(kpts[index2][0])) / 2
            + bbox["left"]
        )

        y = round(
            (float(kpts[index1][1]) + float(kpts[index2][1])) / 2
            + bbox["top"]
        )

        return x, y

    # Meta Animated Drawings skeleton 구조
    joints = [
        {
            "name": "hip",
            "x": midpoint(11, 12)[0],
            "y": midpoint(11, 12)[1],
        },
        {
            "name": "torso",
            "x": midpoint(5, 6)[0],
            "y": midpoint(5, 6)[1],
        },
        {
            "name": "neck",
            "x": point(0)[0],
            "y": point(0)[1],
        },

        {
            "name": "right_shoulder",
            "x": point(6)[0],
            "y": point(6)[1],
        },
        {
            "name": "right_elbow",
            "x": point(8)[0],
            "y": point(8)[1],
        },
        {
            "name": "right_hand",
            "x": point(10)[0],
            "y": point(10)[1],
        },

        {
            "name": "left_shoulder",
            "x": point(5)[0],
            "y": point(5)[1],
        },
        {
            "name": "left_elbow",
            "x": point(7)[0],
            "y": point(7)[1],
        },
        {
            "name": "left_hand",
            "x": point(9)[0],
            "y": point(9)[1],
        },

        {
            "name": "right_hip",
            "x": point(12)[0],
            "y": point(12)[1],
        },
        {
            "name": "right_knee",
            "x": point(14)[0],
            "y": point(14)[1],
        },
        {
            "name": "right_foot",
            "x": point(16)[0],
            "y": point(16)[1],
        },

        {
            "name": "left_hip",
            "x": point(11)[0],
            "y": point(11)[1],
        },
        {
            "name": "left_knee",
            "x": point(13)[0],
            "y": point(13)[1],
        },
        {
            "name": "left_foot",
            "x": point(15)[0],
            "y": point(15)[1],
        },
    ]

    # 관절마다 모델이 얼마나 확신하는지 (0~1). 중간점은 두 점 중 낮은 쪽을 쓴다.
    # 프론트가 낮은 관절을 어른에게 먼저 보여 준다 (S-05-05, S-06).
    def score(index: int) -> float | None:
        if len(kpts[index]) < 3:
            return None
        return min(1.0, max(0.0, float(kpts[index][2])))

    for joint in joints:
        scores = [score(i) for i in JOINT_SOURCES[joint["name"]]]
        joint["score"] = None if None in scores else round(min(scores), 3)

    return joints


def joints_on_mask_ratio(mask: np.ndarray, joints: list, offset_x: int, offset_y: int) -> float:
    """관절 15개 중 마스크(캐릭터 영역) 위에 있는 비율을 센다.

    마스크 안팎이 뒤집히면 관절 대부분이 배경 쪽에 놓이므로 이 값이 크게 떨어진다.
    """

    height, width = mask.shape[:2]
    hits = 0

    for joint in joints:
        x = round(joint["x"]) - offset_x
        y = round(joint["y"]) - offset_y

        if 0 <= x < width and 0 <= y < height and mask[y, x] > 127:
            hits += 1

    return hits / len(joints) if joints else 0.0


def uniform_background(img: np.ndarray) -> tuple[np.ndarray, float] | None:
    """그림 둘레가 한 가지 색이면 (그 색, 둘레의 흩어진 정도). 아니면 None."""

    b = BG_BORDER
    border = np.concatenate([
        img[:b].reshape(-1, 3), img[-b:].reshape(-1, 3),
        img[:, :b].reshape(-1, 3), img[:, -b:].reshape(-1, 3),
    ]).astype(np.float32)
    color = np.median(border, axis=0)
    spread = float(np.percentile(np.linalg.norm(border - color, axis=1), 90))
    return (color, spread) if spread <= BG_UNIFORM_MAX else None


def color_distance_mask(img: np.ndarray, color: np.ndarray, spread: float, joints: list | None = None) -> np.ndarray | None:
    """바탕색과 다른 곳을 캐릭터로 본다. 가장 큰 덩어리와 관절이 놓인 덩어리, 그 안의 구멍까지.

    create_mask 는 색의 가장 어두운 성분(np.min)으로 선을 찾아 흰 종이를 전제로 한다.
    빨강 · 노랑 바탕은 그 기준에서 어두워 바탕이 캐릭터 둘레에 두껍게 붙어 나왔다 (2026-11-03).

    연한 크레용으로 성글게 칠한 팔다리 · 얼굴은 바탕과 가까운 픽셀이 섞여 몸통과 끊어진 덩어리가 된다.
    가장 큰 덩어리 하나만 남기면 그 팔이 통째로 빠졌다 (2026-11-03 저녁, 살구색 팔 · 얼굴).
    그래서 덩어리를 셀 때만 틈을 메워 잇고, 관절이 놓인 덩어리 · 큰 덩어리는 함께 남긴다."""

    dist = np.linalg.norm(img.astype(np.float32) - color, axis=2)
    fg = (dist > max(BG_DISTANCE_MIN, 3 * spread)).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    k = max(5, round(max(img.shape[:2]) * BG_JOIN_GAP) | 1)
    joined = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(joined)
    if count < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = {1 + int(np.argmax(areas))} | {1 + i for i in np.flatnonzero(areas >= areas.max() * BG_KEEP_PART)}
    h, w = labels.shape
    for j in joints or []:
        x, y = round(j["x"]), round(j["y"])
        if 0 <= x < w and 0 <= y < h and labels[y, x] > 0:
            keep.add(int(labels[y, x]))
    mask = ndimage.binary_fill_holes(np.isin(labels, list(keep)))
    return (mask * 255).astype(np.uint8)


def is_white_paper(color: np.ndarray) -> bool:
    """바탕이 흰 종이 · 연한 회색인지 (밝고 색이 거의 없다)."""
    return float(color.min()) >= WHITE_PAPER_MIN and float(color.max() - color.min()) <= WHITE_PAPER_TINT


# 덜 칠한 몸 메우기 (fill_body) — 틈을 이을 굵기(캐릭터 대각선 대비)와 뼈대
BODY_GAP = 0.05
BODY_BONES = (
    ("hip", "torso"), ("torso", "neck"),
    ("neck", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
    ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
    ("hip", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_foot"),
    ("hip", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_foot"),
)


def fill_body(mask: np.ndarray, joints: list, bbox: dict, scale: float = 1.0) -> np.ndarray:
    """칠하지 않았거나 덜 칠한 몸 안쪽이 배경으로 빠진 것을 메운다 (2026-11-03 사용자 「채색을 덜 한 부분이 배경처럼」).

    몸을 칠하지 않은 그림은 윤곽선이 한 곳만 끊겨도 바깥 종이가 그 틈으로 이어져 몸통 안쪽이 통째로 배경이 됐다.
    마스크를 크게 메워(틈 잇기 + 구멍 채우기) 새로 생긴 덩어리 가운데 **관절 뼈대가 지나가는 것만** 더한다 —
    팔과 몸 사이 · 다리 사이처럼 뼈대가 지나지 않는 빈 곳은 그대로 배경이다. scale 은 joints · bbox 를 mask 크기로 맞추는 배율."""

    m = mask > 127
    if not m.any():
        return mask
    diag = float(np.hypot((bbox["right"] - bbox["left"]) * scale, (bbox["bottom"] - bbox["top"]) * scale))
    k = max(5, round(diag * BODY_GAP) | 1)
    closed = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    added = ndimage.binary_fill_holes(closed) & ~m
    if not added.any():
        return mask
    labels, n = ndimage.label(added)
    at = {j["name"]: (j["x"] * scale, j["y"] * scale) for j in joints}
    h, w = m.shape
    keep = set()
    for a, b in BODY_BONES:
        if a not in at or b not in at:
            continue
        for t in np.linspace(0.0, 1.0, 24):
            x = round(at[a][0] + (at[b][0] - at[a][0]) * t)
            y = round(at[a][1] + (at[b][1] - at[a][1]) * t)
            if 0 <= x < w and 0 <= y < h and labels[y, x]:
                keep.add(int(labels[y, x]))
    if not keep:
        return mask
    out = m | np.isin(labels, list(keep))
    return (out * 255).astype(np.uint8)


# 팔다리에 갇힌 종이 빼기 (drop_pockets) — 캐릭터 대각선 대비 비율
POCKET_MIN_AREA = 0.01     # 마스크 넓이 대비 이보다 작은 흰 곳(눈 반짝임 · 옷 무늬)은 그대로
POCKET_TOUCH = 0.02        # 팔다리 뼈대가 이 거리 안에 닿으면 팔다리에 둘러싸인 곳
POCKET_OUTSIDE = 0.015     # 이 거리 안에 캐릭터 바깥이 있으면 가는 선 하나 너머가 바깥
POCKET_DEEP = 0.5          # 뼈대 · 머리 중심이 덩어리 가장 깊은 곳의 이 비율 이상 깊이 지나면 몸
LIMB_BONES = (
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
    ("right_hip", "right_knee"), ("right_knee", "right_foot"),
    ("left_hip", "left_knee"), ("left_knee", "left_foot"),
)


def drop_pockets(img: np.ndarray, mask: np.ndarray, joints: list, bbox: dict, scale: float = 1.0) -> np.ndarray:
    """팔 · 머리카락 · 다리 선에 둘러싸인 종이가 몸으로 잡힌 것을 뺀다 (2026-11-03 사용자 「팔로 감싸진 배경 부분이 신체로」).

    긴 머리카락 곡선에 막대 팔이 닿으면 그 사이 종이가 선으로 닫혀, 구멍 채우기(create_mask · fill_body)가 몸으로 메웠다.
    칠하지 않은 얼굴 · 몸통도 똑같이 선에 닫힌 종이라 색으로는 못 가른다. 그래서 뼈대가 지나는 자리로 가른다 —
    몸통 · 팔다리 뼈대나 머리 중심(목에서 위로 늘인 곳)이 덩어리 한가운데를 지나면 몸이고,
    팔다리 뼈대가 가장자리에만 닿고 가는 선 너머가 바로 바깥인 넓은 종이는 틈이다. scale 은 fill_body 와 같다."""

    m = mask > 127
    if not m.any() or not joints:
        return mask
    at = {j["name"]: (j["x"] * scale, j["y"] * scale) for j in joints}
    diag = float(np.hypot((bbox["right"] - bbox["left"]) * scale, (bbox["bottom"] - bbox["top"]) * scale))
    # 흰 종이만 — 색 바탕은 「종이색」이 바탕색이라 같은 색으로 칠한 손이 틈으로 빠졌다(빨강 바탕 시험).
    # 그림 둘레 전체가 아니라 캐릭터 바로 바깥 띠로 잰다 — 책상이 함께 찍힌 사진도 흰 종이로 보게
    k = max(5, round(diag * 0.03) | 1)
    ring = (cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0) & ~m
    if not ring.any() or not is_white_paper(np.median(img[ring], axis=0)):
        return mask
    cand = (m & (paper_distance(img, mask) < PAPER_NEAR)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
    big = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= m.sum() * POCKET_MIN_AREA]
    if not big:
        return mask
    depth = cv2.distanceTransform(cand, cv2.DIST_L2, 5)
    h, w = m.shape

    def samples(pairs, ts):
        for a, b in pairs:
            if a in at and b in at:
                for t in ts:
                    x = at[a][0] + (at[b][0] - at[a][0]) * t
                    y = at[a][1] + (at[b][1] - at[a][1]) * t
                    if 0 <= round(x) < w and 0 <= round(y) < h:
                        yield round(x), round(y)

    # 목 → 어깨는 뺀다 — 머리가 큰 그림은 목 관절이 얼굴 한가운데 잡혀, 이 뼈대가 얼굴 아래 · 팔 · 옷 사이
    # 빈 종이를 비스듬히 가로질러 틈을 몸으로 남겼다 (2026-11-04 사용자 「오른쪽 팔부분에서 배경을 캐릭터로 여전히」).
    # 칠하지 않은 몸통은 엉덩이 → 몸통 → 목 뼈대가 지킨다
    deep_bones = [bone for bone in BODY_BONES if not (bone[0] == "neck" and bone[1].endswith("_shoulder"))]
    deep_pts = list(samples(deep_bones, np.linspace(0, 1, 24)))
    deep_pts += list(samples([("torso", "neck")], np.linspace(1.3, 2.6, 14)))   # 머리 중심 쪽
    # 손 · 발 관절은 손목 · 발목 쪽에 놓여 칠하지 않은 손바닥 · 발 안쪽을 못 지난다 — 끝 너머로 조금 늘린다
    ends = [("right_elbow", "right_hand"), ("left_elbow", "left_hand"), ("right_knee", "right_foot"), ("left_knee", "left_foot")]
    deep_pts += list(samples(ends, np.linspace(1.0, 1.6, 10)))
    limb_pts = list(samples(LIMB_BONES, np.linspace(0, 1, 24)))
    touch = max(3, round(diag * POCKET_TOUCH))
    near_out = max(3, round(diag * POCKET_OUTSIDE))
    outside = ~m
    out = m.copy()
    for i in big:
        comp = labels == i
        r = float(depth[comp].max())
        if any(labels[y, x] == i and depth[y, x] >= r * POCKET_DEEP for x, y in deep_pts):
            continue
        x0, y0, bw, bh = stats[i, :4]
        pad = touch + near_out + 2
        sl = (slice(max(0, y0 - pad), min(h, y0 + bh + pad)), slice(max(0, x0 - pad), min(w, x0 + bw + pad)))
        local = comp[sl].astype(np.uint8)
        grown = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch + 1, 2 * touch + 1))) > 0
        oy, ox = sl[0].start, sl[1].start
        if not any(0 <= y - oy < grown.shape[0] and 0 <= x - ox < grown.shape[1] and grown[y - oy, x - ox] for x, y in limb_pts):
            continue
        ring = cv2.dilate(local, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * near_out + 1, 2 * near_out + 1))) > 0
        if not (ring & outside[sl]).any():
            continue
        out[comp] = False
    if out.sum() < m.sum() * PAPER_KEEP_MIN:
        return mask
    return (out * 255).astype(np.uint8)


def build_mask(img: np.ndarray, bbox: dict, joints: list) -> np.ndarray:
    """흰 종이는 Meta 원본 방식(선 찾기, create_mask)이 먼저, 색 바탕은 바탕색 방식이 먼저.

    바탕색 방식은 빨강 · 노랑 바탕이 선처럼 잡혀 둘레에 붙던 것 때문에 넣었다(1bd5d7a). 흰 종이에도 먼저 쓰니
    연한 살구색으로 칠한 얼굴 · 팔이 바탕과 가까워 통째로 빠졌다 (2026-11-03 저녁 — 「이렇게 움직여요」에서
    캐릭터가 반 잘림). 흰 종이 둘레에 붙는 종이색은 refine_mask 가 깎는다.
    어느 쪽이든 관절이 캐릭터 위에 덜 놓이면 다른 방식과 견줘 더 많이 덮는 쪽을 쓴다."""

    background = uniform_background(img)
    color_first = background is not None and not is_white_paper(background[0])

    def by_color():
        mask = color_distance_mask(img, *background, joints) if background is not None else None
        return mask, (joints_on_mask_ratio(mask, joints, 0, 0) if mask is not None else 0.0)

    def by_lines():
        try:
            mask = _build_mask_by_lines(img, bbox, joints)
        except RuntimeError:
            return None, 0.0
        return mask, joints_on_mask_ratio(mask, joints, 0, 0)

    first, second = (by_color, by_lines) if color_first else (by_lines, by_color)
    mask, ratio = first()
    if mask is not None and ratio >= 1.0:
        return mask
    other, other_ratio = second()
    if other is not None and (mask is None or other_ratio > ratio):
        return other
    if mask is None:
        raise RuntimeError("NO_CHARACTER_DETECTED: Mask contour not found")
    return mask


def _build_mask_by_lines(img: np.ndarray, bbox: dict, joints: list) -> np.ndarray:
    """bbox 영역의 캐릭터 마스크를 만든다. 선이 끊기거나 잘린 그림도 처리한다.

    create_mask는 이미지 가장자리에서 배경을 채워 나가는 방식이라,
    선이 끊겨 있거나 bbox 경계에서 잘리면 배경이 캐릭터 안으로 새어 안팎이 뒤집힌다.

    대응: (1) bbox보다 넓게 잘라 선이 경계에서 끊기지 않게 하고
          (2) 틈 메우기 강도를 올려가며 관절이 캐릭터 위에 놓일 때까지 재시도한다.
    """

    left, top = round(bbox["left"]), round(bbox["top"])
    right, bottom = round(bbox["right"]), round(bbox["bottom"])

    # bbox 바깥까지 포함해 자른다. 원본 범위를 넘는 만큼은 흰 여백으로 채운다.
    margin = MASK_CROP_MARGIN
    wide = img[
        max(0, top - margin):min(img.shape[0], bottom + margin),
        max(0, left - margin):min(img.shape[1], right + margin),
    ]

    # 잘라낸 영역이 원본 경계에 걸려 여백이 부족하면 가장자리 색으로 채워 준다
    pad_left = margin - (left - max(0, left - margin))
    pad_top = margin - (top - max(0, top - margin))
    pad_right = margin - (min(img.shape[1], right + margin) - right)
    pad_bottom = margin - (min(img.shape[0], bottom + margin) - bottom)

    pad = MASK_PADDING
    padded = cv2.copyMakeBorder(
        wide,
        pad + pad_top, pad + pad_bottom, pad + pad_left, pad + pad_right,
        # 흰색으로 채우면 회색 종이와의 경계가 선으로 읽혀 그림 전체가 캐릭터가 된다
        cv2.BORDER_REPLICATE
    )

    def to_full_image(mask: np.ndarray) -> np.ndarray:
        """padded 기준 마스크를 원본 이미지 크기의 제자리에 놓는다."""

        full = np.zeros(img.shape[:2], dtype=np.uint8)

        # padded에서 잘라낼 영역(원본과 겹치는 부분)
        src_x = pad + pad_left
        src_y = pad + pad_top
        dst_x = max(0, left - margin)
        dst_y = max(0, top - margin)
        w = wide.shape[1]
        h = wide.shape[0]

        full[dst_y:dst_y + h, dst_x:dst_x + w] = mask[src_y:src_y + h, src_x:src_x + w]
        return full

    best_mask = None
    best_ratio = -1.0

    for iterations in MASK_CLOSE_ATTEMPTS:
        try:
            mask = to_full_image(create_mask(padded, close_iterations=iterations))
        except RuntimeError:
            continue

        ratio = joints_on_mask_ratio(mask, joints, 0, 0)

        if ratio > best_ratio:
            best_mask, best_ratio = mask, ratio

        if ratio >= MASK_JOINT_HIT_RATIO:
            return mask

    if best_mask is None:
        raise RuntimeError("NO_CHARACTER_DETECTED: Mask contour not found")

    # 어떤 강도로도 관절이 캐릭터 위에 놓이지 않으면 안팎이 뒤집힌 것으로 보고 되돌린다.
    if best_ratio < MASK_JOINT_HIT_RATIO:
        inverted = cv2.bitwise_not(best_mask)

        if joints_on_mask_ratio(inverted, joints, 0, 0) > best_ratio:
            return inverted

    return best_mask


def paper_distance(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """픽셀마다 종이색과의 거리(RGB). 종이색은 캐릭터 바깥에서만 재고 캐릭터 자리는 바깥 색으로 메운다."""

    h, w = mask.shape
    s = 400 / max(h, w)
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    hole = cv2.resize(mask, small.shape[1::-1], interpolation=cv2.INTER_NEAREST)
    hole = cv2.dilate(hole, np.ones((5, 5), np.uint8))
    paper = cv2.inpaint(cv2.medianBlur(small, 9), hole, 15, cv2.INPAINT_TELEA)
    paper = cv2.resize(paper, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.linalg.norm(cv2.medianBlur(img, 3).astype(np.float32) - paper.astype(np.float32), axis=2)


def refine_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크 둘레에서 종이색인 곳을 깎는다. 원본 크기에서 한다 (2026-11-03).

    선 방식(create_mask)은 선으로 둘러싸인 곳을 통째로 캐릭터로 보고, 줄인 그림에서 만든 마스크를 늘려 쓴다.
    사진으로 찍은 손그림은 팔다리 둘레 · 머리와 머리끈 사이의 종이가 네모 조각으로 붙어 나왔다.

    종이색은 캐릭터 바깥에서만 재고(캐릭터 자리는 바깥 색으로 메움 — 그늘진 종이도 맞는다),
    마스크 테두리에서 이어진 종이색만 지운다. 선에 막힌 안쪽(색칠하지 않은 몸)은 남는다.
    떨어져 나간 팔다리 선은 렌더가 뼈대로 다시 잇는다 (renderer.mesh_mask).
    """

    inside = mask > 127
    dist = paper_distance(img, mask)
    cand = (inside & (dist < PAPER_NEAR)).astype(np.uint8)
    _, labels = cv2.connectedComponents(cand, connectivity=8)
    edge = inside & ~cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    touching = np.unique(labels[edge & (cand > 0)])
    out = (inside & ~np.isin(labels, touching[touching > 0])).astype(np.uint8)

    # 부스러기 정리: 가장 큰 덩어리의 0.2% 보다 작은 조각은 버린다 (팔다리 선 조각은 남는다)
    count, lab, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        out = np.isin(lab, 1 + np.flatnonzero(areas >= areas.max() * 0.002)).astype(np.uint8)

    if out.sum() < inside.sum() * PAPER_KEEP_MIN:
        return mask
    return out * 255


def make_cutout(image_path: str, mask_path: str, out_path: str) -> bool:
    """배경을 투명하게 잘라 낸 캐릭터 PNG (2026-11-03).

    화면이 동작 영상 대신 정지 그림을 보일 때(미리보기를 만드는 중 · 대기 쪽 · 표지) 원본을 쓰면
    사진 배경까지 나왔다. 렌더와 같게 마스크를 1px 안쪽으로 깎는다.
    **원본과 같은 크기로 둔다** — 관절 좌표(원본 기준)를 쓰는 화면이 원본 자리에 그대로 바꿔 끼운다.
    """

    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None or mask.shape[:2] != img.shape[:2]:
        return False
    # 종이색을 이미 깎았으니(refine_mask) 1px 만 — 가는 팔다리 선이 사라지지 않게
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    if not (mask > 127).any():
        return False
    rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    return bool(cv2.imwrite(out_path, rgba))


def to_original_scale(bbox: dict, scale: float) -> dict:
    """분석용으로 줄인 이미지 기준 bbox를 원본 이미지 기준으로 되돌린다."""

    return {
        key: round(value / scale)
        for key, value in bbox.items()
    }


def flatten_transparency(path: Path) -> None:
    """투명 배경 그림을 흰 종이 위에 놓은 그림으로 바꿔 같은 자리에 저장한다.

    cv2.imread 는 투명 칸을 검정으로 읽는다.
    렌더도 이 파일을 다시 읽으므로 분석 전에 파일 자체를 바꾼다.
    """

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return

    if img.dtype == np.uint16:
        img = (img / 257).astype(np.uint8)

    alpha = img[..., 3:] / 255.0
    flat = (img[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
    path.write_bytes(cv2.imencode(".png", flat)[1].tobytes())


def analyze_image(image_path: str, mask_path: str | None = None) -> dict:
    """그림을 분석해 bbox, 관절 15개, 마스크 정보를 반환한다.

    mask_path를 주면 캐릭터 영역 마스크를 원본 크기의 PNG로 저장한다.
    렌더링(/internal/v1/render)에 필요하다.
    """

    started_at = time.perf_counter()

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not check_torchserve():
        return {
            "success": False,
            "bbox": None,
            "joints": [],
            "message": "MODEL_UNAVAILABLE: TorchServe is not running"
        }

    flatten_transparency(path)

    img = cv2.imread(str(path))

    if img is None:
        return {
            "success": False,
            "bbox": None,
            "joints": [],
            "message": "INVALID_IMAGE: Could not read image"
        }

    if len(img.shape) != 3:
        return {
            "success": False,
            "bbox": None,
            "joints": [],
            "message": "INVALID_IMAGE: Image must have 3 channels"
        }

    # Meta 원본 코드와 동일하게 분석은 최대 1000px로 줄인 이미지로 한다.
    # 단, 응답 좌표는 scale로 다시 나눠 원본 이미지 기준으로 되돌린다.
    scale = 1.0
    original = img

    if np.max(img.shape) > MAX_ANALYSIS_SIZE:
        scale = MAX_ANALYSIS_SIZE / np.max(img.shape)

        img = cv2.resize(
            img,
            (
                round(scale * img.shape[1]),
                round(scale * img.shape[0])
            )
        )

    # 1. 캐릭터 검출
    bbox = detect_character(img)
    confidence = round(bbox["score"], 3)

    # 이미지 범위를 벗어나는 bbox 방지
    left = max(0, bbox["left"])
    top = max(0, bbox["top"])
    right = min(img.shape[1], bbox["right"])
    bottom = min(img.shape[0], bbox["bottom"])

    if right <= left or bottom <= top:
        return {
            "success": False,
            "bbox": None,
            "joints": [],
            "message": "INVALID_BBOX: Character bounding box is invalid"
        }

    bbox = {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }

    # 2. 캐릭터 영역 crop
    cropped = img[
        top:bottom,
        left:right
    ]

    # 3. Pose Estimator
    joints = estimate_pose(
        cropped,
        bbox
    )

    scores = [joint["score"] for joint in joints if joint["score"] is not None]

    if scores and sum(scores) / len(scores) < LOW_CONFIDENCE_JOINT_MEAN:
        return {
            "success": False,
            "bbox": None,
            "joints": [],
            "message": (
                f"LOW_CONFIDENCE: Mean joint score {sum(scores) / len(scores):.2f}"
            ),
        }

    mask = build_mask(img, bbox, joints)

    # 마스크를 원본 이미지 크기로 되돌려 저장한다 (렌더링에서 사용)
    mask_saved = False

    if mask_path is not None:
        # 마스크는 분석용으로 줄인 이미지 기준이므로 원본 크기로 되돌려 저장한다.
        # 캐릭터 영역(bbox)이 아니라 이미지 전체 크기다.
        mask_original = cv2.resize(
            mask,
            # 원본 크기 그대로 — 나눗셈 반올림이면 1px 어긋나(3,000px 사진) 렌더가 MASK_SIZE_MISMATCH 로 실패했다 (2026-11-03)
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        if mask_original.shape[:2] == original.shape[:2]:
            mask_original = refine_mask(original, mask_original)
        # 종이색 깎기 다음에 — 깎기가 틈으로 이어진 칠하지 않은 몸 안쪽을 또 지우므로 마지막에 메운다
        mask_original = fill_body(mask_original, joints, bbox, 1 / scale)
        # 메운 다음에 — 팔 · 머리카락에 갇힌 종이는 위 구멍 채우기들이 몸으로 메운 것까지 함께 뺀다
        if mask_original.shape[:2] == original.shape[:2]:
            mask_original = drop_pockets(original, mask_original, joints, bbox, 1 / scale)

        Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
        mask_saved = bool(cv2.imwrite(mask_path, mask_original))

    # 4. 모든 좌표를 원본 이미지 기준으로 복원
    bbox = to_original_scale(bbox, scale)

    joints = [
        {
            "name": joint["name"],
            "x": round(joint["x"] / scale),
            "y": round(joint["y"] / scale),
            "score": joint["score"],
        }
        for joint in joints
    ]

    # 5. 최종 분석 결과
    return {
        "success": True,
        "bbox": bbox,
        "joints": joints,
        "confidence": confidence,
        "mask": {
            "width": round(mask.shape[1] / scale),
            "height": round(mask.shape[0] / scale),
            "saved": mask_saved,
            "covers": "full_image",
        },
        "message": "Analysis complete",
        "model_version": MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "coordinate_space": COORDINATE_SPACE,
        "processing_time_ms": round((time.perf_counter() - started_at) * 1000),
    }