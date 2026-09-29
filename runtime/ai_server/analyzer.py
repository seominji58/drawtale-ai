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

    bbox = detection_results[0]["bbox"]

    left, top, right, bottom = [
        round(value) for value in bbox
    ]

    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "score": detection_results[0]["score"],
    }

def create_mask(img: np.ndarray) -> np.ndarray:
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

    img = cv2.morphologyEx(
        img,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
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

    return joints


def to_original_scale(bbox: dict, scale: float) -> dict:
    """분석용으로 줄인 이미지 기준 bbox를 원본 이미지 기준으로 되돌린다."""

    return {
        key: round(value / scale)
        for key, value in bbox.items()
    }


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

    mask = create_mask(cropped)

    # 마스크를 원본 이미지 크기로 되돌려 저장한다 (렌더링에서 사용)
    mask_saved = False

    if mask_path is not None:
        mask_w = round(mask.shape[1] / scale)
        mask_h = round(mask.shape[0] / scale)

        mask_original = cv2.resize(
            mask,
            (mask_w, mask_h),
            interpolation=cv2.INTER_NEAREST
        )

        Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
        mask_saved = bool(cv2.imwrite(mask_path, mask_original))

    # 4. 모든 좌표를 원본 이미지 기준으로 복원
    bbox = to_original_scale(bbox, scale)

    joints = [
        {
            "name": joint["name"],
            "x": round(joint["x"] / scale),
            "y": round(joint["y"] / scale),
        }
        for joint in joints
    ]

    # 5. 최종 분석 결과
    return {
        "success": True,
        "bbox": bbox,
        "joints": joints,
        "mask": {
            "width": round(mask.shape[1] / scale),
            "height": round(mask.shape[0] / scale),
            "saved": mask_saved,
        },
        "message": "Analysis complete",
        "model_version": MODEL_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "coordinate_space": COORDINATE_SPACE,
        "processing_time_ms": round((time.perf_counter() - started_at) * 1000),
    }