from pathlib import Path
import json

import cv2
import numpy as np
import requests

from skimage import measure
from scipy import ndimage

from ai_server.pose_adapter import run_pose_adapter
from ai_server.runtime_config import TORCHSERVE_URL


# ============================================================
# Character / Skeleton Types
# ============================================================

HUMAN = "human"
QUADRUPED = "quadruped"
NONHUMAN = "nonhuman"

HUMAN_SKELETON = "humanoid_15"
QUADRUPED_SKELETON = "quadruped_16"
NONHUMAN_SKELETON = "nonhuman_generic"


# analyzer.py
# D:\capston\A1-AI-Runtime\ai_server\analyzer.py
#
# parents[1]
# -> D:\capston\A1-AI-Runtime
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SKELETON_ROOT = (
    PROJECT_ROOT
    / "configs"
    / "skeletons"
)


# ============================================================
# TorchServe Health
# ============================================================

def check_torchserve() -> bool:
    try:
        response = requests.get(
            f"{TORCHSERVE_URL}/ping",
            timeout=3,
        )

        return response.status_code == 200

    except requests.RequestException:
        return False


# ============================================================
# Human Character Detection
# ============================================================

def detect_character(
    img: np.ndarray,
) -> dict:
    """
    Meta Animated Drawings detector 호출.
    가장 높은 score의 bbox를 반환한다.
    """

    img_bytes = cv2.imencode(
        ".png",
        img,
    )[1].tobytes()

    try:
        response = requests.post(
            (
                f"{TORCHSERVE_URL}"
                "/predictions/"
                "drawn_humanoid_detector"
            ),
            files={
                "data": img_bytes
            },
            verify=False,
            timeout=30,
        )

    except requests.Timeout as e:
        raise RuntimeError(
            "MODEL_TIMEOUT: "
            "Detector request timed out"
        ) from e

    except requests.RequestException as e:
        raise RuntimeError(
            "MODEL_UNAVAILABLE: "
            f"Detector request failed: {e}"
        ) from e

    if response.status_code >= 300:
        raise RuntimeError(
            "MODEL_UNAVAILABLE: "
            "Detector request failed: "
            f"HTTP {response.status_code}"
        )

    detection_results = json.loads(
        response.content
    )

    if (
        isinstance(
            detection_results,
            dict,
        )
        and detection_results.get(
            "code"
        ) == 404
    ):
        raise RuntimeError(
            "MODEL_UNAVAILABLE: "
            "drawn_humanoid_detector "
            "model is not available"
        )

    if not detection_results:
        raise RuntimeError(
            "NO_CHARACTER_DETECTED: "
            "No drawn humanoid detected"
        )

    detection_results.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    bbox = detection_results[0][
        "bbox"
    ]

    left, top, right, bottom = [
        round(value)
        for value in bbox
    ]

    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "score": (
            detection_results[0][
                "score"
            ]
        ),
    }


# ============================================================
# Character Mask
# ============================================================

def create_mask(
    img: np.ndarray,
) -> np.ndarray:
    """
    Meta Animated Drawings 방식으로
    binary mask를 생성한다.

    0   = background
    255 = character
    """

    # RGB -> grayscale-like
    img = np.min(
        img,
        axis=2,
    )

    # Adaptive Threshold
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        115,
        8,
    )

    img = cv2.bitwise_not(
        img
    )

    # Morphology
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    img = cv2.morphologyEx(
        img,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    img = cv2.morphologyEx(
        img,
        cv2.MORPH_DILATE,
        kernel,
        iterations=2,
    )

    # Flood Fill mask
    flood_mask = np.zeros(
        [
            img.shape[0] + 2,
            img.shape[1] + 2,
        ],
        np.uint8,
    )

    flood_mask[
        1:-1,
        1:-1,
    ] = img.copy()

    im_floodfill = np.full(
        img.shape,
        255,
        np.uint8,
    )

    h, w = img.shape[:2]

    for x in range(
        0,
        w - 1,
        10,
    ):
        cv2.floodFill(
            im_floodfill,
            flood_mask,
            (x, 0),
            0,
        )

        cv2.floodFill(
            im_floodfill,
            flood_mask,
            (x, h - 1),
            0,
        )

    for y in range(
        0,
        h - 1,
        10,
    ):
        cv2.floodFill(
            im_floodfill,
            flood_mask,
            (0, y),
            0,
        )

        cv2.floodFill(
            im_floodfill,
            flood_mask,
            (w - 1, y),
            0,
        )

    im_floodfill[0, :] = 0
    im_floodfill[-1, :] = 0
    im_floodfill[:, 0] = 0
    im_floodfill[:, -1] = 0

    mask2 = cv2.bitwise_not(
        im_floodfill
    )

    mask = None
    biggest = 0

    contours = measure.find_contours(
        mask2,
        0.0,
    )

    for contour in contours:
        x = np.zeros(
            mask2.T.shape,
            np.uint8,
        )

        cv2.fillPoly(
            x,
            [
                np.int32(
                    contour
                )
            ],
            1,
        )

        size = len(
            np.where(
                x == 1
            )[0]
        )

        if size > biggest:
            mask = x
            biggest = size

    if mask is None:
        raise RuntimeError(
            "NO_CHARACTER_DETECTED: "
            "Mask contour not found"
        )

    mask = (
        ndimage
        .binary_fill_holes(
            mask
        )
        .astype(int)
    )

    mask = (
        255
        * mask.astype(
            np.uint8
        )
    )

    return mask.T


# ============================================================
# Human Pose Estimator
# ============================================================

def estimate_pose(
    cropped: np.ndarray,
    bbox: dict,
) -> list:
    """
    Meta 17 keypoints
    -> DrawTale humanoid 15 joints.
    """

    img_bytes = cv2.imencode(
        ".png",
        cropped,
    )[1].tobytes()

    try:
        response = requests.post(
            (
                f"{TORCHSERVE_URL}"
                "/predictions/"
                "drawn_humanoid_pose_estimator"
            ),
            files={
                "data": img_bytes
            },
            verify=False,
            timeout=60,
        )

    except requests.Timeout as e:
        raise RuntimeError(
            "MODEL_TIMEOUT: "
            "Pose estimator "
            "request timed out"
        ) from e

    except requests.RequestException as e:
        raise RuntimeError(
            "MODEL_UNAVAILABLE: "
            "Pose estimator "
            f"request failed: {e}"
        ) from e

    if response.status_code >= 300:
        raise RuntimeError(
            "MODEL_UNAVAILABLE: "
            "Pose estimator "
            "request failed: "
            f"HTTP {response.status_code}"
        )

    pose_results = json.loads(
        response.content
    )

    if (
        isinstance(
            pose_results,
            dict,
        )
        and pose_results.get(
            "code"
        ) == 404
    ):
        raise RuntimeError(
            "MODEL_UNAVAILABLE: "
            "drawn_humanoid_pose_estimator "
            "model is not available"
        )

    if not pose_results:
        raise RuntimeError(
            "NO_CHARACTER_DETECTED: "
            "Pose estimation "
            "returned no result"
        )

    if len(pose_results) != 1:
        raise RuntimeError(
            "Pose estimation returned "
            "an unexpected number "
            "of skeletons"
        )

    keypoints = (
        pose_results[0]
        .get(
            "keypoints",
            [],
        )
    )

    if len(keypoints) < 17:
        raise RuntimeError(
            "Pose estimation returned "
            f"only {len(keypoints)} "
            "keypoints"
        )

    kpts = np.array(
        keypoints
    )

    def point(
        index: int,
    ) -> tuple:
        x = round(
            float(
                kpts[index][0]
            )
            + bbox["left"]
        )

        y = round(
            float(
                kpts[index][1]
            )
            + bbox["top"]
        )

        return x, y

    def midpoint(
        index1: int,
        index2: int,
    ) -> tuple:
        x = round(
            (
                float(
                    kpts[index1][0]
                )
                + float(
                    kpts[index2][0]
                )
            )
            / 2
            + bbox["left"]
        )

        y = round(
            (
                float(
                    kpts[index1][1]
                )
                + float(
                    kpts[index2][1]
                )
            )
            / 2
            + bbox["top"]
        )

        return x, y

    hip = midpoint(
        11,
        12,
    )

    torso = midpoint(
        5,
        6,
    )

    neck = point(0)

    right_shoulder = point(6)
    right_elbow = point(8)
    right_hand = point(10)

    left_shoulder = point(5)
    left_elbow = point(7)
    left_hand = point(9)

    right_hip = point(12)
    right_knee = point(14)
    right_foot = point(16)

    left_hip = point(11)
    left_knee = point(13)
    left_foot = point(15)

    joints = [
        {
            "name": "hip",
            "x": hip[0],
            "y": hip[1],
        },
        {
            "name": "torso",
            "x": torso[0],
            "y": torso[1],
        },
        {
            "name": "neck",
            "x": neck[0],
            "y": neck[1],
        },

        {
            "name": "right_shoulder",
            "x": right_shoulder[0],
            "y": right_shoulder[1],
        },
        {
            "name": "right_elbow",
            "x": right_elbow[0],
            "y": right_elbow[1],
        },
        {
            "name": "right_hand",
            "x": right_hand[0],
            "y": right_hand[1],
        },

        {
            "name": "left_shoulder",
            "x": left_shoulder[0],
            "y": left_shoulder[1],
        },
        {
            "name": "left_elbow",
            "x": left_elbow[0],
            "y": left_elbow[1],
        },
        {
            "name": "left_hand",
            "x": left_hand[0],
            "y": left_hand[1],
        },

        {
            "name": "right_hip",
            "x": right_hip[0],
            "y": right_hip[1],
        },
        {
            "name": "right_knee",
            "x": right_knee[0],
            "y": right_knee[1],
        },
        {
            "name": "right_foot",
            "x": right_foot[0],
            "y": right_foot[1],
        },

        {
            "name": "left_hip",
            "x": left_hip[0],
            "y": left_hip[1],
        },
        {
            "name": "left_knee",
            "x": left_knee[0],
            "y": left_knee[1],
        },
        {
            "name": "left_foot",
            "x": left_foot[0],
            "y": left_foot[1],
        },
    ]

    return joints


# ============================================================
# Human Analyze Pipeline
# ============================================================

def analyze_human(
    image_path: str,
) -> dict:
    path = Path(
        image_path
    )

    if not path.exists():
        raise FileNotFoundError(
            "Image not found: "
            f"{image_path}"
        )

    if not check_torchserve():
        return {
            "success": False,
            "character_type": HUMAN,
            "skeleton_type": (
                HUMAN_SKELETON
            ),
            "bbox": None,
            "joints": [],
            "mask": None,
            "message": (
                "MODEL_UNAVAILABLE: "
                "TorchServe is not running"
            ),
        }

    img = cv2.imread(
        str(path)
    )

    if img is None:
        return {
            "success": False,
            "character_type": HUMAN,
            "skeleton_type": (
                HUMAN_SKELETON
            ),
            "bbox": None,
            "joints": [],
            "mask": None,
            "message": (
                "INVALID_IMAGE: "
                "Could not read image"
            ),
        }

    if len(img.shape) != 3:
        return {
            "success": False,
            "character_type": HUMAN,
            "skeleton_type": (
                HUMAN_SKELETON
            ),
            "bbox": None,
            "joints": [],
            "mask": None,
            "message": (
                "INVALID_IMAGE: "
                "Image must have "
                "3 channels"
            ),
        }

    # Meta 원본처럼
    # 최대 dimension 1000px
    if np.max(
        img.shape
    ) > 1000:
        scale = (
            1000
            / np.max(
                img.shape
            )
        )

        img = cv2.resize(
            img,
            (
                round(
                    scale
                    * img.shape[1]
                ),
                round(
                    scale
                    * img.shape[0]
                ),
            ),
        )

    bbox = detect_character(
        img
    )

    left = max(
        0,
        bbox["left"],
    )

    top = max(
        0,
        bbox["top"],
    )

    right = min(
        img.shape[1],
        bbox["right"],
    )

    bottom = min(
        img.shape[0],
        bbox["bottom"],
    )

    if (
        right <= left
        or bottom <= top
    ):
        return {
            "success": False,
            "character_type": HUMAN,
            "skeleton_type": (
                HUMAN_SKELETON
            ),
            "bbox": None,
            "joints": [],
            "mask": None,
            "message": (
                "INVALID_BBOX: "
                "Character bounding "
                "box is invalid"
            ),
        }

    bbox = {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
    }

    cropped = img[
        top:bottom,
        left:right,
    ]

    joints = run_pose_adapter(
        cropped=cropped,
        bbox=bbox,
        meta_estimator=(
            estimate_pose
        ),
    )

    mask = create_mask(
        cropped
    )

    return {
        "success": True,
        "character_type": HUMAN,
        "skeleton_type": (
            HUMAN_SKELETON
        ),
        "bbox": bbox,
        "joints": joints,
        "mask": {
            "width": int(
                mask.shape[1]
            ),
            "height": int(
                mask.shape[0]
            ),
        },
        "message": (
            "Analysis complete"
        ),
    }


# ============================================================
# Skeleton Config
# ============================================================

def load_skeleton_config(
    skeleton_type: str,
) -> dict:
    if skeleton_type == QUADRUPED_SKELETON:
        config_path = (
            SKELETON_ROOT
            / "quadruped"
            / "quadruped_16.json"
        )

    elif skeleton_type == "nonhuman_starfish_6":
        config_path = (
            SKELETON_ROOT
            / "nonhuman"
            / "nonhuman_starfish_6.json"
        )

    else:
        raise RuntimeError(
            "UNKNOWN_SKELETON_TYPE: "
            f"{skeleton_type}"
        )

    if not config_path.exists():
        raise FileNotFoundError(
            "Skeleton config not found: "
            f"{config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)

# ============================================================
# Skeleton Joint Validation
# ============================================================

def validate_joints_for_skeleton(
    joints: list,
    skeleton_type: str,
) -> bool:
    config = load_skeleton_config(
        skeleton_type
    )

    expected_names = {
        joint["name"]
        for joint
        in config["joints"]
    }

    received_names = {
        joint["name"]
        for joint
        in joints
    }

    if (
        len(joints)
        != config["joint_count"]
    ):
        raise ValueError(
            "INVALID_JOINT_COUNT: "
            f"expected "
            f"{config['joint_count']}, "
            f"got {len(joints)}"
        )

    missing = (
        expected_names
        - received_names
    )

    extra = (
        received_names
        - expected_names
    )

    if missing:
        raise ValueError(
            "MISSING_JOINTS: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if extra:
        raise ValueError(
            "UNKNOWN_JOINTS: "
            + ", ".join(
                sorted(
                    extra
                )
            )
        )

    return True


# ============================================================
# Skeleton Edge Builder
# ============================================================

def build_skeleton_edges(
    joints: list,
    skeleton_type: str,
) -> list:
    validate_joints_for_skeleton(
        joints,
        skeleton_type,
    )

    config = load_skeleton_config(
        skeleton_type
    )

    joint_map = {
        joint["name"]: joint
        for joint
        in joints
    }

    edges = []

    for joint_cfg in config[
        "joints"
    ]:
        child_name = (
            joint_cfg["name"]
        )

        parent_name = (
            joint_cfg["parent"]
        )

        if parent_name is None:
            continue

        child = joint_map[
            child_name
        ]

        parent = joint_map[
            parent_name
        ]

        edges.append(
            {
                "parent": (
                    parent_name
                ),
                "child": (
                    child_name
                ),
                "parent_x": (
                    parent["x"]
                ),
                "parent_y": (
                    parent["y"]
                ),
                "child_x": (
                    child["x"]
                ),
                "child_y": (
                    child["y"]
                ),
            }
        )

    return edges


# ============================================================
# Quadruped Analyze Pipeline
# ============================================================

def analyze_quadruped(
    image_path: str,
) -> dict:
    """
    현재 단계에서는 A2의 4족 Pose 모델이
    아직 연결되지 않았기 때문에
    quadruped_manual_joints.json을 이용한다.

    이후 실제 quadruped pose estimator가
    연결되면 이 함수 내부만 교체하면 된다.
    """

    image = Path(
        image_path
    )

    if not image.exists():
        raise FileNotFoundError(
            "Image not found: "
            f"{image_path}"
        )

    img = cv2.imread(
        str(image)
    )

    if img is None:
        return {
            "success": False,
            "character_type": (
                QUADRUPED
            ),
            "skeleton_type": (
                QUADRUPED_SKELETON
            ),
            "bbox": None,
            "joints": [],
            "mask": None,
            "message": (
                "INVALID_IMAGE: "
                "Could not read image"
            ),
        }

    manual_joints_path = (
        SKELETON_ROOT
        / "quadruped"
        / "quadruped_manual_joints.json"
    )

    if not manual_joints_path.exists():
        return {
            "success": False,
            "character_type": (
                QUADRUPED
            ),
            "skeleton_type": (
                QUADRUPED_SKELETON
            ),
            "bbox": None,
            "joints": [],
            "mask": None,
            "message": (
                "MODEL_UNAVAILABLE: "
                "quadruped manual "
                "joints file not found"
            ),
        }

    with open(
        manual_joints_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    joints = data.get(
        "joints",
        [],
    )

    validate_joints_for_skeleton(
        joints,
        QUADRUPED_SKELETON,
    )

    return {
        "success": True,
        "character_type": (
            QUADRUPED
        ),
        "skeleton_type": (
            QUADRUPED_SKELETON
        ),
        "bbox": None,
        "joints": joints,
        "mask": None,
        "message": (
            "Quadruped analysis complete "
            "(manual joints test mode)"
        ),
    }


# ============================================================
# Nonhuman Analyze Pipeline
# ============================================================

def analyze_nonhuman(
    image_path: str,
) -> dict:
    path = Path(
        image_path
    )

    if not path.exists():
        raise FileNotFoundError(
            "Image not found: "
            f"{image_path}"
        )

    img = cv2.imread(
        str(path)
    )

    if img is None:
        return {
            "success": False,
            "character_type": NONHUMAN,
            "skeleton_type": "nonhuman_starfish_6",
            "bbox": None,
            "joints": [],
            "mask": None,
            "message": (
                "INVALID_IMAGE: "
                "Could not read image"
            ),
        }

    manual_joints_path = (
        SKELETON_ROOT
        / "nonhuman"
        / "nonhuman_starfish_manual_joints.json"
    )

    if not manual_joints_path.exists():
        return {
            "success": False,
            "character_type": NONHUMAN,
            "skeleton_type": "nonhuman_starfish_6",
            "bbox": None,
            "joints": [],
            "mask": None,
            "message": (
                "MODEL_UNAVAILABLE: "
                "nonhuman starfish manual "
                "joints file not found"
            ),
        }

    with open(
        manual_joints_path,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    joints = data.get(
        "joints",
        [],
    )

    validate_joints_for_skeleton(
        joints,
        "nonhuman_starfish_6",
    )

    return {
        "success": True,
        "character_type": NONHUMAN,
        "skeleton_type": "nonhuman_starfish_6",
        "bbox": None,
        "joints": joints,
        "mask": None,
        "message": (
            "Nonhuman starfish analysis complete "
            "(manual joints test mode)"
        ),
    }

# ============================================================
# Analyze Router
# ============================================================

def analyze_image(
    image_path: str,
    character_type: str = HUMAN,
) -> dict:
    """
    Frontend/Backend가 전달한 character_type에 따라
    분석 pipeline을 선택한다.

    human
        -> humanoid_15

    quadruped
        -> quadruped_16

    nonhuman
        -> nonhuman_generic
    """

    if character_type == HUMAN:
        return analyze_human(
            image_path
        )

    if character_type == QUADRUPED:
        return analyze_quadruped(
            image_path
        )

    if character_type == NONHUMAN:
        return analyze_nonhuman(
            image_path
        )

    return {
        "success": False,
        "character_type": (
            character_type
        ),
        "skeleton_type": None,
        "bbox": None,
        "joints": [],
        "mask": None,
        "message": (
            "INVALID_CHARACTER_TYPE: "
            "human, quadruped, "
            "or nonhuman"
        ),
    }