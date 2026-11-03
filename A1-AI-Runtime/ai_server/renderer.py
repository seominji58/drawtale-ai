from pathlib import Path
import shutil
import os

import cv2
import yaml

from ai_server.analyzer import create_mask


# ?꾨줈?앺듃 寃쎈줈
BASE_DIR = Path(__file__).resolve().parents[2]

# ?뚮뜑留?寃곌낵 ????대뜑
OUTPUT_ROOT = BASE_DIR / "render_results"


def create_animation_annotation(result: dict, source_image: str) -> str:
    """
    /analyze 寃곌낵瑜?AnimatedDrawings annotation ?뺤떇?쇰줈 蹂?섑븳??
    """

    # 遺꾩꽍 ?ㅽ뙣 ?뺤씤
    if not result.get("success"):
        raise ValueError(
            result.get("message", "Analysis failed")
        )

    bbox = result.get("bbox")
    joints = result.get("joints")

    if bbox is None:
        raise ValueError("bbox is missing")

    if not joints:
        raise ValueError("joints are missing")

    # 愿?덉쓣 ?대쫫?쇰줈 ?쎄쾶 李얠쓣 ???덈룄濡?蹂??
    joint_map = {
        joint["name"]: joint
        for joint in joints
    }

    # AnimatedDrawings?먯꽌 ?꾩슂??愿??
    required_joints = [
        "hip",
        "torso",
        "neck",
        "right_shoulder",
        "right_elbow",
        "right_hand",
        "left_shoulder",
        "left_elbow",
        "left_hand",
        "right_hip",
        "right_knee",
        "right_foot",
        "left_hip",
        "left_knee",
        "left_foot",
    ]

    # ?꾨씫??愿???뺤씤
    missing = [
        name
        for name in required_joints
        if name not in joint_map
    ]

    if missing:
        raise ValueError(
            f"Missing joints: {missing}"
        )

    # --------------------------------
    # 異쒕젰 ?대뜑 ?앹꽦
    # --------------------------------

    OUTPUT_ROOT.mkdir(
        exist_ok=True,
        parents=True
    )

    output_dir = OUTPUT_ROOT / "current"

    output_dir.mkdir(
        exist_ok=True,
        parents=True
    )

    # 湲곗〈 寃곌낵 ??젣
    for path in output_dir.iterdir():

        if path.is_file():
            path.unlink()

        elif path.is_dir():
            shutil.rmtree(path)

    # --------------------------------
    # ?먮낯 ?대?吏 ?쎄린
    # --------------------------------

    image = cv2.imread(source_image)

    if image is None:
        raise ValueError(
            f"Failed to read image: {source_image}"
        )

    # analyzer.py? ?숈씪?섍쾶
    # 理쒕? ?ш린瑜?1000?쇰줈 議곗젙
    max_dim = max(image.shape[:2])

    if max_dim > 1000:

        scale = 1000.0 / max_dim

        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

    # --------------------------------
    # Bounding Box
    # --------------------------------

    left = max(
        0,
        int(bbox["left"])
    )

    top = max(
        0,
        int(bbox["top"])
    )

    right = min(
        image.shape[1],
        int(bbox["right"])
    )

    bottom = min(
        image.shape[0],
        int(bbox["bottom"])
    )

    if right <= left:
        raise ValueError(
            "Invalid bounding box width"
        )

    if bottom <= top:
        raise ValueError(
            "Invalid bounding box height"
        )

    # --------------------------------
    # 罹먮┃???대?吏 Crop
    # --------------------------------

    cropped = image[
        top:bottom,
        left:right
    ]

    if cropped.size == 0:
        raise ValueError(
            "Cropped image is empty"
        )

    crop_height = cropped.shape[0]
    crop_width = cropped.shape[1]

    # --------------------------------
    # Texture ?앹꽦
    # --------------------------------

    texture = cv2.cvtColor(
        cropped,
        cv2.COLOR_BGR2BGRA
    )

    cv2.imwrite(
        str(output_dir / "texture.png"),
        texture
    )

    # --------------------------------
    # Mask ?앹꽦
    # --------------------------------

    mask = create_mask(cropped)

    cv2.imwrite(
        str(output_dir / "mask.png"),
        mask
    )

    # --------------------------------
    # ?먮낯 ?대?吏 ???
    # --------------------------------

    cv2.imwrite(
        str(output_dir / "image.png"),
        image
    )

    # --------------------------------
    # Skeleton ?앹꽦
    # --------------------------------

    skeleton = []

    def local_point(joint):
        """
        analyzer.py?먯꽌 ?살? ?먮낯 ?대?吏 醫뚰몴瑜?
        AnimatedDrawings??crop 湲곗? 醫뚰몴濡?蹂?섑븳??
        """

        x = int(joint["x"]) - left
        y = int(joint["y"]) - top

        # ?대?吏 踰붿쐞 ?덉쑝濡??쒗븳
        x = max(
            0,
            min(x, crop_width - 1)
        )

        y = max(
            0,
            min(y, crop_height - 1)
        )

        return [x, y]

    # --------------------------------
    # Root
    # --------------------------------

    hip = joint_map["hip"]

    skeleton.append({
        "loc": local_point(hip),
        "name": "root",
        "parent": None
    })

    # --------------------------------
    # 遺紐?愿怨?
    # --------------------------------

    parent_map = {

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

    # --------------------------------
    # 愿??異붽?
    # --------------------------------

    for name in required_joints:

        joint = joint_map[name]

        skeleton.append({
            "loc": local_point(joint),
            "name": name,
            "parent": parent_map[name]
        })

    # --------------------------------
    # char_cfg.yaml ?앹꽦
    # --------------------------------

    char_cfg = {
        "height": crop_height,
        "width": crop_width,
        "skeleton": skeleton
    }

    with open(
        output_dir / "char_cfg.yaml",
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            char_cfg,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    # --------------------------------
    # bounding_box.yaml ?앹꽦
    # --------------------------------

    bounding_box = {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom
    }

    with open(
        output_dir / "bounding_box.yaml",
        "w",
        encoding="utf-8"
    ) as f:

        yaml.safe_dump(
            bounding_box,
            f,
            sort_keys=False
        )

    # 寃곌낵 ?대뜑 諛섑솚
    return str(output_dir)
def apply_joint_corrections(annotation_dir: str, joints: list[dict]) -> str:
    """
    ?ъ슜?먭? ?섏젙???먮낯 ?대?吏 湲곗? Joint 醫뚰몴瑜?    AnimatedDrawings??crop-local skeleton 醫뚰몴濡?蹂?섑븯??    char_cfg.yaml??諛섏쁺?쒕떎.
    """

    if not joints:
        raise ValueError("joints are missing")

    required_joints = [
        "hip",
        "torso",
        "neck",
        "right_shoulder",
        "right_elbow",
        "right_hand",
        "left_shoulder",
        "left_elbow",
        "left_hand",
        "right_hip",
        "right_knee",
        "right_foot",
        "left_hip",
        "left_knee",
        "left_foot",
    ]

    joint_map = {
        joint["name"]: joint
        for joint in joints
    }

    missing = [
        name
        for name in required_joints
        if name not in joint_map
    ]

    if missing:
        raise ValueError(
            f"Missing joints: {missing}"
        )

    char_cfg_path = os.path.join(
        annotation_dir,
        "char_cfg.yaml"
    )

    bounding_box_path = os.path.join(
        annotation_dir,
        "bounding_box.yaml"
    )

    if not os.path.exists(char_cfg_path):
        raise FileNotFoundError(
            f"char_cfg.yaml not found: {char_cfg_path}"
        )

    if not os.path.exists(bounding_box_path):
        raise FileNotFoundError(
            f"bounding_box.yaml not found: {bounding_box_path}"
        )

    with open(
        char_cfg_path,
        "r",
        encoding="utf-8"
    ) as f:
        char_cfg = yaml.safe_load(f)

    with open(
        bounding_box_path,
        "r",
        encoding="utf-8"
    ) as f:
        bounding_box = yaml.safe_load(f)

    crop_width = int(char_cfg["width"])
    crop_height = int(char_cfg["height"])

    left = int(bounding_box["left"])
    top = int(bounding_box["top"])

    def local_point(joint: dict) -> list[int]:
        x = int(joint["x"]) - left
        y = int(joint["y"]) - top

        x = max(
            0,
            min(x, crop_width - 1)
        )

        y = max(
            0,
            min(y, crop_height - 1)
        )

        return [x, y]

    hip_loc = local_point(joint_map["hip"])

    for joint in char_cfg["skeleton"]:
        name = joint["name"]

        if name == "root":
            joint["loc"] = hip_loc
            continue

        if name in joint_map:
            joint["loc"] = local_point(
                joint_map[name]
            )

    with open(
        char_cfg_path,
        "w",
        encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            char_cfg,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    return char_cfg_path
def apply_joint_corrections(annotation_dir: str, joints: list[dict]) -> str:
    """
    ?ъ슜?먭? ?섏젙???먮낯 ?대?吏 湲곗? Joint 醫뚰몴瑜?    AnimatedDrawings??crop-local skeleton 醫뚰몴濡?蹂?섑븯??    char_cfg.yaml??諛섏쁺?쒕떎.
    """

    if not joints:
        raise ValueError("joints are missing")

    required_joints = [
        "hip",
        "torso",
        "neck",
        "right_shoulder",
        "right_elbow",
        "right_hand",
        "left_shoulder",
        "left_elbow",
        "left_hand",
        "right_hip",
        "right_knee",
        "right_foot",
        "left_hip",
        "left_knee",
        "left_foot",
    ]

    joint_map = {
        joint["name"]: joint
        for joint in joints
    }

    missing = [
        name
        for name in required_joints
        if name not in joint_map
    ]

    if missing:
        raise ValueError(
            f"Missing joints: {missing}"
        )

    char_cfg_path = os.path.join(
        annotation_dir,
        "char_cfg.yaml"
    )

    bounding_box_path = os.path.join(
        annotation_dir,
        "bounding_box.yaml"
    )

    if not os.path.exists(char_cfg_path):
        raise FileNotFoundError(
            f"char_cfg.yaml not found: {char_cfg_path}"
        )

    if not os.path.exists(bounding_box_path):
        raise FileNotFoundError(
            f"bounding_box.yaml not found: {bounding_box_path}"
        )

    with open(
        char_cfg_path,
        "r",
        encoding="utf-8"
    ) as f:
        char_cfg = yaml.safe_load(f)

    with open(
        bounding_box_path,
        "r",
        encoding="utf-8"
    ) as f:
        bounding_box = yaml.safe_load(f)

    crop_width = int(char_cfg["width"])
    crop_height = int(char_cfg["height"])

    left = int(bounding_box["left"])
    top = int(bounding_box["top"])

    def local_point(joint: dict) -> list[int]:
        x = int(joint["x"]) - left
        y = int(joint["y"]) - top

        x = max(
            0,
            min(x, crop_width - 1)
        )

        y = max(
            0,
            min(y, crop_height - 1)
        )

        return [x, y]

    hip_loc = local_point(joint_map["hip"])

    for joint in char_cfg["skeleton"]:
        name = joint["name"]

        if name == "root":
            joint["loc"] = hip_loc
            continue

        if name in joint_map:
            joint["loc"] = local_point(
                joint_map[name]
            )

    with open(
        char_cfg_path,
        "w",
        encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            char_cfg,
            f,
            sort_keys=False,
            allow_unicode=True
        )

    return char_cfg_path

