"""Meta AnimatedDrawings 렌더링 래퍼.

Backend가 보낸 관절 좌표(사용자 보정 반영)로 Meta 캐릭터 설정을 만들고
애니메이션 MP4를 생성한다.

렌더링은 화면 없는 환경에서 동작해야 하므로 OSMesa(USE_MESA)를 사용한다.
OpenCV에 H.264 인코더가 없어 MP4를 직접 만들 수 없으므로,
GIF로 렌더링한 뒤 ffmpeg으로 MP4로 변환한다.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

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

# 사용 가능한 동작. Backend가 이 이름으로 요청한다.
MOTIONS: dict[str, tuple[str, str]] = {
    "wave_hello": ("wave_hello", "fair1_ppf"),
    "jumping": ("jumping", "fair1_ppf"),
    "jumping_jacks": ("jumping_jacks", "cmu1_pfp"),
    "dab": ("dab", "fair1_ppf"),
    "zombie": ("zombie", "fair1_spf"),
}
DEFAULT_MOTION = "wave_hello"

# 렌더링 시 캐릭터 영역 둘레에 두는 여유 (bbox 크기 대비 비율).
# 캐릭터가 잘리지 않을 때까지 앞에서부터 차례로 시도한다.
CROP_MARGINS = (0.12, 0.25, 0.40, 0.60)


class RenderError(RuntimeError):
    """message는 "CODE: 설명" 형식으로 둬 Backend가 코드를 읽을 수 있게 한다."""


def build_char_cfg(joints: list[dict], width: int, height: int) -> dict:
    """관절 15개로 Meta char_cfg.yaml 내용을 만든다. 좌표는 crop 이미지 기준."""

    by_name = {j["name"]: j for j in joints}

    missing = [n for n in JOINT_PARENTS if n != "root" and n not in by_name]
    if missing:
        raise RenderError(f"INVALID_JOINTS: missing joints {missing}")

    # root는 hip과 같은 위치에 둔다 (Meta 예시와 동일)
    hip = by_name["hip"]
    by_name["root"] = {"name": "root", "x": hip["x"], "y": hip["y"]}

    skeleton = []
    for name, parent in JOINT_PARENTS.items():
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

    # bbox는 캐릭터에 딱 맞게(때로는 팔다리 끝을 빼고) 잡히므로, 그대로 자르면 끝이 잘린다.
    # 마스크가 가장자리에 닿지 않을 때까지 여유를 넓혀 가며 자른다.
    left, top, right, bottom = expand_crop(mask, bbox, image.shape[1], image.shape[0])

    if right <= left or bottom <= top:
        raise RenderError("INVALID_BBOX: character bounding box is invalid")

    cropped = image[top:bottom, left:right]
    height, width = cropped.shape[:2]

    # 마스크는 원본 전체 크기이므로 같은 범위로 자른다
    mask = mask[top:bottom, left:right]

    # texture는 RGBA여야 한다. 마스크 밖은 투명 처리.
    texture = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
    texture[:, :, 3] = mask

    cv2.imwrite(str(char_dir / "texture.png"), texture)
    cv2.imwrite(str(char_dir / "mask.png"), np.where(mask > 127, 255, 0).astype(np.uint8))

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

    char_cfg = build_char_cfg(crop_joints, width=width, height=height)
    with open(char_dir / "char_cfg.yaml", "w") as f:
        yaml.dump(char_cfg, f)

    return char_dir


def _gif_to_mp4(gif_path: Path, mp4_path: Path) -> None:
    """OpenCV가 H.264를 지원하지 않으므로 ffmpeg으로 변환한다."""

    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(gif_path),
            "-movflags", "faststart",
            "-pix_fmt", "yuv420p",
            # H.264는 가로·세로가 짝수여야 한다
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(mp4_path),
        ],
        capture_output=True,
        timeout=300,
    )

    if result.returncode != 0 or not mp4_path.exists():
        detail = result.stderr.decode("utf-8", "replace")[-500:]
        raise RenderError(f"ENCODING_FAILED: ffmpeg failed: {detail}")


def render_animation(
    image_path: str,
    mask_path: str,
    joints: list[dict],
    bbox: dict,
    motion: str,
    output_path: str,
) -> dict:
    """관절 좌표로 애니메이션 MP4를 만든다."""

    started_at = time.perf_counter()

    if motion not in MOTIONS:
        raise RenderError(
            f"UNKNOWN_MOTION: {motion} (available: {', '.join(MOTIONS)})"
        )

    motion_name, retarget_name = MOTIONS[motion]

    work_dir = Path(tempfile.mkdtemp(prefix="drawtale-render-"))

    try:
        char_dir = prepare_character(image_path, mask_path, joints, bbox, work_dir)

        gif_path = work_dir / "video.gif"
        mvc_cfg = {
            "scene": {
                "ANIMATED_CHARACTERS": [
                    {
                        "character_cfg": str(char_dir / "char_cfg.yaml"),
                        "motion_cfg": str(
                            ANIMATED_DRAWINGS_DIR
                            / f"examples/config/motion/{motion_name}.yaml"
                        ),
                        "retarget_cfg": str(
                            ANIMATED_DRAWINGS_DIR
                            / f"examples/config/retarget/{retarget_name}.yaml"
                        ),
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

        mvc_path = work_dir / "mvc_cfg.yaml"
        with open(mvc_path, "w") as f:
            yaml.dump(mvc_cfg, f)

        # Meta 코드는 상대 경로를 자기 저장소 기준으로 해석한다
        cwd = os.getcwd()
        try:
            os.chdir(ANIMATED_DRAWINGS_DIR)
            import animated_drawings.render

            animated_drawings.render.start(str(mvc_path))
        except RenderError:
            raise
        except Exception as e:
            raise RenderError(f"RENDER_FAILED: {e}") from e
        finally:
            os.chdir(cwd)

        if not gif_path.exists():
            raise RenderError("RENDER_FAILED: no output produced")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        _gif_to_mp4(gif_path, output)

        return {
            "format": "mp4",
            "motion": motion,
            "size_bytes": output.stat().st_size,
            "processing_time_ms": round((time.perf_counter() - started_at) * 1000),
        }

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
