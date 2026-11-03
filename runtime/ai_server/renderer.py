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
    "jumping_gentle": (_GENTLE_MOTIONS / "jumping_gentle.yaml", _META_RETARGET / "fair1_ppf.yaml"),
    "walk": (_GENTLE_MOTIONS / "walk.yaml", _GENTLE_MOTIONS / "cmu1_human.yaml"),
    "run": (_GENTLE_MOTIONS / "run.yaml", _GENTLE_MOTIONS / "cmu1_human.yaml"),
    "dab_gentle": (_GENTLE_MOTIONS / "dab_gentle.yaml", _META_RETARGET / "fair1_ppf.yaml"),
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

# 움직이는 틀(메시)을 이을 뼈대 띠 굵기 (자른 그림 대각선 대비)
BONE_BAND = 0.012
BONES = (
    ("hip", "torso"), ("torso", "neck"),
    ("neck", "right_shoulder"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
    ("neck", "left_shoulder"), ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
    ("hip", "right_hip"), ("right_hip", "right_knee"), ("right_knee", "right_foot"),
    ("hip", "left_hip"), ("left_hip", "left_knee"), ("left_knee", "left_foot"),
)


def mesh_mask(mask: np.ndarray, joints: list[dict]) -> np.ndarray:
    """Meta 는 마스크에서 가장 큰 덩어리 하나만 움직인다 (나머지는 버림).
    종이색을 깎으면 가는 팔다리 선이 몸통과 떨어진 조각이 되므로, 관절 사이 뼈대 띠로 이어 붙인다.
    보이는 부분은 텍스처 투명도(깎은 마스크)가 정하니 띠 자리의 종이는 보이지 않는다 (2026-11-03)."""

    out = mask.copy()
    at = {j["name"]: (int(j["x"]), int(j["y"])) for j in joints}
    width = max(3, round(BONE_BAND * float(np.hypot(*mask.shape[:2]))))
    for a, b in BONES:
        if a in at and b in at:
            cv2.line(out, at[a], at[b], 255, width)
    # 윤곽을 매끈하게 — Meta 는 윤곽 점마다 계산이 커져, 깎은 마스크의 들쭉날쭉한 윤곽 그대로면
    # 렌더 한 번에 메모리가 1.6GB 를 넘어 컨테이너(2GB)가 죽었다. 틀이 조금 커져도 텍스처가 투명이라 안 보인다
    k = max(3, round(BONE_BAND * float(np.hypot(*mask.shape[:2]))))
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    out = np.where(cv2.GaussianBlur(out, (0, 0), k) > 127, 255, 0).astype(np.uint8)
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
    height, width = cropped.shape[:2]

    # 마스크는 원본 전체 크기이므로 같은 범위로 자른다
    mask = mask[top:bottom, left:right].copy()

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
    mesh = mesh_mask(np.where(mask > 127, 255, 0).astype(np.uint8), crop_joints)
    cv2.imwrite(str(char_dir / "mask.png"), cv2.GaussianBlur(mesh, (0, 0), MESH_SMOOTH))

    char_cfg = build_char_cfg(crop_joints, width=width, height=height)
    with open(char_dir / "char_cfg.yaml", "w") as f:
        yaml.dump(char_cfg, f)

    return char_dir


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
) -> dict:
    """관절 좌표로 애니메이션 MP4를 만든다."""

    started_at = time.perf_counter()

    if motion not in MOTIONS:
        raise RenderError(
            f"UNKNOWN_MOTION: {motion} (available: {', '.join(MOTIONS)})"
        )

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
