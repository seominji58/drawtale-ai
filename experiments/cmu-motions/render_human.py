"""사람 그림에 CMU 동작을 걸어 본다 (시험용).

사람은 AI 가 관절을 찾아 주므로 손으로 찍지 않는다.
분석(analyze)을 먼저 돌려 받은 결과와, 서비스 코드의 캐릭터 준비 함수를 그대로 쓴다.
우리 API 는 등록된 동작만 받으므로 동작 설정만 직접 만들어 Meta 엔진을 부른다.

쓰는 방법은 같은 폴더의 README.md 5장에 있다.

    python render_human.py <request_id> <walk|run> <분석결과.json>
"""

import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/app")
from ai_server import renderer  # noqa: E402

BASE = Path("/w")  # 컨테이너에 물린 이 폴더

req_id = sys.argv[1]
motion_name = sys.argv[2]
analyze_json = Path(sys.argv[3] if len(sys.argv) > 3 else BASE / "analyze.json")

# analyze 가 남긴 작업 폴더에 원본 그림과 마스크가 있다
session = Path("/tmp/drawtale") / req_id
if not session.is_dir():
    raise SystemExit(
        f"작업 폴더가 없습니다: {session}\n"
        "analyze 를 돌린 그 컨테이너 안에서 실행해야 합니다 (세션이 컨테이너에 남습니다)."
    )

analysis = json.loads((session / "analysis.json").read_text())
joints = json.loads(analyze_json.read_text())["joints"]

work = BASE / "out" / "human"
work.mkdir(parents=True, exist_ok=True)

# 서비스 코드 그대로 — 자르기 · 마스크 · 관절 좌표 옮기기 · char_cfg 생성
char_dir = renderer.prepare_character(
    str(session / analysis["source"]),
    str(session / "mask.png"),
    joints,
    analysis["bbox"],
    work,
)

bvh = BASE / "bvh" / f"{motion_name}.bvh"
frames = sum(1 for _ in bvh.open(encoding="utf8")) - 1  # 넉넉히 잡고 아래에서 줄인다
motion = {
    "filepath": str(bvh),
    "start_frame_idx": 0,
    # 다듬은 파일의 프레임 수에 맞춘다 (모자라면 Meta 가 알아서 멈춘다)
    "end_frame_idx": 36 if motion_name == "run" else 78,
    "groundplane_joint": "LeftFoot",
    "forward_perp_joint_vectors": [
        ["LeftShoulder", "RightShoulder"],
        ["LeftUpLeg", "RightUpLeg"],
    ],
    "scale": 0.025,
    "up": "+y",
}
motion_path = work / f"motion_{motion_name}.yaml"
motion_path.write_text(yaml.dump(motion))

mvc = {
    "scene": {
        "ANIMATED_CHARACTERS": [
            {
                "character_cfg": str(char_dir / "char_cfg.yaml"),
                "motion_cfg": str(motion_path),
                "retarget_cfg": str(BASE / "retarget" / "cmu1_human.yaml"),
            }
        ]
    },
    "controller": {
        "MODE": "video_render",
        "OUTPUT_VIDEO_PATH": str(BASE / "out" / f"human_{motion_name}.gif"),
    },
    "view": {"USE_MESA": True},
}
mvc_path = work / f"mvc_{motion_name}.yaml"
mvc_path.write_text(yaml.dump(mvc))

# Meta 코드는 상대 경로를 자기 저장소 기준으로 읽는다
os.chdir("/opt/AnimatedDrawings")
import animated_drawings.render as render  # noqa: E402

render.start(str(mvc_path))
print(f"끝났습니다: out/human_{motion_name}.gif")
