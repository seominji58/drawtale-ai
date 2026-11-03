"""동작 하나를 Meta 예제 아이 그림 하나에 렌더한다 (시험용).

    python render_one.py <동작_변형> <char1|char3|...>   →  out/{캐릭터}_{동작_변형}.gif
"""

import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent.resolve()
motion, char = sys.argv[1], sys.argv[2]
out = HERE / "out"
out.mkdir(exist_ok=True)
mvc = {
    "scene": {"ANIMATED_CHARACTERS": [{
        "character_cfg": f"/opt/AnimatedDrawings/examples/characters/{char}/char_cfg.yaml",
        "motion_cfg": str(HERE / "work" / f"{motion}.yaml"),
        "retarget_cfg": str(HERE.parent / "retarget" / "cmu1_human.yaml"),
    }]},
    "controller": {"MODE": "video_render", "OUTPUT_VIDEO_PATH": str(out / f"{char}_{motion}.gif")},
    "view": {"USE_MESA": True},
}
path = HERE / "work" / f"mvc_{char}_{motion}.yaml"
path.write_text(yaml.safe_dump(mvc))
os.chdir("/opt/AnimatedDrawings")
import animated_drawings.render as render  # noqa: E402

render.start(str(path))
