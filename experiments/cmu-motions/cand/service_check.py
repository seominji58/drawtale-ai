"""새 동작을 서비스 렌더 함수(renderer.render_animation, sprite 포함)로 돌려 본다 (시험용, 2026-11-04).

서비스와 같은 길: 마스크 깎기 · 800px 줄이기 · 메시 마스크 · 투명 프레임 묶음(WebP).
그림은 Meta 예제 아이 그림(texture · mask · 관절)을 원본처럼 넘긴다. 동작마다 걸린 시간 · 프레임 수를 적는다.

    python service_check.py [동작 ...]   (없으면 새 동작 전부)  →  out/svc_{동작}.webp, out/service_check.txt
"""

import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "runtime"))
from ai_server import renderer  # noqa: E402

HERE = Path(__file__).parent.resolve()
CHAR = Path("/opt/AnimatedDrawings/examples/characters/char1")
NEW = ["breathe", "sway", "cheer", "laugh", "hooray", "clap", "skip_walk", "march", "sad_walk",
       "point", "reach", "kick", "search", "tilt", "shiver", "swim", "fly", "monkey", "elephant"]

cfg = yaml.safe_load((CHAR / "char_cfg.yaml").read_text())
joints = [{"name": j["name"], "x": j["loc"][0], "y": j["loc"][1]} for j in cfg["skeleton"]]
bbox = {"left": 0, "top": 0, "right": cfg["width"], "bottom": cfg["height"]}

rows = []
for motion in sys.argv[1:] or NEW:
    t = time.perf_counter()
    try:
        r = renderer.render_animation(str(CHAR / "texture.png"), str(CHAR / "mask.png"), joints, bbox, motion,
                                      str(HERE / "out" / f"svc_{motion}.mp4"), str(HERE / "out" / f"svc_{motion}.webp"))
        sprite = r.get("sprite") or {}
        line = f"{motion:10s} ok   {time.perf_counter() - t:5.1f}s  frames {sprite.get('frames')}"
    except Exception as e:  # noqa: BLE001
        line = f"{motion:10s} FAIL {time.perf_counter() - t:5.1f}s  {e}"
    print(line, flush=True)
    rows.append(line)
(HERE / "out" / "service_check.txt").write_text("\n".join(rows) + "\n", encoding="utf8")
print(json.dumps({"ok": sum(" ok " in r for r in rows), "total": len(rows)}))
