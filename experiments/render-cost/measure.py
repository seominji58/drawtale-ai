"""렌더 시간이 무엇에 비례하는지 잰다 (시험용).

같은 그림 · 같은 동작으로 **그리는 프레임 수만** 바꿔 가며 잰다.
Meta 예제 캐릭터를 쓰므로 AI 런타임 이미지만 있으면 되고, 모델 서버는 필요 없다.

결과와 해석은 설계 문서
[12 AI 아키텍처 4장](https://github.com/seominji58/drawtale-pt_docs/blob/main/12-ai-architecture.md)
에 있다.

    docker run --rm -v "${PWD}:/w" drawtale-ai-ai:latest python /w/experiments/render-cost/measure.py
"""

import os
import sys
import time
from pathlib import Path

import yaml

AD = Path("/opt/AnimatedDrawings")
OUT = Path("/tmp/render-cost")

# 재는 프레임 수. 첫 번째는 두 번 돌린다 — 처음 한 번은 파이썬 불러오기와
# 그래픽 초기화가 섞여서 느리게 나온다 (버리는 값).
FRAME_COUNTS = [37, 37, 237, 839]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    motion = yaml.safe_load((AD / "examples/config/motion/wave_hello.yaml").read_text())
    motion["filepath"] = str(AD / "examples/bvh/fair1/wave_hello.bvh")
    motion_path = OUT / "motion.yaml"

    mvc_path = OUT / "mvc.yaml"
    mvc_path.write_text(yaml.dump({
        "scene": {"ANIMATED_CHARACTERS": [{
            "character_cfg": str(AD / "examples/characters/char1/char_cfg.yaml"),
            "motion_cfg": str(motion_path),
            "retarget_cfg": str(AD / "examples/config/retarget/fair1_ppf.yaml"),
        }]},
        "controller": {"MODE": "video_render", "OUTPUT_VIDEO_PATH": str(OUT / "out.gif")},
        "view": {"USE_MESA": True},
    }))

    # Meta 코드는 상대 경로를 자기 저장소 기준으로 읽는다
    os.chdir(AD)
    sys.path.insert(0, str(AD))
    import animated_drawings.render as render

    results = []
    for n in FRAME_COUNTS:
        motion["end_frame_idx"] = n - 1
        motion_path.write_text(yaml.dump(motion))
        started = time.time()
        render.start(str(mvc_path))
        results.append((n, time.time() - started))

    print("\n== 결과 ==")
    for n, secs in results:
        print(f"{n:>5} 프레임   {secs:5.1f}초")

    # 첫 값은 버리고 직선을 긋는다
    (n1, t1), (n2, t2) = results[1], results[-1]
    per = (t2 - t1) / (n2 - n1)
    print(f"\n프레임당 {per:.3f}초 + 고정 {t1 - n1 * per:.1f}초")
    print("고정 몫은 그물 만들기 · 그래픽 초기화 · 영상 인코딩이다 (프레임 수와 무관).")


if __name__ == "__main__":
    main()
