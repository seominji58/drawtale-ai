"""장면 등장인물용 동작 두 개를 관절 각도로 만든다 (2026-11-09): scene_wave(한 팔 들고 흔들기) · scene_walk(일자 제자리 걸음).

장면 등장인물(프론트 tools/people-moves)은 팔을 몸에서 22° 쯤 벌리고 서 있는 정면 그림이다 (33명 위팔 -68° / -113°,
다리 -88° / -92°, 표준편차 4° 안쪽). 앞모습 재적용(cmu1_front)은 뼈 길이는 그림 그대로 두고 방향만 BVH 에서 가져오므로
- 인사(wave_hello_gentle)는 두 팔을 다 옆으로 들어 손바닥이 바닥을 보고 몸통이 기울었다 → 한 팔은 그린 자세 그대로,
  다른 팔은 위팔을 두고 팔꿈치만 굽혀 아래팔을 세워 좌우로 흔든다(내리지 않고 화면이 흔들기를 되풀이) (사용자 「한 팔은 냅두고 한 팔만 위로 올려서 흔들면 되는데」)
- 걷기(walk_front)는 무릎을 앞으로 드는 걸음이라 앞에서 보면 다리가 옆으로 크게 꺾였다 → 무릎은 굽히지 않고 두 다리를
  곧게 아래로 둔 채 드는 발 쪽 골반을 올려 다리가 통째로 살짝 올라가는 일자 걸음, 팔은 조금만 흔든다
첫 장은 그린 자세라 그림띠가 멈춘 그림에서 이어진다 (걷기는 끝 장도).

재적용은 scene_front.yaml(cmu1_front 에서 팔도 정면 투영). 뼈대는 walk_steps.bvh(CMU)를 그대로 쓰고 모든 관절을 Z(정면 평면) 회전만으로 놓는다. 각도는 화면 기준(0 = 오른쪽, 90 = 위),
그림의 왼쪽(화면 오른쪽)이 BVH Left.

  python runtime/tools/make_scene_motions.py      → runtime/ai_server/motions/scene_wave.* · scene_walk.*
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bvh_fk import Bvh, angle2d  # noqa: E402

MOTIONS = Path(__file__).resolve().parents[1] / "ai_server" / "motions"
FPS = 30

# 그린 자세 (33명 평균, 프론트 tools/people-moves 로 잰 값)
REST = {"LeftArm": -68, "LeftForeArm": -67, "RightArm": -113, "RightForeArm": -113,
        "LeftUpLeg": -88, "LeftLeg": -88, "RightUpLeg": -92, "RightLeg": -93}
CHILD = {"LeftArm": "LeftForeArm", "LeftForeArm": "LeftHand", "RightArm": "RightForeArm", "RightForeArm": "RightHand",
         "LeftUpLeg": "LeftLeg", "LeftLeg": "LeftFoot", "RightUpLeg": "RightLeg", "RightLeg": "RightFoot"}


def pose(b: Bvh, want: dict, hips_tilt: float = 0.0, bob: float = 0.0) -> np.ndarray:
    """화면 각도(want: 관절 → 그 관절에서 나가는 뼈의 각도)를 Z 회전으로 푼 한 장"""

    row = np.zeros(len(b.col))
    row[b.col[("Hips", "Zrotation")]] = hips_tilt
    if ("Spine1", "Zrotation") in b.col:
        row[b.col[("Spine1", "Zrotation")]] = -hips_tilt   # 골반이 기울어도 윗몸은 곧게

    def acc(name):   # 조상(자기 빼고) Z 회전 합
        s, p = 0.0, b.parent[b.idx(name)]
        while p != -1:
            s += row[b.col[(b.names[p], "Zrotation")]]; p = b.parent[p]
        return s

    # 부모부터 차례로 (팔 · 다리 사슬)
    for j in ("LeftUpLeg", "LeftLeg", "RightUpLeg", "RightLeg", "LeftArm", "LeftForeArm", "RightArm", "RightForeArm"):
        off = b.offset[b.idx(CHILD[j])]
        base = np.degrees(np.arctan2(off[1], off[0]))
        row[b.col[(j, "Zrotation")]] = want[j] - base - acc(j)
    # 발 · 손은 앞 뼈를 따라 (그대로 0)
    P = b.world(row)
    feet = min(P["LeftToeBase"][1], P["RightToeBase"][1], P["LeftFoot"][1], P["RightFoot"][1])
    row[b.col[("Hips", "Yposition")]] = -feet + bob
    return row


def ease(t):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(t, 0, 1))


WAVE_FPS = 20
WAVE_UP = 9          # 손 드는 장 (0.45초). 그 뒤 흔들기 세 번(30장)은 끝과 처음이 이어져 화면이 끝없이 되풀이한다
WAVE_CYCLE = 10      # 한 번 흔드는 장 (0.5초)


def wave(b: Bvh) -> np.ndarray:
    """왼팔(화면 오른쪽) 아래팔을 세워 들고(9장) 좌우로 세 번 흔든다(30장). 내리지 않는다: 화면이 흔들기 30장을 쪽이 바뀔 때까지
    되풀이한다 (11/9 사용자 「손 들고 계속 흔드는 거 무한 유지」). 오른팔 · 다리는 그린 자세.
    화면 왼쪽 팔은 우체부 가방 · 경찰 권총집이 손 옆에 붙어 있어 팔을 들면 같이 딸려 올라갔다 (11/9).
    48장 이하라 렌더가 장을 줄이지 않는다 (장 사이 1/20초 그대로)"""

    out = []
    for i in range(WAVE_UP + 3 * WAVE_CYCLE + 1):   # 렌더가 마지막 장을 빼서 하나 더 (렌더 결과 39장 = 들기 9 + 흔들기 30)
        rise = ease(i / WAVE_UP)
        # 위팔은 그린 자세 그대로(-68°) 두고 팔꿈치만 굽혀 아래팔을 세워(68°) 좌우로 흔든다. 손은 얼굴 옆까지.
        # 위팔을 어깨 위 · 옆으로 들면 어깨 옆 몸통이 끌려가 몸 전체가 기울었다 (11/9 시험 3가지)
        sway = np.sin(2 * np.pi * max(0, i - WAVE_UP) / WAVE_CYCLE) * 18
        w = dict(REST)
        w["LeftForeArm"] = -67 + (68 + sway + 67) * rise
        out.append(pose(b, w))
    return np.array(out)


def walk(b: Bvh) -> np.ndarray:
    """일자 걸음 네 걸음(한 걸음 0.5초): 두 다리는 그린 자세 그대로 곧게 아래로 두고, 드는 발 쪽 골반을 올려 그 다리가 통째로
    살짝 올라가게 한다. 몸은 위아래로 조금, 팔은 조금 흔든다. 앞모습 재적용은 뼈 길이가 그대로라 무릎을 굽히면 다리가 옆으로
    꺾이거나(무릎 앞으로) X 자로 엇갈렸고(무릎 옆으로), 다리를 옆으로 들면 팔자걸음이 됐다 (11/9 사용자 「걷기 이상」 「일자걸음으로」)"""

    step, steps = 0.5, 4
    n = int(step * steps * FPS)
    out = []
    for i in range(n):
        t = i / FPS
        ph = (t / step) % 2                         # 0~1 왼발 듦, 1~2 오른발 듦
        lift = np.sin(np.pi * (ph % 1))             # 0 → 1 → 0
        side = 1 if ph < 1 else -1
        w = dict(REST)
        swing = 3 * lift * side                     # 팔은 조금
        for arm in ("LeftArm", "LeftForeArm", "RightArm", "RightForeArm"):
            w[arm] -= swing
        out.append(pose(b, w, hips_tilt=6 * lift * side, bob=0.2 * lift))
    return np.array(out)


YAML = """# 관절 각도로 만든 장면 등장인물 동작 (runtime/tools/make_scene_motions.py, 2026-11-09). scene_front.yaml 과 함께 쓴다
filepath: {name}.bvh
start_frame_idx: 0
end_frame_idx: {last}
groundplane_joint: LeftFoot
forward_perp_joint_vectors:
- - LeftShoulder
  - RightShoulder
- - LeftUpLeg
  - RightUpLeg
scale: 0.025
up: +y
"""


def main() -> None:
    b = Bvh(MOTIONS / "walk_steps.bvh")
    for name, make, fps in (("scene_wave", wave, WAVE_FPS), ("scene_walk", walk, FPS)):
        fr = make(b)
        b.write(MOTIONS / f"{name}.bvh", fr, 1 / fps)
        (MOTIONS / f"{name}.yaml").write_text(YAML.format(name=name, last=len(fr) - 1), encoding="utf-8")
        # 확인: 첫 장 각도가 그린 자세와 같은지
        P = Bvh(MOTIONS / f"{name}.bvh").world(fr[0])
        got = {j: round(angle2d(P, j, CHILD[j])) for j in REST}
        print(name, len(fr), "장", got)


if __name__ == "__main__":
    main()
