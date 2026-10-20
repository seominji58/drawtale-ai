"""BVH 프레임을 솎아 낸다 (시험용. README.md 참고).
 CMU 는 초당 120프레임이라 그대로 쓰면 4배 느리게 보인다.

    python resample.py <입력.bvh> <출력.bvh> <몇 개마다 하나>
"""
import io, sys

src, dst, step = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = io.open(src, encoding="utf8").read().splitlines()
i = next(n for n, l in enumerate(lines) if l.strip().startswith("Frames:"))
head, n_old = lines[:i], int(lines[i].split(":")[1])
ft_line = lines[i + 1]
ft_old = float(ft_line.split(":")[1])
frames = lines[i + 2:]
kept = frames[::step]
out = head + [f"Frames: {len(kept)}", f"Frame Time:\t{ft_old*step:.7f}"] + kept
io.open(dst, "w", encoding="utf8").write("\n".join(out) + "\n")
print(f"{n_old}프레임 {1/ft_old:.0f}fps -> {len(kept)}프레임 {1/(ft_old*step):.0f}fps")
