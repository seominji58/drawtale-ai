"""BVH 읽기 · 관절 위치 계산 (2026-11-09, 장면 등장인물 동작 만들기용). 앞모습(x 오른쪽, y 위)으로 본 뼈 방향을 잰다."""
import re
from pathlib import Path

import numpy as np


class Bvh:
    def __init__(self, path: Path):
        text = Path(path).read_text()
        head, motion = text.split("MOTION", 1)
        self.header = head + "MOTION\n"
        self.names, self.parent, self.offset, self.chan = [], [], [], []
        stack, last = [], None
        for tok in re.finditer(r"(ROOT|JOINT)\s+(\S+)|End Site|OFFSET\s+(\S+)\s+(\S+)\s+(\S+)|CHANNELS\s+(\d+)\s+([^\n]+)|(\{)|(\})", head):
            if tok.group(1):
                self.names.append(tok.group(2)); self.parent.append(stack[-1] if stack else -1)
                self.offset.append(None); self.chan.append([]); last = len(self.names) - 1
            elif tok.group(0) == "End Site":
                last = "end"
            elif tok.group(3) is not None:
                if last != "end" and last is not None and self.offset[last] is None:
                    self.offset[last] = np.array([float(tok.group(i)) for i in (3, 4, 5)])
            elif tok.group(6):
                self.chan[last] = tok.group(7).split()
            elif tok.group(8):
                stack.append(last)
            elif tok.group(9):
                stack.pop()
        lines = motion.strip().splitlines()
        self.dt = float(lines[1].split(":")[1])
        self.frames = np.array([[float(v) for v in l.split()] for l in lines[2:]])
        self.col = {}
        c = 0
        for i, ch in enumerate(self.chan):
            for k in ch:
                self.col[(self.names[i], k)] = c; c += 1

    def idx(self, name): return self.names.index(name)

    def world(self, row: np.ndarray) -> dict:
        R = {}; P = {}
        for i, n in enumerate(self.names):
            rot = np.eye(3); pos = self.offset[i].copy()
            for k in self.chan[i]:
                v = row[self.col[(n, k)]]
                if k.endswith("position"):
                    pos["XYZ".index(k[0])] += v if self.parent[i] == -1 else 0
                else:
                    a = np.radians(v); c, s = np.cos(a), np.sin(a)
                    m = {"X": [[1, 0, 0], [0, c, -s], [0, s, c]], "Y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
                         "Z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[k[0]]
                    rot = rot @ np.array(m)
            p = self.parent[i]
            if p == -1:
                R[n], P[n] = rot, pos
            else:
                pn = self.names[p]
                P[n] = P[pn] + R[pn] @ self.offset[i]; R[n] = R[pn] @ rot
        return P

    def write(self, path: Path, frames: np.ndarray, dt: float):
        body = "\n".join(" ".join(f"{v:.5f}" for v in r) for r in frames)
        Path(path).write_text(self.header + f"Frames: {len(frames)}\nFrame Time: {dt}\n" + body + "\n")


def angle2d(P, a, b):
    """앞에서 본 a → b 방향 각도 (도, 0 = 오른쪽(화면), 90 = 위)"""
    d = P[b] - P[a]
    return float(np.degrees(np.arctan2(d[1], d[0])))
