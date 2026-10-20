"""반복 재생 이음새가 눈에 띄는지 렌더된 영상으로 확인한다.

`pick_window.py` 는 관절 좌표로 재고, 이쪽은 **실제로 그려진 그림**으로 잽니다.
마지막 프레임에서 첫 프레임으로 돌아갈 때의 차이가 **보통 프레임 사이 차이와 비슷하면**
매끄럽게 이어집니다. 훨씬 크면 반복할 때마다 튑니다.

    python check_seam.py <영상.gif> [<영상.gif> ...]
"""

import sys

import numpy as np
from PIL import Image, ImageSequence


def main() -> None:
    for path in sys.argv[1:]:
        frames = [np.asarray(f.convert("RGB"), dtype=np.float32)
                  for f in ImageSequence.Iterator(Image.open(path))]
        if len(frames) < 3:
            print(f"{path}: 프레임이 너무 적습니다 ({len(frames)}개)")
            continue

        between = [float(np.abs(frames[i + 1] - frames[i]).mean())
                   for i in range(len(frames) - 1)]
        seam = float(np.abs(frames[0] - frames[-1]).mean())
        average, biggest = float(np.mean(between)), float(np.max(between))

        verdict = "매끄럽습니다" if seam <= biggest else "튈 수 있습니다"
        print(f"{path}: 프레임 {len(frames)}개 | 프레임 사이 차이 평균 {average:.2f} 최대 {biggest:.2f} "
              f"| 이음새 {seam:.2f} ({seam / average:.1f}배) -> {verdict}")


if __name__ == "__main__":
    main()
