"""오린 그림(투명 바탕)의 투명도를 몸 모양(마스크)으로 쓴다 (2026-11-09, 장면 등장인물).

분석은 아이 그림용이라 흰 종이 위 선을 찾고 칠하지 않은 흰 곳을 종이로 지운다. 장면 등장인물 그림(크레용 일러스트,
이미 배경을 오려 투명 바탕)은 흰 셔츠 소매 · 흰 가운이 종이로 지워져 렌더에서 팔이 사라지고 손만 떠 있었다
(동굴 안내원 · 가게 주인 · 간호사, 프론트 tools/people-moves/check.py). 이런 그림은 투명도가 곧 몸 모양이다.

analyze 의 mask_from="alpha" 일 때만 쓴다, 아이 그림 분석은 바뀌지 않는다.
"""

from pathlib import Path

import cv2
import numpy as np


def read_alpha(path: str | Path) -> np.ndarray | None:
    """투명 칸이 있는 그림이면 0/255 마스크, 아니면 None. 분석이 파일을 흰 바탕으로 바꾸기 전에 읽는다"""

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    a = img[..., 3]
    if a.dtype == np.uint16:
        a = (a / 257).astype(np.uint8)
    mask = np.where(a > 24, 255, 0).astype(np.uint8)
    # 그림이 거의 다 불투명하면(투명 바탕이 아니면) 쓰지 않는다
    if mask.mean() / 255 > 0.97:
        return None
    # 오리기 가장자리의 작은 틈 · 구멍 메우기 (선으로 닫힌 안쪽은 몸이다)
    filled = mask.copy()
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(filled, flood, (0, 0), 255)
    holes = cv2.bitwise_not(filled)
    return cv2.bitwise_or(mask, holes)


def write_mask(mask: np.ndarray, mask_path: str | Path) -> bool:
    Path(mask_path).parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(mask_path), mask))
