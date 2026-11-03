# ai_server/motion_dictionary.py

MOTION_DICTIONARY = {
    "wave_hello": {
        "file": "wave_hello.yaml",
        "keywords": [
            "인사",
            "인사하다",
            "손 흔들기",
            "손을 흔들다",
            "안녕",
            "반갑다",
        ],
    },

    "jumping": {
        "file": "jumping.yaml",
        "keywords": [
            "점프",
            "점프하다",
            "뛰다",
            "뛰어오르다",
        ],
    },

    "jumping_jacks": {
        "file": "jumping_jacks.yaml",
        "keywords": [
            "운동",
            "운동하다",
            "팔벌려뛰기",
            "팔 벌려 뛰기",
        ],
    },

    "jesse_dance": {
        "file": "jesse_dance.yaml",
        "keywords": [
            "춤",
            "춤추다",
            "춤을 추다",
            "댄스",
        ],
    },

    "dab": {
        "file": "dab.yaml",
        "keywords": [
            "dab",
            "Dab",
            "댑",
        ],
    },

    "zombie": {
        "file": "zombie.yaml",
        "keywords": [
            "좀비",
            "좀비처럼 움직이다",
            "좀비처럼 걷다",
        ],
    },
}


def find_motion(text: str) -> str | None:
    """
    Story Step 문장을 입력받아
    사용할 Motion 이름을 반환한다.
    """

    if not text:
        return None

    text = text.lower().strip()

    for motion_name, motion_info in MOTION_DICTIONARY.items():
        for keyword in motion_info["keywords"]:
            if keyword.lower() in text:
                return motion_name

    return None


def get_motion_file(motion_name: str) -> str | None:
    """
    Motion 이름으로 실제 YAML 파일명을 반환한다.
    """

    motion = MOTION_DICTIONARY.get(motion_name)

    if motion is None:
        return None

    return motion["file"]