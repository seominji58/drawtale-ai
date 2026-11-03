from typing import List
from pydantic import BaseModel

from ai_server.motion_dictionary import find_motion


class StoryStep(BaseModel):
    step_id: int
    text: str
    motion: str | None = None


class StorySteps(BaseModel):
    steps: List[StoryStep]


def create_story_steps(texts: List[str]) -> StorySteps:
    """
    Story 문장 목록을 입력받아
    각 문장에 적절한 Motion을 자동으로 연결한다.
    """

    steps = []

    for index, text in enumerate(texts, start=1):
        motion = find_motion(text)

        step = StoryStep(
            step_id=index,
            text=text,
            motion=motion
        )

        steps.append(step)

    return StorySteps(steps=steps)