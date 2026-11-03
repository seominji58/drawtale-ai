# ai_server/story_validator.py

from ai_server.motion_dictionary import find_motion


def validate_story_steps(story_steps):
    """
    Story Steps를 검사한다.
    """

    errors = []

    if not story_steps.steps:
        errors.append("Story Step이 없습니다.")
        return errors

    for step in story_steps.steps:
        if not step.text or not step.text.strip():
            errors.append(
                f"Step {step.step_id}: 이야기 문장이 없습니다."
            )
            continue

        motion = find_motion(step.text)

        if motion is None:
            errors.append(
                f"Step {step.step_id}: "
                f"처리 가능한 Motion을 찾을 수 없습니다. "
                f"문장: {step.text}"
            )

    return errors


def is_valid_story(story_steps):
    """
    Story가 애니메이션 처리 가능한지 확인한다.
    """

    errors = validate_story_steps(story_steps)

    return len(errors) == 0


def get_invalid_steps(story_steps):
    """
    Motion을 찾을 수 없는 Story Step만 반환한다.
    """

    invalid_steps = []

    for step in story_steps.steps:
        if not step.text or not step.text.strip():
            invalid_steps.append(step)
            continue

        motion = find_motion(step.text)

        if motion is None:
            invalid_steps.append(step)

    return invalid_steps


def build_retry_prompt(story_steps):
    """
    검증에 실패한 Story Step을
    LLM이 수정할 수 있도록 재처리용 프롬프트를 생성한다.
    """

    invalid_steps = get_invalid_steps(story_steps)

    if not invalid_steps:
        return None

    available_motions = [
        "wave_hello",
        "jumping",
        "jumping_jacks",
        "jesse_dance",
        "dab",
        "zombie",
    ]

    motion_text = ", ".join(available_motions)

    lines = []

    for step in invalid_steps:
        lines.append(
            f"Step {step.step_id}: {step.text}"
        )

    invalid_text = "\n".join(lines)

    return f"""
다음 Story Step은 현재 캐릭터 애니메이션에서
지원할 수 없는 행동을 포함하고 있습니다.

수정해야 할 Story Step:
{invalid_text}

사용 가능한 Motion은 다음과 같습니다.

{motion_text}

위 Motion으로 표현할 수 있도록
문장을 다시 작성하라.

규칙:
1. 기존 이야기의 의미를 최대한 유지한다.
2. 지원하지 않는 행동을 제거한다.
3. 새로운 캐릭터나 사건을 추가하지 않는다.
4. 각 Step은 한 문장으로 작성한다.
5. JSON 형식으로만 출력한다.

출력 형식:

{{
  "steps": [
    {{
      "step_id": 1,
      "text": "수정된 문장"
    }}
  ]
}}
"""