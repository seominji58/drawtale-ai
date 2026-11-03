import os
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs


ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "ELEVENLABS_API_KEY가 설정되지 않았습니다."
    )

client = ElevenLabs(api_key=API_KEY)

def generate_speech(text: str, output_path: str):
    """
    한국어 문장을 ElevenLabs TTS로 변환하여
    음성 파일로 저장한다.
    """

    audio = client.text_to_speech.convert(
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        output_format="mp3_44100_128",
        text=text,
        model_id="eleven_multilingual_v2",
    )

    with open(output_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    return output_path


def generate_story_audio(story_steps, output_dir: str):
    """
    StorySteps의 각 문장을 ElevenLabs TTS로 변환하여
    Step별 MP3 파일을 생성한다.
    """

    os.makedirs(output_dir, exist_ok=True)

    results = []

    for step in story_steps.steps:
        output_path = os.path.join(
            output_dir,
            f"step_{step.step_id:03d}.mp3"
        )

        generate_speech(
            text=step.text,
            output_path=output_path
        )

        results.append({
            "step_id": step.step_id,
            "text": step.text,
            "audio": output_path
        })

    return results