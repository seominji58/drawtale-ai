import os
import shutil
import subprocess

from PIL import Image

from ai_server.story_steps import StorySteps
from ai_server.story_validator import is_valid_story, get_invalid_steps
from ai_server.motion_dictionary import find_motion, get_motion_file
from ai_server.tts import generate_story_audio


from ai_server.runtime_config import (
    ANNOTATION_DIR,
    ANIMATED_DRAWINGS_DIR,
    MOTION_CONFIG_DIR,
    RETARGET_CONFIG_DEFAULT,
    RETARGET_CONFIG_JESSE,
    AGENT_RENDER_DIR,
    AUDIO_OUTPUT_DIR,
)


def tool_map_motion(story_steps: StorySteps):
    """
    Story Step留덈떎 ?ъ슜??Motion??寃곗젙?쒕떎.
    """

    results = []

    for step in story_steps.steps:
        motion = find_motion(step.text)

        results.append({
            "step_id": step.step_id,
            "text": step.text,
            "motion": motion
        })

    return results


def tool_validate_story(story_steps: StorySteps):
    """
    Story媛 ?꾩옱 ?좊땲硫붿씠???쒖뒪?쒖뿉??
    泥섎━ 媛?ν븳吏 寃利앺븳??
    """

    valid = is_valid_story(story_steps)
    invalid_steps = get_invalid_steps(story_steps)

    return {
        "valid": valid,
        "invalid_steps": [
            {
                "step_id": step.step_id,
                "text": step.text
            }
            for step in invalid_steps
        ]
    }


def tool_generate_tts(story_steps: StorySteps):
    """
    Story Step蹂??뚯꽦???앹꽦?쒕떎.
    """

    output_dir = AUDIO_OUTPUT_DIR

    return generate_story_audio(
        story_steps,
        output_dir
    )


def render_single_step(step, output_dir):
    """
    ?섎굹??Story Step???대떦 Motion?쇰줈 ?뚮뜑留곹븳??
    """

    motion_name = find_motion(step.text)

    if not motion_name:
        raise RuntimeError(
            f"Step {step.step_id}????묓븯??Motion???놁뒿?덈떎."
        )

    motion_file = get_motion_file(motion_name)

    if not motion_file:
        raise RuntimeError(
            f"Motion ?뚯씪??李얠쓣 ???놁뒿?덈떎: {motion_name}"
        )

    motion_config = os.path.join(
        MOTION_CONFIG_DIR,
        motion_file
    )

    if not os.path.exists(motion_config):
        raise FileNotFoundError(
            f"Motion YAML??議댁옱?섏? ?딆뒿?덈떎: {motion_config}"
        )

    if not os.path.exists(ANNOTATION_DIR):
        raise FileNotFoundError(
            f"Annotation directory媛 議댁옱?섏? ?딆뒿?덈떎: {ANNOTATION_DIR}"
        )

    # --------------------------------
    # Motion???곕Ⅸ Retarget ?ㅼ젙 ?좏깮
    # --------------------------------

    if motion_name == "jesse_dance":
        retarget_config = RETARGET_CONFIG_JESSE
    else:
        retarget_config = RETARGET_CONFIG_DEFAULT

    if not os.path.exists(retarget_config):
        raise FileNotFoundError(
            f"Retarget YAML??議댁옱?섏? ?딆뒿?덈떎: {retarget_config}"
        )

    # --------------------------------
    # Step蹂??낅┰ annotation directory ?앹꽦
    # --------------------------------

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    shutil.copytree(
        ANNOTATION_DIR,
        output_dir
    )

    # --------------------------------
    # AnimatedDrawings ?ㅽ뻾
    # --------------------------------

    command = [
        "python",
        os.path.join(ANIMATED_DRAWINGS_DIR, "examples", "annotations_to_animation.py"),
        output_dir,
        motion_config,
        retarget_config
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = ANIMATED_DRAWINGS_DIR

    result = subprocess.run(
        command,
        cwd=ANIMATED_DRAWINGS_DIR,
        env=env,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Step {step.step_id} render ?ㅽ뙣:\n"
            f"{result.stderr}"
        )

    video_path = os.path.join(
        output_dir,
        "video.gif"
    )

    if not os.path.exists(video_path):
        raise RuntimeError(
            f"?뚮뜑留?寃곌낵 GIF媛 ?앹꽦?섏? ?딆븯?듬땲?? {video_path}"
        )

    return {
        "step_id": step.step_id,
        "motion": motion_name,
        "video": video_path
    }


def combine_gifs(gif_paths, output_path):
    """
    ?щ윭 Step GIF瑜??쒖꽌?濡??섎굹??GIF濡?寃고빀?쒕떎.
    """

    if not gif_paths:
        raise RuntimeError("寃고빀??GIF媛 ?놁뒿?덈떎.")

    frames = []

    for gif_path in gif_paths:

        with Image.open(gif_path) as gif:

            for frame_index in range(gif.n_frames):

                gif.seek(frame_index)

                frame = gif.convert("RGBA").copy()

                duration = gif.info.get(
                    "duration",
                    100
                )

                frames.append(
                    (frame, duration)
                )

    if not frames:
        raise RuntimeError("GIF frame???놁뒿?덈떎.")

    first_frame, first_duration = frames[0]

    output_frames = [
        frame
        for frame, _ in frames
    ]

    durations = [
        duration
        for _, duration in frames
    ]

    first_frame.save(
        output_path,
        save_all=True,
        append_images=output_frames[1:],
        duration=durations,
        loop=0
    )

    return output_path


def tool_render_animation(story_steps: StorySteps):
    """
    Story Step留덈떎 Motion???곸슜?섏뿬 ?뚮뜑留곹븯怨?
    理쒖쥌 GIF ?섎굹濡?寃고빀?쒕떎.
    """

    if os.path.exists(AGENT_RENDER_DIR):
        shutil.rmtree(AGENT_RENDER_DIR)

    os.makedirs(
        AGENT_RENDER_DIR,
        exist_ok=True
    )

    step_results = []

    for step in story_steps.steps:

        step_dir = os.path.join(
            AGENT_RENDER_DIR,
            f"step_{step.step_id:03d}"
        )

        result = render_single_step(
            step,
            step_dir
        )

        step_results.append(result)

    gif_paths = [
        result["video"]
        for result in step_results
    ]

    final_video = os.path.join(
        AGENT_RENDER_DIR,
        "final_animation.gif"
    )

    combine_gifs(
        gif_paths,
        final_video
    )

    return {
        "steps": step_results,
        "video": final_video,
        "message": "All story steps rendered and combined"
    }


def run_agent_tools(story_steps: StorySteps):
    """
    Agent Tool ?꾩껜 ?몄텧 ?먮쫫

    Story
      ??
    Motion Mapping
      ??
    Validation
      ??
    TTS
      ??
    Animation Render
      ??
    Final GIF
    """

    motion_result = tool_map_motion(
        story_steps
    )

    validation_result = tool_validate_story(
        story_steps
    )

    if not validation_result["valid"]:
        return {
            "success": False,
            "stage": "validation",
            "motion": motion_result,
            "validation": validation_result
        }

    audio_result = tool_generate_tts(
        story_steps
    )

    render_result = tool_render_animation(
        story_steps
    )

    return {
        "success": True,
        "motion": motion_result,
        "validation": validation_result,
        "audio": audio_result,
        "animation": render_result
    }


