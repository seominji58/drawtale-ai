import os


ANNOTATION_DIR = os.getenv(
    "ANNOTATION_DIR",
    r"D:\capston\render_results\current"
)

ANIMATED_DRAWINGS_DIR = os.getenv(
    "ANIMATED_DRAWINGS_DIR",
    r"D:\capston\AnimatedDrawings"
)

MOTION_CONFIG_DIR = os.getenv(
    "MOTION_CONFIG_DIR",
    os.path.join(
        ANIMATED_DRAWINGS_DIR,
        "examples",
        "config",
        "motion"
    )
)

RETARGET_CONFIG_DEFAULT = os.getenv(
    "RETARGET_CONFIG_DEFAULT",
    os.path.join(
        ANIMATED_DRAWINGS_DIR,
        "examples",
        "config",
        "retarget",
        "fair1_ppf.yaml"
    )
)

RETARGET_CONFIG_JESSE = os.getenv(
    "RETARGET_CONFIG_JESSE",
    os.path.join(
        ANIMATED_DRAWINGS_DIR,
        "examples",
        "config",
        "retarget",
        "fair1_ppf_jesse.yaml"
    )
)

AGENT_RENDER_DIR = os.getenv(
    "AGENT_RENDER_DIR",
    r"D:\capston\render_results\agent"
)

AUDIO_OUTPUT_DIR = os.getenv(
    "AUDIO_OUTPUT_DIR",
    r"D:\capston\A1-AI-Runtime\audio"
)

TORCHSERVE_URL = os.getenv(
    "TORCHSERVE_URL",
    "http://127.0.0.1:8080"
)
