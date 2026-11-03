import os
from typing import Callable, Any


def run_pose_adapter(
    cropped: Any,
    bbox: dict,
    meta_estimator: Callable,
    a2_estimator: Callable | None = None
) -> list:
    """
    Pose provider를 교체 가능하게 만드는 A1 adapter.

    POSE_PROVIDER=meta
        -> Meta Animated Drawings pose estimator 사용

    POSE_PROVIDER=a2
        -> A2 pose estimator 사용
    """

    provider = os.getenv(
        "POSE_PROVIDER",
        "meta"
    ).strip().lower()

    if provider == "meta":
        return meta_estimator(
            cropped,
            bbox
        )

    if provider == "a2":
        if a2_estimator is None:
            raise RuntimeError(
                "MODEL_UNAVAILABLE: "
                "A2 pose adapter is not configured"
            )

        return a2_estimator(
            cropped,
            bbox
        )

    raise RuntimeError(
        f"INVALID_CONFIG: Unsupported pose provider: {provider}"
    )
