"""Copy the motions picked in experiments/cmu-motions/cand into runtime/ai_server/motions (2026-11-04).

Why: stories showed the drawing standing still on 45% of pages, with only six motions for 1,024 card
stories (pt_docs 17). 32 CMU candidates were rendered on a child's drawing on 11/3 and 11/4; the ones
that kept the body readable in 2D and looked different from each other are listed in PICKED
(18 passed, but in 2D many looked alike — 9 kept, one per kind of movement). Two motions CMU does not have (hooray,
shiver) are built from joint angles by cand/make_handmade.py.

The sources are generated, not committed (cand/work/): run cand/prep.py, make_handmade.py and
prep_front.py first. All of them use the frontal-legs retarget (motions/cmu1_front.yaml, 955f281).

    docker run --rm -v "$PWD:/repo" -w /repo drawtale-ai-ai:latest python runtime/tools/install_more_motions.py
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "experiments" / "cmu-motions" / "cand" / "work"
OUT = REPO / "runtime" / "ai_server" / "motions"

# service name -> (source file stem in cand/work, CMU take or how it was made)
PICKED: dict[str, tuple[str, str]] = {
    "breathe": ("idle_front", "140_06"),
    "march": ("march_front", "91_19, knees x2.0"),
    "point": ("point_front", "13_27"),
    "kick": ("kick_ball_front", "10_01"),
    "search": ("search_ground_front", "77_06"),
    "swim": ("swim_front", "79_02"),
    "fly": ("bird_front", "79_58"),
    "hooray": ("hooray_hand", "joint angles on breathe"),
    "shiver": ("shiver_hand", "joint angles on breathe"),
}

for name, (stem, origin) in PICKED.items():
    cfg = yaml.safe_load((SRC / f"{stem}.yaml").read_text())
    (OUT / f"{name}.bvh").write_bytes((SRC / f"{stem}.bvh").read_bytes())
    cfg["filepath"] = f"{name}.bvh"
    head = (
        f"# {origin} — install_more_motions.py, 2026-11-04. Use with cmu1_front.yaml\n"
        "# CMU Mocap (source: CMU-NOTICE.md) unless built from joint angles\n"
    )
    (OUT / f"{name}.yaml").write_text(head + yaml.safe_dump(cfg, sort_keys=False), encoding="utf8")
    print(name, cfg["end_frame_idx"] + 1, "frames")
