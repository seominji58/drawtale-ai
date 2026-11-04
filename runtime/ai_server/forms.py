"""형태 정의 — 그림의 몸 구조마다 관절 · 관절 찾기 · 동작 · 화면 안내 (pt_docs 19).

forms/{id}.json 하나가 형태 하나다. 분석 · 렌더가 여기서 읽고, backend 가 GET /internal/v1/forms 로 받아
검증 · 화면(관절 고치기 · 형태 고르기)에 쓴다. 새 형태(사슬형 · 날개형 …)는 파일과 관절 찾기 방법을 더해 넣는다.
"""

import json
from functools import lru_cache
from pathlib import Path

FORMS_DIR = Path(__file__).parent / "forms"
DEFAULT = "biped"


@lru_cache(maxsize=1)
def all_forms() -> dict[str, dict]:
    out = {}
    for path in sorted(FORMS_DIR.glob("*.json")):
        form = json.loads(path.read_text(encoding="utf-8"))
        out[form["id"]] = form
    return out


def get(form_id: str | None) -> dict:
    forms = all_forms()
    if (form_id or DEFAULT) not in forms:
        raise KeyError(f"UNKNOWN_FORM: {form_id} (available: {', '.join(forms)})")
    return forms[form_id or DEFAULT]


def parents(form_id: str) -> dict[str, str | None]:
    """관절 이름 → 부모. 보이지 않는 뿌리(root)가 맨 앞 — Meta char_cfg 순서."""
    return {"root": None, **{j["name"]: j["parent"] for j in get(form_id)["joints"]}}


def bones(form_id: str) -> list[tuple[str, str]]:
    """그림 조각을 이어 붙일 뼈(부모 → 자식). 뿌리는 root_at 관절(엉덩이) 자리라 그 관절로 바꿔 잇는다 —
    빼면 뿌리에 달린 다리(4족형 앞뒤 다리)가 몸통과 이어지지 않는다."""
    at = get(form_id).get("root_at", "hip")
    return [(at if p == "root" else p, c) for c, p in parents(form_id).items() if p and c != at]


def form_of(joint_names: set[str]) -> str:
    """관절 이름으로 형태를 알아본다 — 렌더 요청은 형태를 따로 받지 않아도 된다(관절이 형태마다 다르다)."""
    names = joint_names - {"root"}
    for form_id, form in all_forms().items():
        if names == {j["name"] for j in form["joints"]}:
            return form_id
    raise KeyError(f"INVALID_JOINTS: no form has joints {sorted(names)}")


def form_or_default(joint_names: set[str]) -> str:
    """form_of 와 같되, 맞는 형태가 없으면 2족형 — 그다음 검사가 사람 관절 기준으로 무엇이 빠졌는지 알린다."""
    try:
        return form_of(joint_names)
    except KeyError:
        return DEFAULT
