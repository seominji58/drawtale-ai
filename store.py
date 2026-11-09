"""보호자 목소리 보관 — 목소리 하나에 폴더 하나 (참조 음성 + 정보).

Modal 에서는 Volume 위에서 돈다. 여러 컨테이너가 같은 Volume 을 보므로
쓰고 나면 commit, 없는 목소리를 찾을 때는 reload 한 번 해 본다.
"""

import json
import re
import shutil
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_VOICE_ID = re.compile(r"^qv_[0-9a-f]{32}$")


@dataclass(frozen=True)
class Voice:
    id: str
    ref_path: Path
    ref_text: str | None  # 없으면 x-vector 방식 (글 없이 목소리 특징만)
    seconds: float

    @property
    def mode(self) -> str:
        return "text" if self.ref_text else "x_vector"


class VoiceStore:
    def __init__(
        self,
        root: Path,
        commit: Callable[[], None] | None = None,
        reload: Callable[[], None] | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._commit = commit or (lambda: None)
        self._reload = reload or (lambda: None)

    def _dir(self, voice_id: str) -> Path | None:
        # 길 바꾸기(../) 를 막는다: 정해진 모양의 id 만 받는다
        return self.root / voice_id if _VOICE_ID.match(voice_id or "") else None

    def create(self, ref_wav: bytes, ref_text: str | None, seconds: float) -> Voice:
        voice_id = f"qv_{uuid.uuid4().hex}"
        folder = self.root / voice_id
        folder.mkdir()
        (folder / "ref.wav").write_bytes(ref_wav)
        meta = {
            "ref_text": ref_text,
            "seconds": seconds,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        self._commit()
        return Voice(voice_id, folder / "ref.wav", ref_text, seconds)

    def get(self, voice_id: str) -> Voice | None:
        folder = self._dir(voice_id)
        if folder is None:
            return None
        if not (folder / "meta.json").exists():
            # 다른 컨테이너가 방금 만든 목소리일 수 있다
            try:
                self._reload()
            except Exception:
                pass
            if not (folder / "meta.json").exists():
                return None
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return Voice(voice_id, folder / "ref.wav", meta.get("ref_text"), meta.get("seconds", 0.0))

    def delete(self, voice_id: str) -> bool:
        """지웠으면 True, 원래 없었으면 False."""

        folder = self._dir(voice_id)
        if folder is None:
            return False
        try:
            self._reload()
        except Exception:
            pass
        if not folder.exists():
            return False
        shutil.rmtree(folder)
        self._commit()
        return True
