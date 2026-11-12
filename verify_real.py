"""배포된 drawtale-tts 를 실제 GPU 로 끝까지 시험하고 결과를 남긴다.

    python verify_real.py --url https://...modal.run --ref ref_audio.wav --ref-text "녹음에서 읽은 문장"

토큰은 TTS_API_TOKEN 환경 변수나 ~/.tts_token 에서 읽는다 (화면 · 결과 파일에 남기지 않는다).
결과: verify_out/ 에 생성 음성(.mp3)과 report.md (시간 · 성공/실패). 시험에 쓴 목소리는 끝에 지운다.
콜드 스타트를 재려면 마지막 요청 후 5분 넘게 쉰 뒤에 실행한다.
"""

import argparse
import base64
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

TEXTS = [
    "친구와 함께 재미있게 놀았어요.",
    "햇살이 반짝이는 아침, 나는 숲길을 따라 천천히 걸어갔어요.",
    "부엉이가 나뭇가지 위에서 고개를 갸웃거리며 나를 바라보았어요.",
]


def token() -> str:
    t = os.environ.get("TTS_API_TOKEN") or Path("~/.tts_token").expanduser().read_text().strip()
    if not t:
        raise SystemExit("토큰이 없습니다 (TTS_API_TOKEN 또는 ~/.tts_token)")
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--ref", required=True, help="참조 음성 파일")
    ap.add_argument("--ref-text", default=None, help="참조 음성에서 읽은 문장 (없으면 x-vector)")
    ap.add_argument("--out", default="verify_out")
    ap.add_argument("--keep-voice", action="store_true", help="시험 목소리를 지우지 않는다")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    rows: list[tuple[str, str, str]] = []  # (단계, 결과, 시간)
    http = httpx.Client(
        base_url=args.url.rstrip("/"),
        headers={"Authorization": f"Bearer {token()}"},
        timeout=600,
        follow_redirects=True,  # Modal 303 (오래 걸림)
    )

    def step(name: str, fn):
        t0 = time.perf_counter()
        try:
            detail = fn()
            rows.append((name, f"성공 {detail or ''}".strip(), f"{time.perf_counter() - t0:.1f}초"))
            print(f"✓ {name} ({time.perf_counter() - t0:.1f}s) {detail or ''}")
            return True
        except Exception as e:  # noqa: BLE001
            rows.append((name, f"실패: {e}", f"{time.perf_counter() - t0:.1f}초"))
            print(f"✗ {name}: {e}")
            return False

    def health():
        r = http.get("/internal/v1/health")
        r.raise_for_status()
        return r.json()["model"]

    voice = {}

    def register():
        data = {"ref_text": args.ref_text} if args.ref_text else {}
        with open(args.ref, "rb") as f:
            r = http.post("/internal/v1/voices", files={"audio": (Path(args.ref).name, f)}, data=data)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} {r.text[:200]}")
        voice.update(r.json())
        return f"mode={voice['mode']} 참조 {voice['seconds']}초"

    def speak(i: int, text: str):
        def run():
            r = http.post("/internal/v1/tts", json={"voice_id": voice["voice_id"], "text": text, "speed": 0.9})
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code} {r.text[:200]}")
            (out / f"tts_{i}.mp3").write_bytes(r.content)
            return f"{len(text)}자 → tts_{i}.mp3 (서버 처리 {r.headers.get('x-processing-seconds')}초)"

        return run
        
    def plain():
        r = http.post(
            "/internal/v1/tts",
            json={"voice_id": voice["voice_id"], "text": TEXTS[0], "speed": 0.9, "enhance": False},
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} {r.text[:200]}")
        (out / "tts_1_plain.mp3").write_bytes(r.content)
        return "→ tts_1_plain.mp3 (음색 다듬기 없이 — tts_1.mp3 와 비교)"
        
    def dialogue():
        # 다른 곳(ElevenLabs)에서 만든 대사 대신 위에서 만든 음성을 넣어 잇기만 확인한다
        clip = base64.b64encode((out / "tts_1.mp3").read_bytes()).decode()
        lines = [
            {"voice_id": voice["voice_id"], "text": "숲속에서 부엉이를 만났어요."},
            {"audio": clip},
            {"voice_id": voice["voice_id"], "text": "나도 반갑게 인사했어요."},
        ]
        r = http.post("/internal/v1/dialogue", json={"lines": lines, "speed": 1.0})
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code} {r.text[:200]}")
        (out / "dialogue.mp3").write_bytes(r.content)
        return f"→ dialogue.mp3 (서버 처리 {r.headers.get('x-processing-seconds')}초)"

    def delete():
        r = http.delete(f"/internal/v1/voices/{voice['voice_id']}")
        r.raise_for_status()
        r = http.post("/internal/v1/tts", json={"voice_id": voice["voice_id"], "text": "안녕"})
        if r.status_code != 404:
            raise RuntimeError(f"지운 뒤에도 읽힘: HTTP {r.status_code}")
        return "지운 뒤 404 확인"

    step("health (콜드 스타트 포함일 수 있음)", health)
    if step("목소리 등록 (실제 Qwen 준비 포함)", register):
        for i, text in enumerate(TEXTS, 1):
            step(f"읽기 {i} ({'콜드 직후' if i == 1 else '따뜻한 상태'})", speak(i, text))
        if (out / "tts_1.mp3").exists():
            step("비교용: 음색 다듬기 없이", plain)
            step("대사 잇기 (dialogue)", dialogue)
        if not args.keep_voice:
            step("목소리 지우기", delete)

    report = [
        "# drawtale-tts 실제 GPU 시험 결과",
        "",
        f"- 시각: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- 주소: {args.url}",
        f"- 참조 음성: {Path(args.ref).name}, ref_text {'있음' if args.ref_text else '없음'}",
        "- GPU 종류 · 비용: Modal 대시보드 (Apps → drawtale-tts → 로그의 `[engine] 모델 로드` 줄, Usage)",
        "",
        "| 단계 | 결과 | 걸린 시간 (요청 왕복) |",
        "|---|---|---|",
        *[f"| {a} | {b} | {c} |" for a, b, c in rows],
        "",
        "생성 음성: tts_1~3.mp3, dialogue.mp3 — 참조 음성과 직접 들어 비교해 유사도를 적는다.",
        "",
        "- 유사도 (사람이 듣고): ",
        "- 발음 · 끊김 · 잡음: ",
    ]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n결과: {out}/report.md")


if __name__ == "__main__":
    main()
