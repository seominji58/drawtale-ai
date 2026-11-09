"""시험 그림 모음 돌리기 — 분석 · 렌더를 고치기 전후로 같은 그림 전부를 돌려 비교한다 (2026-11-09).

하나를 고치면 다른 그림이 깨지는 일이 되풀이됐다(악어 꼬리를 살리니 기린에 종이가 들어옴, 촘촘한 그물로 렌더 10배 느려짐 …).
고친 뒤 배포 전에 이것을 돌려, 기준(baseline)과 달라진 그림만 골라 본다.

그림 폴더는 형태별 하위 폴더: {폴더}/{형태}/*.png|jpg (형태 = forms id: biped · quadruped · chain · winged · finned · radial).
아이 그림이 들어 있어 저장소에는 넣지 않는다 (지금은 quadruped-test/regression).

    python runtime/tools/regress.py IMAGES OUT --baseline     # 지금 결과를 기준으로 저장
    python runtime/tools/regress.py IMAGES OUT                # 기준과 비교 → OUT/보고.html · 달라진 것 목록

그림마다 분석 1번 + 동작(형태별 MOTIONS) 렌더. 재는 것:
  - 관절을 찾았나(joints_found), 마스크 넓이(그림 영역), sprite_box(렌더 칸 안 자리), 렌더 시간
  - 동작마다 장들을 겹친 「움직임 지도」 그림(OUT/{형태}/{이름}_{동작}.png) — 눈으로 전후를 비교
달라짐 판정: 관절 찾음 여부가 바뀜 · 마스크 넓이 ±8% · sprite_box 칸 0.04 넘게 · 렌더 시간 2배 넘게
AI 서버: 환경 변수 AI_URL (기본 http://127.0.0.1:8001)
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

BASE = os.environ.get("AI_URL", "http://127.0.0.1:8001") + "/internal/v1"
MOTIONS = {
    "biped": ["walk", "breathe", "wave_hello_gentle"],
    "quadruped": ["walk", "breathe", "search"],
    "chain": ["walk", "breathe", "shiver"],
    "winged": ["walk", "breathe"],
    "finned": ["walk", "breathe"],
    "radial": ["walk", "breathe"],
}
IMG = {".png", ".jpg", ".jpeg", ".webp"}


def post(path: str, fields: dict, files: dict | None = None) -> dict:
    b = uuid.uuid4().hex
    body = b""
    for k, v in fields.items():
        body += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for k, (fn, data) in (files or {}).items():
        body += (f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fn}\"\r\n"
                 "Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n"
    body += f"--{b}--\r\n".encode()
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.load(urllib.request.urlopen(req, timeout=900))


def overlay(sheet: Image.Image, cell: int, n: int) -> Image.Image:
    """장들을 반투명하게 겹친 움직임 지도 (흰 바탕). 첫 장은 진하게"""
    cols = sheet.width // cell
    acc = np.zeros((cell, cell), float)
    for i in range(n):
        f = np.asarray(sheet.crop(((i % cols) * cell, (i // cols) * cell, (i % cols + 1) * cell, (i // cols + 1) * cell)))[..., 3] / 255
        acc = np.maximum(acc, f * 0.35)
    first = np.asarray(sheet.crop((0, 0, cell, cell))).astype(float)
    a = first[..., 3:4] / 255
    rgb = np.ones((cell, cell, 3)) * 255 * (1 - acc[..., None]) + np.array([90, 120, 220]) * acc[..., None]
    rgb = rgb * (1 - a) + first[..., :3] * a
    return Image.fromarray(rgb.clip(0, 255).astype(np.uint8)).resize((256, 256))


def run_one(path: Path, form: str, out: Path) -> dict:
    t = time.time()
    a = post("/analyze", {"form": form}, {"file": (path.name, path.read_bytes())})
    res = {"form": form, "analyze_s": round(time.time() - t, 1), "ok": bool(a.get("success")),
           "joints_found": a.get("joints_found"), "motions": {}}
    if not a.get("success"):
        res["error"] = a.get("message")
        return res
    # 마스크 넓이 (원본 대비) — 세션 잘라 낸 그림이 없으면 건너뜀
    for m in MOTIONS.get(form, ["walk"]):
        t = time.time()
        r = post("/render", {"request_id": a["request_id"], "joints": json.dumps(a["joints"]), "motion": m, "sprite": "true"})
        sec = round(time.time() - t, 1)
        if "sprite" not in r:
            res["motions"][m] = {"error": r.get("message"), "s": sec}
            continue
        sheet = Image.open(io.BytesIO(base64.b64decode(r["sprite"]))).convert("RGBA")
        cell, n = r["sprite_cell"], r["sprite_frames"]
        first = np.asarray(sheet.crop((0, 0, cell, cell)))[..., 3] > 30
        overlay(sheet, cell, n).save(out / f"{path.stem}_{m}.png")
        res["motions"][m] = {"s": sec, "box": r.get("sprite_box"), "area": round(float(first.mean()), 4)}
    return res


def changed(base: dict, now: dict) -> list[str]:
    why = []
    if base.get("ok") != now.get("ok"):
        why.append(f"분석 {'성공' if base.get('ok') else '실패'} → {'성공' if now.get('ok') else '실패'}")
    if base.get("joints_found") != now.get("joints_found"):
        why.append(f"관절 찾음 {base.get('joints_found')} → {now.get('joints_found')}")
    for m, b in base.get("motions", {}).items():
        n = now.get("motions", {}).get(m)
        if not n:
            continue
        if ("error" in b) != ("error" in n):
            why.append(f"{m} 렌더 {'실패' if 'error' in n else '성공'}")
            continue
        if "error" in n:
            continue
        if b["area"] and abs(n["area"] / b["area"] - 1) > 0.08:
            why.append(f"{m} 그림 넓이 {b['area']:.3f} → {n['area']:.3f}")
        if b.get("box") and n.get("box") and max(abs(x - y) for x, y in zip(b["box"], n["box"])) > 0.04:
            why.append(f"{m} 칸 안 자리 {b['box']} → {n['box']}")
        if n["s"] > 2 * max(b["s"], 2.0):
            why.append(f"{m} 렌더 {b['s']}초 → {n['s']}초")
    return why


def main() -> None:
    images, out = Path(sys.argv[1]), Path(sys.argv[2])
    save_base = "--baseline" in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for form_dir in sorted(p for p in images.iterdir() if p.is_dir()):
        form = form_dir.name
        (out / form).mkdir(exist_ok=True)
        for img in sorted(p for p in form_dir.iterdir() if p.suffix.lower() in IMG):
            key = f"{form}/{img.stem}"
            try:
                results[key] = run_one(img, form, out / form)
            except Exception as e:  # noqa: BLE001
                results[key] = {"form": form, "ok": False, "error": str(e)}
            print(key, "관절" if results[key].get("joints_found") else "기본뼈대",
                  {m: v.get("s") for m, v in results[key].get("motions", {}).items()}, flush=True)
    base_path = images / "baseline.json"
    if save_base:
        base_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"기준 저장: {base_path} ({len(results)}장)")
        return
    base = json.loads(base_path.read_text(encoding="utf-8")) if base_path.exists() else {}
    diffs = {k: changed(base[k], v) for k, v in results.items() if k in base}
    diffs = {k: v for k, v in diffs.items() if v}
    (out / "결과.json").write_text(json.dumps({"results": results, "changed": diffs}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{len(results)}장 중 달라진 것 {len(diffs)}장")
    for k, v in diffs.items():
        print(" ", k, "·", " / ".join(v))


if __name__ == "__main__":
    main()
