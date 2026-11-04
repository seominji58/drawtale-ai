import os
"""로컬 AI 서버(feat/forms)로 4족형 분석 → 걷기 렌더 (시험용). out/svc_q_{번호}.png 에 관절 · 걷기 6장면."""
import base64, io, json, sys, time, urllib.request, uuid
from PIL import Image, ImageDraw
BASE = os.environ.get("AI_BASE", "http://127.0.0.1:18001") + "/internal/v1"

def post(path, fields, files=None):
    b = uuid.uuid4().hex; body = b""
    for k, v in fields.items():
        body += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for k, (name, data) in (files or {}).items():
        body += f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{name}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode() + data + b"\r\n"
    body += f"--{b}--\r\n".encode()
    req = urllib.request.Request(BASE + path, body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    return json.load(urllib.request.urlopen(req, timeout=600))

form = sys.argv[1]
for n in sys.argv[2:]:
    path = f"real/{n}.up.jpg" if not n.endswith(".png") else n
    t = time.time()
    a = post("/analyze", {"form": form}, {"file": (path.split("/")[-1], open(path, "rb").read())})
    if not a["success"]:
        print(n, "분석 실패", a["message"]); continue
    line = f"{n} 분석 {time.time()-t:.1f}s form={a['form']} 찾음={a['joints_found']} 신뢰도={a['confidence']}"
    t = time.time()
    try:
        r = post("/render", {"request_id": a["request_id"], "joints": json.dumps(a["joints"]), "motion": "walk", "sprite": "true"})
        sp = Image.open(io.BytesIO(base64.b64decode(r["sprite"]))); c = sp.size[0] // 8
        fr = [sp.crop(((k % 8) * c, (k // 8) * c, (k % 8 + 1) * c, (k // 8 + 1) * c)).convert("RGBA") for k in range(0, r["sprite_frames"], max(1, r["sprite_frames"] // 6))][:6]
        line += f" | 렌더 {time.time()-t:.1f}s 프레임 {r['sprite_frames']}"
    except Exception as e:
        fr = []; line += f" | 렌더 실패 {e}"
    src = Image.open(path).convert("RGB"); d = ImageDraw.Draw(src); rr = max(5, src.width // 120)
    for j in a["joints"]:
        col = (0, 90, 255) if j["name"].endswith("_b") else (0, 170, 0) if j["name"].endswith("_f") else (255, 200, 0)
        d.ellipse((j["x"] - rr, j["y"] - rr, j["x"] + rr, j["y"] + rr), fill=col, outline="black")
    src.thumbnail((300, 300))
    strip = Image.new("RGB", (300 + 200 * 6, 300), "white"); strip.paste(src, (0, 0))
    for i, f in enumerate(fr):
        g = Image.new("RGBA", f.size, "white"); g.alpha_composite(f); strip.paste(g.convert("RGB").resize((200, 200)), (300 + i * 200, 50))
    strip.save(f"out/svc_{form}_{n.split('/')[-1].split('.')[0]}.png"); print(line, flush=True)
