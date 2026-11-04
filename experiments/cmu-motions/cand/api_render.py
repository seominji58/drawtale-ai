"""로컬 AI 서버(운영과 같은 길: analyze → render sprite=true)로 동작을 렌더해 프레임 묶음을 저장한다 (시험용)."""
import base64, json, sys, time, urllib.request, uuid
d = json.load(open("out/analyze_char5.json", encoding="utf8"))
for motion in sys.argv[1:]:
    b = uuid.uuid4().hex
    fields = {"request_id": d["request_id"], "joints": json.dumps(d["joints"]), "motion": motion, "sprite": "true"}
    body = "".join(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n" for k, v in fields.items()) + f"--{b}--\r\n"
    req = urllib.request.Request("http://127.0.0.1:18001/internal/v1/render", body.encode(),
                                 {"Content-Type": f"multipart/form-data; boundary={b}"})
    t = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=300))
        open(f"out/api_{motion}.webp", "wb").write(base64.b64decode(r["sprite"]))
        print(f"{motion:18s} ok {time.time() - t:5.1f}s frames {r['sprite_frames']}", flush=True)
    except Exception as e:
        print(f"{motion:18s} FAIL {time.time() - t:5.1f}s {e}", flush=True)
