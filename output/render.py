"""AI 서버에 분석·렌더링을 요청해 output 폴더에 MP4를 저장한다.

사용법:
    python output/render.py <이미지경로> <motion> [관절JSON경로]

관절 JSON을 주면 그 관절로 렌더링하고, 없으면 분석 결과를 그대로 쓴다.
"""
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

AI_URL = 'http://127.0.0.1:8001'
OUT_DIR = Path(__file__).parent


def post(url: str, fields: dict, files: dict | None = None):
    boundary = uuid.uuid4().hex
    body = b''
    for name, value in (files or {}).items():
        path = Path(value)
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; '
            f'name="{name}"; filename="{path.name}"\r\n'
            f'Content-Type: application/octet-stream\r\n\r\n'
        ).encode()
        body += path.read_bytes() + b'\r\n'
    for name, value in fields.items():
        body += (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        ).encode()
    body += f'--{boundary}--\r\n'.encode()

    req = urllib.request.Request(
        url, data=body,
        headers={'Content-Type': f'multipart/form-data; boundary={boundary}'},
    )
    with urllib.request.urlopen(req, timeout=600) as res:
        return res.headers.get('content-type', ''), res.read()


def main() -> None:
    image_path = sys.argv[1]
    motion = sys.argv[2] if len(sys.argv) > 2 else 'wave_hello'
    joints_path = sys.argv[3] if len(sys.argv) > 3 else None

    print('분석 중...')
    ctype, data = post(f'{AI_URL}/internal/v1/analyze', {}, {'file': image_path})
    result = json.loads(data)
    if not result.get('success'):
        print('분석 실패:', result.get('message'))
        return

    request_id = result['request_id']
    print(f"분석 완료: 관절 {len(result['joints'])}개, {result['processing_time_ms']}ms")

    # 원본과 마스크, 관절을 output 폴더에 남겨 둔다
    (OUT_DIR / 'joints.json').write_text(json.dumps(result['joints']))

    joints = (
        Path(joints_path).read_text() if joints_path else json.dumps(result['joints'])
    )

    print(f'렌더링 중 ({motion})...')
    start = time.time()
    ctype, data = post(f'{AI_URL}/internal/v1/render', {
        'request_id': request_id,
        'joints': joints,
        'motion': motion,
    })

    if ctype.startswith('application/json'):
        print('렌더링 실패:', data.decode()[:120])
        return

    suffix = '_fitted' if joints_path else ''
    out = OUT_DIR / f'{motion}{suffix}.mp4'
    out.write_bytes(data)
    print(f'저장: {out}  ({len(data):,} bytes, {time.time() - start:.0f}초)')


if __name__ == '__main__':
    main()
