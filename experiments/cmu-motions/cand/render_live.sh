#!/bin/sh
# 순하게(soft)로는 움직임이 거의 안 보인 것만 생생하게(live)로 다시. 동시 2개
for m in happy scared shrug yawn wait chicken_dance hop push_swing; do echo "${m}_live char1"; done \
  | xargs -P 2 -n 2 sh -c 'python render_one.py "$0" "$1" >/dev/null 2>&1 || echo "FAIL $0"'
