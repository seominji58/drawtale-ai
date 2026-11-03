#!/bin/sh
# 후보를 그림 하나에 렌더. 동시에 2개만 (11/3 10개 동시에 돌려 PC CPU 를 다 잡아먹었다)
for f in work/*_soft.yaml; do m=$(basename $f .yaml); [ -f out/char1_$m.gif ] || echo "$m char1"; done \
  | xargs -P 2 -n 2 sh -c 'python render_one.py "$0" "$1" >/dev/null 2>&1 || echo "FAIL $0 $1"'
ls out | wc -l
