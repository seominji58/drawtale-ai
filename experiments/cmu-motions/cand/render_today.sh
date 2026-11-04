#!/bin/sh
# 11/4: 고른 15개 + 직접 만든 4개를 서비스와 같은 대응표(cmu1_front)로. 동시 2개
python make_handmade.py && python prep_front.py || exit 1
for f in work/*_front.yaml work/*_hand.yaml; do echo "$(basename $f .yaml) char1"; done \
  | xargs -P 2 -n 2 sh -c 'python render_one.py "$0" "$1" front >/dev/null 2>&1 || echo "FAIL $0"'
python sheet.py front 8; python sheet.py hand 8
