#!/bin/sh
# 손뼉 · 갸웃만 다시 (11/4). 동시 2개
python make_handmade.py >/dev/null || exit 1
printf "clap_overhead_hand char1\nhead_tilt_hand char1\n" | xargs -P 2 -n 2 sh -c 'python render_one.py "$0" "$1" front >/dev/null 2>&1 || echo "FAIL $0"'
python sheet.py hand 8
