#!/bin/sh
# 阶段 A 长训监控器：每 5 分钟记录进程、迭代、loss、评测和 checkpoint 状态。
# 用法：sh tools/monitor_stageA.sh
set -u

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
RUN="stageA_9x9_50k"
RUN_DIR="$ROOT/rl/runs/$RUN"
METRICS="$RUN_DIR/metrics.jsonl"
LOG="$RUN_DIR/monitor.log"
PATTERN="training/train.py --name $RUN"

mkdir -p "$RUN_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] monitor started; run=$RUN" >> "$LOG"

while :; do
  now=$(date '+%Y-%m-%d %H:%M:%S')
  alive=0
  if pgrep -f "$PATTERN" >/dev/null 2>&1; then alive=1; fi

  summary=$(python3 - "$METRICS" <<'PY'
import json, os, sys
path = sys.argv[1]
rows = []
try:
    with open(path, encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
except FileNotFoundError:
    pass
if not rows:
    print('iter=0/210 samples=0 buffer=0 policy=n/a value=n/a eval=n/a')
else:
    r = rows[-1]
    ev = r.get('eval') or {}
    print('iter=%s/210 games=%s positions=%s buffer=%s policy=%.4f value=%.4f eval=%s' % (
        r.get('iter', '?'), r.get('games', 0) * r.get('iter', 0),
        sum(x.get('samples', 0) for x in rows), r.get('buffer', 0),
        r.get('policy_loss', 0), r.get('value_loss', 0),
        json.dumps(ev, ensure_ascii=False, separators=(',', ':')) if ev else 'pending'))
PY
)

  ckpt='missing'
  if [ -f "$RUN_DIR/latest.pt" ]; then
    ckpt=$(stat -f '%Sm' -t '%Y-%m-%d %H:%M:%S' "$RUN_DIR/latest.pt" 2>/dev/null || echo present)
  fi
  echo "[$now] alive=$alive checkpoint=$ckpt $summary" | tee -a "$LOG"

  iter=$(printf '%s\n' "$summary" | sed -n 's/.*iter=\([0-9][0-9]*\)\/210.*/\1/p')
  iter=${iter:-0}
  if [ "$iter" -ge 210 ] && [ "$alive" -eq 0 ]; then
    echo "[$now] COMPLETE: reached iter 210 and process exited" | tee -a "$LOG"
    exit 0
  fi
  if [ "$alive" -eq 0 ] && [ "$iter" -gt 0 ] && [ "$iter" -lt 210 ]; then
    echo "[$now] WARNING: process exited early at iter $iter" | tee -a "$LOG"
    exit 2
  fi
  sleep 300
done
