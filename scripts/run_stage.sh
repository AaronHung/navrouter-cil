#!/usr/bin/env bash
# 兩台共用的長時間執行包裝：開 tmux session，macOS 上自動加 caffeinate -ims，
# 每 15 分鐘寫一行心跳，結束時寫結束碼。失敗不重試。
#
#   NAVCIL_MACHINE=mac scripts/run_stage.sh <name> <command...>
#   例：NAVCIL_MACHINE=mac scripts/run_stage.sh 1a python scripts/run_1a.py
#
# log：outputs/navcil/${NAVCIL_MACHINE}/logs/<name>.log
# 看進度：tmux attach -t navcil-<name>   或   tail -f <log>
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: NAVCIL_MACHINE=<name> $0 <name> <command...>" >&2
  exit 2
fi
: "${NAVCIL_MACHINE:?請設定 NAVCIL_MACHINE}"

NAME="$1"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${REPO}/outputs/navcil/${NAVCIL_MACHINE}/logs"
LOG="${LOGDIR}/${NAME}.log"
SESSION="navcil-${NAME}"
mkdir -p "${LOGDIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} 已存在，停止。" >&2
  exit 1
fi

PREFIX=""
if [[ "$(uname -s)" == "Darwin" ]]; then
  PREFIX="caffeinate -ims"
fi

CMD="$(printf '%q ' "$@")"
INNER=$(cat <<EOF
cd $(printf '%q' "${REPO}")
export NAVCIL_MACHINE=$(printf '%q' "${NAVCIL_MACHINE}") PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
echo "[\$(date '+%F %T')] start: ${CMD}" >> $(printf '%q' "${LOG}")
( while sleep 900; do echo "[\$(date '+%F %T')] heartbeat" >> $(printf '%q' "${LOG}"); done ) &
HB=\$!
set +e
${PREFIX} ${CMD} >> $(printf '%q' "${LOG}") 2>&1
RC=\$?
kill \${HB} 2>/dev/null
echo "[\$(date '+%F %T')] exit=\${RC}" >> $(printf '%q' "${LOG}")
EOF
)

tmux new-session -d -s "${SESSION}" "bash -c $(printf '%q' "${INNER}")"
echo "started tmux session ${SESSION}; log: ${LOG}"
