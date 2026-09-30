#!/usr/bin/env bash
# NC-15 的長時間執行包裝：與 run_stage.sh 相同，但 macOS 上改用 caffeinate -dimsu，
# 且每 15 分鐘寫一行進度（PROGRESS_DIR 下 .done 標記數）。失敗不重試。
#
#   NAVCIL_MACHINE=mac PROGRESS_DIR=<dir> scripts/nc15_run.sh <name> <command...>
#
# log：outputs/navcil/${NAVCIL_MACHINE}/logs/<name>.log
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: NAVCIL_MACHINE=<name> [PROGRESS_DIR=<dir>] $0 <name> <command...>" >&2
  exit 2
fi
: "${NAVCIL_MACHINE:?請設定 NAVCIL_MACHINE}"

NAME="$1"; shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${REPO}/outputs/navcil/${NAVCIL_MACHINE}/logs"
LOG="${LOGDIR}/${NAME}.log"
SESSION="navcil-${NAME}"
PDIR="${PROGRESS_DIR:-${REPO}/outputs/navcil/${NAVCIL_MACHINE}/nc15}"
mkdir -p "${LOGDIR}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session ${SESSION} 已存在，停止。" >&2
  exit 1
fi

PREFIX=""
if [[ "$(uname -s)" == "Darwin" ]]; then
  PREFIX="caffeinate -dimsu"
fi

CMD="$(printf '%q ' "$@")"
INNER=$(cat <<EOF
cd $(printf '%q' "${REPO}")
export NAVCIL_MACHINE=$(printf '%q' "${NAVCIL_MACHINE}") PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
echo "[\$(date '+%F %T')] start: ${CMD}" >> $(printf '%q' "${LOG}")
( while sleep 900; do echo "[\$(date '+%F %T')] heartbeat progress: \$(find $(printf '%q' "${PDIR}") -name '*.done' 2>/dev/null | wc -l | tr -d ' ') .done" >> $(printf '%q' "${LOG}"); done ) &
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
