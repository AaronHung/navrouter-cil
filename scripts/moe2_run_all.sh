#!/usr/bin/env bash
# MOE-2：依 vec → E1 … E7 → E8 的順序跑完全部階段（PREREG-18 細則 15、25–26）。失敗不重試，接著跑不依賴它的階段。
#
#   tmux new-session -d -s moe2 "env NAVCIL_MACHINE=mac MOE2_PY=<python> caffeinate -dimsu scripts/moe2_run_all.sh"
#
# 環境變數：
#   MOE2_PY           python 直譯器（預設 python）
#   MOE2_OUT          outputs/navcil/<machine>/ 下的輸出子目錄（預設 moe2；冒煙測試用 moe2_smoke）
#   MOE2_FOLDS        預設 1-10（冒煙測試用 1）
#   MOE2_HEARTBEAT_S  心跳間隔秒數，預設 900
#
# 產物（都在 outputs/navcil/<machine>/<out>/）：
#   HEARTBEAT.log     每 15 分鐘一行：時間、階段、已完成／總數、失敗數；每階段開始與結束各一行
#   FAILED_<階段>.txt 失敗階段的 traceback
#   STAGE_TIMES.tsv   每階段的起訖時間、秒數、結束碼
#   logs/<階段>.log   每階段的 stdout／stderr
# 每個階段結束後都重新產生報告（scripts/moe2_report.py）。跑完不關機、不做任何 git 操作。
set -u

: "${NAVCIL_MACHINE:?請設定 NAVCIL_MACHINE}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}" || exit 2
export NAVCIL_MACHINE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

PY="${MOE2_PY:-python}"
OUT="${MOE2_OUT:-moe2}"
FOLDS="${MOE2_FOLDS:-1-10}"
HB_S="${MOE2_HEARTBEAT_S:-900}"
ROOT="${REPO}/outputs/navcil/${NAVCIL_MACHINE}/${OUT}"
mkdir -p "${ROOT}/logs" "${ROOT}/progress"
CUR="${ROOT}/progress/current_stage"
echo "init" > "${CUR}"

ts() { date '+%F %T'; }

beat() {
  local s p nf
  s="$(cat "${CUR}" 2>/dev/null || echo '?')"
  p="$(cat "${ROOT}/progress/${s}.txt" 2>/dev/null || echo '0 0')"
  nf="$(find "${ROOT}" -maxdepth 1 -name 'FAILED_*.txt' | wc -l | tr -d ' ')"
  echo "[$(ts)] stage=${s} done=${p% *}/${p#* } failed=${nf} $*" >> "${ROOT}/HEARTBEAT.log"
}

( while sleep "${HB_S}"; do beat heartbeat; done ) &
HB_PID=$!
trap 'kill ${HB_PID} 2>/dev/null' EXIT

report() {
  "${PY}" scripts/moe2_report.py --out "${OUT}" >> "${ROOT}/logs/report.log" 2>&1 \
    || echo "[$(ts)] moe2_report.py 失敗（見 logs/report.log）" >> "${ROOT}/HEARTBEAT.log"
}

# run_stage <階段>；回傳該階段的結束碼
run_stage() {
  local name="$1"
  local log="${ROOT}/logs/${name}.log" t0 t1 rc start
  echo "${name}" > "${CUR}"
  if [[ -f "${ROOT}/${name}.done" ]]; then
    beat "skip（已完成）"
    return 0
  fi
  start="$(ts)"; t0=$(date +%s)
  beat start
  echo "[${start}] start ${name} folds=${FOLDS}" >> "${log}"
  "${PY}" "scripts/moe2_${name}.py" --device cpu --folds "${FOLDS}" --out "${OUT}" >> "${log}" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "[$(ts)] exit=${rc}" >> "${log}"
  printf '%s\t%s\t%s\t%s\t%s\n' "${name}" "${start}" "$(ts)" "$((t1 - t0))" "${rc}" >> "${ROOT}/STAGE_TIMES.tsv"
  if [[ ${rc} -ne 0 && ! -f "${ROOT}/FAILED_${name}.txt" ]]; then
    { echo "[$(ts)] 階段 ${name} 結束碼 ${rc}（程式未寫出 traceback；以下為 log 末 80 行）"; tail -n 80 "${log}"; } > "${ROOT}/FAILED_${name}.txt"
  fi
  beat "end rc=${rc}"
  report
  return ${rc}
}

skip_stage() {
  local name="$1" why="$2"
  echo "${name}" > "${CUR}"
  echo "[$(ts)] 階段 ${name} 未執行：${why}" > "${ROOT}/FAILED_${name}.txt"
  printf '%s\t%s\t%s\t%s\t%s\n' "${name}" "$(ts)" "$(ts)" "0" "skipped" >> "${ROOT}/STAGE_TIMES.tsv"
  beat "${name} skipped（${why}）"
}

beat "run_all start folds=${FOLDS} py=${PY}"
run_stage vec; RCV=$?
for s in e1 e2 e3 e4 e5 e6 e7; do
  if [[ ${RCV} -eq 0 ]]; then
    run_stage "${s}"
  else
    skip_stage "${s}" "vec 失敗（E1–E7 依賴向量快取；PREREG-18 細則 15）"
  fi
done
run_stage e8
echo "all" > "${CUR}"
report
beat "run_all finished"
