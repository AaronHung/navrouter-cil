#!/usr/bin/env bash
# MOE-1：依 S1→S7 的順序跑完全部階段（PREREG-17）。過夜無人看管：失敗不重試，接著跑不依賴它的階段。
#
#   tmux new-session -d -s moe1 "env NAVCIL_MACHINE=mac MOE1_PY=<python> caffeinate -dimsu scripts/moe1_run_all.sh"
#
# 環境變數：
#   MOE1_PY           python 直譯器（預設 python）
#   MOE1_OUT          outputs/navcil/<machine>/ 下的輸出子目錄（預設 moe1；冒煙測試用 moe1_smoke）
#   MOE1_FOLDS        預設 1-10（冒煙測試用 1）
#   MOE1_EPOCHS       新訓練的 epoch 數，預設 5（冒煙測試用 1；K3 一律 5）
#   MOE1_HEARTBEAT_S  心跳間隔秒數，預設 900
#
# 產物（都在 outputs/navcil/<machine>/<out>/）：
#   HEARTBEAT.log     每 15 分鐘一行：時間、階段、已完成／總數、失敗數
#   FAILED_<階段>.txt 失敗階段的 traceback
#   STAGE_TIMES.tsv   每階段的起訖時間、秒數、結束碼
#   logs/<階段>.log   每階段的 stdout／stderr
# 每個階段結束後都重新產生報告（scripts/moe1_report.py）。跑完不關機、不做任何 git 操作。
set -u

: "${NAVCIL_MACHINE:?請設定 NAVCIL_MACHINE}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}" || exit 2
export NAVCIL_MACHINE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1

PY="${MOE1_PY:-python}"
OUT="${MOE1_OUT:-moe1}"
FOLDS="${MOE1_FOLDS:-1-10}"
EPOCHS="${MOE1_EPOCHS:-5}"
HB_S="${MOE1_HEARTBEAT_S:-900}"
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
  "${PY}" scripts/moe1_report.py --out "${OUT}" >> "${ROOT}/logs/report.log" 2>&1 \
    || echo "[$(ts)] moe1_report.py 失敗（見 logs/report.log）" >> "${ROOT}/HEARTBEAT.log"
}

# run_stage <階段> [額外參數...]；回傳該階段的結束碼
run_stage() {
  local name="$1"; shift
  local log="${ROOT}/logs/${name}.log" t0 t1 rc start
  echo "${name}" > "${CUR}"
  if [[ -f "${ROOT}/${name}.done" ]]; then
    beat "skip（已完成）"
    return 0
  fi
  start="$(ts)"; t0=$(date +%s)
  beat start
  echo "[${start}] start ${name} folds=${FOLDS} epochs=${EPOCHS}" >> "${log}"
  "${PY}" "scripts/moe1_${name}.py" --device cpu --folds "${FOLDS}" --out "${OUT}" --epochs "${EPOCHS}" "$@" >> "${log}" 2>&1
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

beat "run_all start folds=${FOLDS} epochs=${EPOCHS} py=${PY}"
run_stage s1
run_stage s2; RC2=$?
run_stage s3
run_stage s4
run_stage s5
if [[ ${RC2} -eq 0 ]]; then
  run_stage s6
else
  echo "s6" > "${CUR}"
  echo "[$(ts)] 階段 s6 未執行：S2 失敗（S6 依賴 S2 的 σ 固定版程式；PREREG-17 細則 9）" > "${ROOT}/FAILED_s6.txt"
  printf '%s\t%s\t%s\t%s\t%s\n' "s6" "$(ts)" "$(ts)" "0" "skipped" >> "${ROOT}/STAGE_TIMES.tsv"
  beat "s6 skipped（S2 失敗）"
  report
fi
# S7：S1–S6 都結束後才跑（seed 45／46 的部分只在 s4.done 存在時跑，由 moe1_s7.py 判斷）
run_stage s7
echo "all" > "${CUR}"
report
beat "run_all finished"
