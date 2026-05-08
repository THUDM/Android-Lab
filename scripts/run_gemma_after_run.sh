#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/mnt/c/Working/Android-Lab/.wsl-venv/bin/python}"
LOG_DIR="${ROOT_DIR}/logs/evaluation/full_runs"
WAIT_RUN_NAME=""
PARALLEL=2
POLL_SECONDS=60
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_gemma_after_run.sh --wait-run RUN_NAME [--parallel 2] [--poll-seconds 60] [--dry-run]

Waits for an existing AndroidLab eval run to finish, clears stale android_eval
containers, then runs the four Gemma configs sequentially:
  cot, rtv-k1, rtv-k3, rtv-kmax

Environment:
  PYTHON_BIN  Python executable to use. Defaults to /mnt/c/Working/Android-Lab/.wsl-venv/bin/python
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait-run)
      WAIT_RUN_NAME="${2:?missing value for --wait-run}"
      shift 2
      ;;
    -p|--parallel)
      PARALLEL="${2:?missing value for --parallel}"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="${2:?missing value for --poll-seconds}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${WAIT_RUN_NAME}" ]]; then
  echo "--wait-run is required." >&2
  usage >&2
  exit 2
fi

mkdir -p "${LOG_DIR}"

run_is_active() {
  pgrep -af "eval.py .*${WAIT_RUN_NAME}" >/dev/null 2>&1
}

stop_android_eval_containers() {
  local ids
  ids="$(docker ps --filter ancestor=android_eval:latest -q || true)"
  if [[ -n "${ids}" ]]; then
    echo "Stopping stale android_eval containers:"
    echo "${ids}"
    # shellcheck disable=SC2086
    docker stop ${ids} >/dev/null
  fi
}

run_eval() {
  local label="$1"
  local config="$2"
  local suffix run_name log_path
  suffix="$(date +%Y%m%d_%H%M%S)"
  run_name="gemma4_${label}_${suffix}"
  log_path="${LOG_DIR}/${run_name}.out.log"

  echo
  echo "=== ${label}: ${run_name} ==="
  echo "Config: ${config}"
  echo "Log: ${log_path}"
  echo "Command: PYTHONUNBUFFERED=1 ${PYTHON_BIN} -u eval.py -n ${run_name} -c ${config} -p ${PARALLEL}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  stop_android_eval_containers
  PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u eval.py \
    -n "${run_name}" \
    -c "${config}" \
    -p "${PARALLEL}" \
    > "${log_path}" 2>&1
  stop_android_eval_containers
}

cd "${ROOT_DIR}"

echo "Waiting for run to finish: ${WAIT_RUN_NAME}"
while run_is_active; do
  echo "$(date '+%Y-%m-%d %H:%M:%S') still active; checking again in ${POLL_SECONDS}s"
  if [[ "${DRY_RUN}" == "1" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

if [[ "${DRY_RUN}" != "1" ]]; then
  echo "Run finished or no longer active: ${WAIT_RUN_NAME}"
  stop_android_eval_containers
fi

run_eval "cot_generate" "configs/gemma4-e4b-it-cot-generate-local.yaml"
run_eval "rtv_k1" "configs/gemma4-e4b-it-rtv-k1-local.yaml"
run_eval "rtv_k3" "configs/gemma4-e4b-it-rtv-k3-local.yaml"
run_eval "rtv_kmax" "configs/gemma4-e4b-it-rtv-kmax-local.yaml"

echo
echo "Gemma queue complete."
