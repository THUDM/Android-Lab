#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/Working/Android-Lab-UpstreamBaseline

if [[ -x ".wsl-venv/bin/python" ]]; then
  PY=".wsl-venv/bin/python"
elif [[ -x "/mnt/c/Working/Android-Lab/.wsl-venv/bin/python" ]]; then
  PY="/mnt/c/Working/Android-Lab/.wsl-venv/bin/python"
else
  PY="python"
fi

task_id="clock_10"
models="qwen,gemma"
strategies="cot,k1,kmid,kmax"
timeout_s=900
parallel=1
suffix="$(date +%Y%m%d_%H%M%S)"
dry_run=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_strategy_smoke.sh [options]

Options:
  --task TASK_ID             AndroidLab task id. Default: clock_10
  --models qwen,gemma        Comma-separated: qwen, gemma. Default: qwen,gemma
  --strategies cot,k1,kmid,kmax
                             Comma-separated strategies. Default: cot,k1,kmid,kmax
  --timeout SECONDS          Per-run timeout. Default: 900
  --parallel N               AndroidLab parallel workers. Default: 1
  --suffix TEXT              Run-name suffix. Default: current timestamp
  --dry-run                  Print commands without executing
  -h, --help                 Show this help

Examples:
  bash scripts/run_strategy_smoke.sh --task clock_10 --models qwen --strategies k1,kmid
  bash scripts/run_strategy_smoke.sh --task clock_13 --strategies cot,kmax --timeout 1200
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      task_id="$2"; shift 2 ;;
    --models)
      models="$2"; shift 2 ;;
    --strategies)
      strategies="$2"; shift 2 ;;
    --timeout)
      timeout_s="$2"; shift 2 ;;
    --parallel)
      parallel="$2"; shift 2 ;;
    --suffix)
      suffix="$2"; shift 2 ;;
    --dry-run)
      dry_run=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

config_for() {
  local model="$1"
  local strategy="$2"
  case "${model}:${strategy}" in
    qwen:cot)  echo "configs/qwen35-4b-cot-generate-local.yaml" ;;
    qwen:k1)   echo "configs/qwen35-4b-rtv-k1-local.yaml" ;;
    qwen:kmid) echo "configs/qwen35-4b-rtv-k5-local.yaml" ;;
    qwen:kmax) echo "configs/qwen35-4b-rtv-kmax-local.yaml" ;;
    gemma:cot)  echo "configs/gemma4-e4b-it-cot-generate-local.yaml" ;;
    gemma:k1)   echo "configs/gemma4-e4b-it-rtv-k1-local.yaml" ;;
    gemma:kmid) echo "configs/gemma4-e4b-it-rtv-k3-local.yaml" ;;
    gemma:kmax) echo "configs/gemma4-e4b-it-rtv-kmax-local.yaml" ;;
    *)
      echo "Unsupported model/strategy: ${model}/${strategy}" >&2
      return 1 ;;
  esac
}

cleanup_docker() {
  docker ps --filter ancestor=android_eval:latest -q | xargs -r docker stop >/dev/null || true
}

mkdir -p logs/evaluation

IFS=',' read -r -a model_list <<< "${models}"
IFS=',' read -r -a strategy_list <<< "${strategies}"

for model in "${model_list[@]}"; do
  for strategy in "${strategy_list[@]}"; do
    config_path="$(config_for "${model}" "${strategy}")"
    run_name="smoke_${model}_${strategy}_${task_id}_${suffix}"
    log_path="logs/evaluation/${run_name}.log"

    echo "==> ${run_name}"
    echo "config=${config_path}"
    echo "log=${log_path}"

    cmd=("${PY}" eval.py -n "${run_name}" -c "${config_path}" --task_id "${task_id}" -p "${parallel}")
    if [[ "${dry_run}" == "1" ]]; then
      printf 'timeout %s ' "${timeout_s}"
      printf '%q ' "${cmd[@]}"
      printf '\n'
      continue
    fi

    if timeout "${timeout_s}s" "${cmd[@]}" > "${log_path}" 2>&1; then
      status="ok"
    else
      status="timeout_or_error"
    fi
    echo "status=${status}"
    tail -80 "${log_path}" || true
    cleanup_docker
  done
done
