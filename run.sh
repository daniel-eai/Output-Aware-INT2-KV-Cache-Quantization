#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT}/config.sh"

usage() {
    echo "usage: bash run.sh <mode> <task>" >&2
    echo "modes: bf16 naive_int2 quarot quarot_optr oscar oscar_optr" >&2
    echo "tasks: aime24 aime25 gpqa mbpp_plus lcb_v6" >&2
}

MODE="${1:-}"
TASK="${2:-}"
case "${MODE}" in
    bf16|naive_int2|quarot|quarot_optr|oscar|oscar_optr) ;;
    *) usage; exit 2 ;;
esac
case "${TASK}" in
    aime24|aime25|gpqa|mbpp_plus|lcb_v6) ;;
    *) usage; exit 2 ;;
esac

kv2q_activate_env
kv2q_configure_build_env

LOCAL_CACHE_ROOT="${KV2QUANT_LOCAL_CACHE_ROOT:-/tmp/optr_${USER:-$(id -un)}}"
export HF_HOME="${HF_HOME:-${ROOT}/caches/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TORCH_HOME="${TORCH_HOME:-${HF_HOME}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LOCAL_CACHE_ROOT}/xdg}"
export TMPDIR="${TMPDIR:-${LOCAL_CACHE_ROOT}/tmp}"
export TMP="${TMPDIR}"
mkdir -p \
    "${HF_HOME}" \
    "${HF_DATASETS_CACHE}" \
    "${TORCH_HOME}" \
    "${XDG_CACHE_HOME}" \
    "${TMPDIR}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}/rotation/_disable_hf_kernels:${ROOT}/rotation/_triton_per_rank:${ROOT}/engine/sglang-research/python:${PYTHONPATH:-}"

MODEL="${KV2QUANT_MODEL}"
MODEL_TAG="${KV2QUANT_MODEL_TAG}"
ART="${ROOT}/artifacts/${MODEL_TAG}"
TP_SIZE="${TP_SIZE:-1}"
GROUP_SIZE="${GROUP_SIZE:-128}"
PAGE_SIZE="${PAGE_SIZE:-8}"
K_CLIP="${K_CLIP:-0.96}"
V_CLIP="${V_CLIP:-0.92}"
PREFIX_TOKENS="${PREFIX_TOKENS:-64}"
RECENT_TOKENS="${RECENT_TOKENS:-256}"
HP_PREFIX_POOL_TOKENS="${HP_PREFIX_POOL_TOKENS:-8192}"

PORT="${SGLANG_PORT:-31077}"
DIST_PORT="${SGLANG_DIST_PORT:-41077}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
NUM_EXAMPLES="${NUM_EXAMPLES:-}"
NUM_WORKERS="${NUM_WORKERS:-16}"
SEEDS="${SEEDS:-5}"
SEED_START="${SEED_START:-1}"
SEED_BASE="${SEED_BASE:-1000}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT}/results}"

if [[ -z "${CONTEXT_LENGTH:-}" ]]; then
    case "${MODEL}" in
        *Phi-4*) CONTEXT_LENGTH=32768 ;;
        *) CONTEXT_LENGTH=40960 ;;
    esac
fi

if [[ "${MODE}" == "bf16" ]]; then
    QUANTIZED=0
    MEM_FRAC="${MEM_FRAC:-0.85}"
    MAX_RUNNING="${MAX_RUNNING:-8}"
    CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
else
    QUANTIZED=1
    MEM_FRAC="${MEM_FRAC:-0.70}"
    MAX_RUNNING="${MAX_RUNNING:-16}"
    CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-16}"
fi

K_ROT=""
V_ROT=""
K_MEAN=""
case "${MODE}" in
    bf16) ;;
    naive_int2)
        K_ROT="fixed/k_rotation_identity.pt"
        V_ROT="fixed/v_rotation_identity.pt"
        ;;
    quarot)
        K_ROT="quarot/k_rotation_hadamard.pt"
        V_ROT="quarot/v_rotation_hadamard.pt"
        ;;
    quarot_optr)
        K_ROT="quarot_optr/k_rotation.pt"
        V_ROT="quarot_optr/v_rotation.pt"
        K_MEAN="${ART}/k_mean_decode.pt"
        ;;
    oscar)
        K_ROT="k_rotation_qqt_r_h_pbr.pt"
        V_ROT="v_rotation_sst_r_h_pbr.pt"
        ;;
    oscar_optr)
        K_ROT="oscar_optr/k_rotation.pt"
        V_ROT="oscar_optr/v_rotation.pt"
        K_MEAN="${ART}/k_mean_decode.pt"
        ;;
esac

artifact_path() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        artifacts/*) printf '%s/%s\n' "${ROOT}" "$1" ;;
        *) printf '%s/%s\n' "${ART}" "$1" ;;
    esac
}

if [[ "${QUANTIZED}" == "1" ]]; then
    K_ROT_PATH="$(artifact_path "${K_ROT_OVERRIDE:-${K_ROT}}")"
    V_ROT_PATH="$(artifact_path "${V_ROT_OVERRIDE:-${V_ROT}}")"
    if [[ -n "${K_MEAN_OVERRIDE:-}" ]]; then
        K_MEAN="$(artifact_path "${K_MEAN_OVERRIDE}")"
    fi
    [[ -f "${K_ROT_PATH}" ]] || { echo "missing rotation: ${K_ROT_PATH}" >&2; exit 3; }
    [[ -f "${V_ROT_PATH}" ]] || { echo "missing rotation: ${V_ROT_PATH}" >&2; exit 3; }
    if [[ -n "${K_MEAN}" ]]; then
        [[ -f "${K_MEAN}" ]] || { echo "missing key mean: ${K_MEAN}" >&2; exit 3; }
    fi
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
EXAMPLE_TAG="${NUM_EXAMPLES:-all}"
RUN_DIR="${RUN_DIR_OVERRIDE:-${RESULTS_ROOT}/${TASK}/${MODEL_TAG}/${MODE}/seeds${SEEDS}_ex${EXAMPLE_TAG}_t${MAX_NEW_TOKENS}_${STAMP}}"
mkdir -p "${RUN_DIR}"

RUN_CACHE_NAME="${TASK}_${MODEL_TAG}_${MODE}_p${PORT}_${STAMP}"
TRITON_CACHE_ROOT="${TRITON_CACHE_ROOT:-${LOCAL_CACHE_ROOT}/triton}"
export OSCAR_TRITON_PER_RANK_BASE="${OSCAR_TRITON_PER_RANK_BASE:-${TRITON_CACHE_ROOT}/${RUN_CACHE_NAME}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${OSCAR_TRITON_PER_RANK_BASE}/main}"
mkdir -p "${OSCAR_TRITON_PER_RANK_BASE}" "${TRITON_CACHE_DIR}"

echo "[optr] mode=${MODE} task=${TASK} model=${MODEL}"
echo "[optr] seeds=${SEED_START}..${SEEDS} output=${RUN_DIR}"

SERVER_ARGS=(
    --model-path "${MODEL}"
    --tensor-parallel-size "${TP_SIZE}"
    --prefill-attention-backend "${KV2QUANT_PREFILL_BACKEND}"
    --decode-attention-backend "${KV2QUANT_DECODE_BACKEND}"
    --mem-fraction-static "${MEM_FRAC}"
    --max-running-requests "${MAX_RUNNING}"
    --host 127.0.0.1
    --port "${PORT}"
    --dist-init-addr "127.0.0.1:${DIST_PORT}"
    --trust-remote-code
    --context-length "${CONTEXT_LENGTH}"
)

if [[ "${EXTRA_SERVER_ARGS:-}" != *disable-cuda-graph* ]]; then
    SERVER_ARGS+=(--cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}")
fi
if [[ -n "${EXTRA_SERVER_ARGS:-}" ]]; then
    read -r -a EXTRA_SERVER_ARRAY <<< "${EXTRA_SERVER_ARGS}"
    SERVER_ARGS+=("${EXTRA_SERVER_ARRAY[@]}")
fi

PREFIX_ENV=(
    env
    -u SGLANG_PORT
    -u SGLANG_DIST_PORT
    -u SGLANG_OSCAR_K_ROTATION_PATH
    -u SGLANG_OSCAR_V_ROTATION_PATH
    -u SGLANG_OSCAR_K_MEAN_PATH
    -u SGLANG_OSCAR_K_MEAN_IN_POOL
    -u SGLANG_OSCAR_ABSORB_V_ROTATION
    -u SGLANG_ENABLE_MIXED_KV_WINDOWS
    CUDA_VISIBLE_DEVICES="${KV2QUANT_GPUS}"
)

if [[ "${QUANTIZED}" == "1" ]]; then
    SERVER_ARGS+=(
        --kv-cache-dtype int2
        --kv-cache-quant-group-size "${GROUP_SIZE}"
        --page-size "${PAGE_SIZE}"
        --enable-cache-report
    )
    K_MEAN_ACTIVE=0
    [[ -n "${K_MEAN}" ]] && K_MEAN_ACTIVE=1
    PREFIX_ENV+=(
        SGLANG_ENABLE_MIXED_KV_WINDOWS=1
        SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
        SGLANG_OSCAR_ABSORB_V_ROTATION=1
        SGLANG_MIXED_KV_HP_MAX_SPLITS=8
        SGLANG_MIXED_KV_PREFIX_TOKENS="${PREFIX_TOKENS}"
        SGLANG_MIXED_KV_RECENT_TOKENS="${RECENT_TOKENS}"
        SGLANG_MIXED_KV_HP_PREFIX_POOL_TOKENS="${HP_PREFIX_POOL_TOKENS}"
        SGLANG_MIXED_KV_HP_DTYPE=bfloat16
        SGLANG_MIXED_KV_SCALE_DTYPE=float32
        SGLANG_OSCAR_K_ROTATION_PATH="${K_ROT_PATH}"
        SGLANG_OSCAR_V_ROTATION_PATH="${V_ROT_PATH}"
        SGLANG_OSCAR_K_CLIP_RATIO="${K_CLIP}"
        SGLANG_OSCAR_V_CLIP_RATIO="${V_CLIP}"
        SGLANG_OSCAR_K_MEAN_IN_POOL="${K_MEAN_ACTIVE}"
        SGLANG_OSCAR_K_MEAN_PATH="${K_MEAN}"
    )
fi

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]]; then
        kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
        for _ in $(seq 1 10); do
            kill -0 "${SERVER_PID}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${SERVER_PID}" 2>/dev/null; then
            kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
        fi
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup EXIT INT TERM

server_is_ready() {
    python - "${PORT}" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(
        f"http://127.0.0.1:{sys.argv[1]}/health",
        timeout=2,
    ) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
}

start_server() {
    local log_file="$1"
    local rng="$2"
    "${PREFIX_ENV[@]}" setsid python -m sglang.launch_server \
        "${SERVER_ARGS[@]}" \
        --random-seed "${rng}" >> "${log_file}" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 360); do
        if server_is_ready; then
            echo "[optr] server ready rng=${rng}"
            return 0
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "server exited before becoming ready" >&2
            tail -80 "${log_file}" >&2 || true
            return 1
        fi
        sleep 5
    done
    echo "server health check timed out" >&2
    return 1
}

stop_server() {
    cleanup
    sleep 3
}

EXAMPLE_ARGS=()
if [[ -n "${NUM_EXAMPLES}" ]]; then
    EXAMPLE_ARGS=(--num-examples "${NUM_EXAMPLES}")
fi

if [[ "${TASK}" == "lcb_v6" ]]; then
    EVALUATOR="${ROOT}/eval/lcb_v6/run_lcb.py"
    PYTHONPATH="${ROOT}/eval/lcb_v6:${PYTHONPATH:-}" \
        python -c "import lcb_core; lcb_core.ensure_sandbox_available()"
else
    EVALUATOR="${ROOT}/eval/reasoning/run_simple_eval.py"
    if [[ "${TASK}" == "mbpp_plus" ]]; then
        echo "[optr] WARNING: MBPP+ executes generated code through EvalPlus." >&2
        echo "[optr] Run it only in a disposable isolated environment." >&2
    fi
fi

for seed in $(seq "${SEED_START}" "${SEEDS}"); do
    rng=$((SEED_BASE + seed))
    seed_dir="${RUN_DIR}/seed${seed}"
    mkdir -p "${seed_dir}"
    echo "[optr] seed=${seed} rng=${rng}"
    start_server "${seed_dir}/server.log" "${rng}"

    EVAL_ARGS=(
        --task "${TASK}"
        --model "${MODEL}"
        --base-url "http://127.0.0.1:${PORT}/v1"
        --max-tokens "${MAX_NEW_TOKENS}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --top-k "${TOP_K}"
        --n-repeats 1
        --num-threads "${NUM_WORKERS}"
        "${EXAMPLE_ARGS[@]}"
        --output-dir "${seed_dir}"
    )
    if [[ "${TASK}" == "lcb_v6" ]]; then
        EVAL_ARGS+=(--seed "${rng}")
    fi

    if ! python "${EVALUATOR}" "${EVAL_ARGS[@]}" 2>&1 | tee "${seed_dir}/runner.log"; then
        echo "evaluation failed at seed ${seed}" >&2
        stop_server
        exit 1
    fi
    stop_server

    score="$(
        { grep -iE "${TASK}/score" "${seed_dir}/eval.log" 2>/dev/null || true; } \
            | grep -oE '[0-9]+\.[0-9]+' \
            | head -1 \
            || true
    )"
    echo "seed${seed} rng=${rng} score=${score:-NA}" \
        | tee -a "${RUN_DIR}/seeds_scores.txt"
done

python - "${RUN_DIR}/seeds_scores.txt" "${RUN_DIR}/summary.txt" <<'PY'
from __future__ import annotations

import re
import statistics
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
scores = []
if source.exists():
    for line in source.read_text().splitlines():
        match = re.search(r"score=([0-9]*\.?[0-9]+)", line)
        if match:
            scores.append(float(match.group(1)))

if scores:
    mean = statistics.mean(scores)
    std = statistics.stdev(scores) if len(scores) > 1 else 0.0
    summary = (
        f"seeds={len(scores)}\n"
        f"scores={[f'{score:.6f}' for score in scores]}\n"
        f"mean={mean:.6f}\n"
        f"std={std:.6f}\n"
        f"table={mean * 100:.2f} +/- {std * 100:.2f}\n"
    )
else:
    summary = "no scores parsed\n"

target.write_text(summary)
print(summary, end="")
PY

echo "[optr] results=${RUN_DIR}"
