#!/usr/bin/env bash
# Prepare the base rotation and key-mean artifacts.
#
# Usage:
#   bash scripts/prepare.sh all
set -euo pipefail

STEP="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_KV2QUANT_MODEL="${KV2QUANT_MODEL:-}"
USER_KV2QUANT_MODEL_TAG="${KV2QUANT_MODEL_TAG:-}"
USER_TP_SIZE="${TP_SIZE:-}"
USER_KV2QUANT_GPUS="${KV2QUANT_GPUS:-}"
USER_KV2QUANT_PREFILL_BACKEND="${KV2QUANT_PREFILL_BACKEND:-}"
USER_KV2QUANT_DECODE_BACKEND="${KV2QUANT_DECODE_BACKEND:-}"
source "${ROOT}/config.sh"

export KV2QUANT_MODEL="${USER_KV2QUANT_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
export KV2QUANT_MODEL_TAG="${USER_KV2QUANT_MODEL_TAG:-qwen3-4B-thinking-2507}"
export TP_SIZE="${USER_TP_SIZE:-8}"
export KV2QUANT_GPUS="${USER_KV2QUANT_GPUS:-0,1,2,3,4,5,6,7}"
export KV2QUANT_PREFILL_BACKEND="${USER_KV2QUANT_PREFILL_BACKEND:-triton}"
export KV2QUANT_DECODE_BACKEND="${USER_KV2QUANT_DECODE_BACKEND:-triton}"

kv2q_activate_env
kv2q_configure_build_env

export HF_HOME="${HF_HOME:-${ROOT}/caches/hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT}/caches/hf/datasets}"
export TORCH_HOME="${TORCH_HOME:-${ROOT}/caches/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/caches/xdg}"
LOCAL_CACHE_ROOT="${KV2QUANT_LOCAL_CACHE_ROOT:-/tmp/optr_${USER:-$(id -un)}}"
export TMPDIR="${TMPDIR:-${LOCAL_CACHE_ROOT}/tmp}"
export TMP="${TMPDIR}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}" "${TMPDIR}"

export PYTHONUNBUFFERED=1

MODEL="${KV2QUANT_MODEL}"
MODEL_TAG="${KV2QUANT_MODEL_TAG}"
ART="${ART:-${ROOT}/artifacts/${MODEL_TAG}}"
CALIB_ROOT="${CALIB_ROOT:-${ROOT}/calibration/${MODEL_TAG}}"
TRACE_RUN_DIR="${TRACE_RUN_DIR:-${CALIB_ROOT}/bf16_gpqa_trace}"
ROT_DUMP_DIR="${ROT_DUMP_DIR:-${CALIB_ROOT}/rotation_prefill/qkv_dumps/gpqa}"
MEAN_DUMP_DIR="${MEAN_DUMP_DIR:-${CALIB_ROOT}/mean_decode/qkv_dumps/gpqa}"
DUMP_ENGINE="${DUMP_ENGINE:-${ROOT}/engine/sglang-research}"

HEAD_DIM="${HEAD_DIM:-128}"
NUM_LAYERS="${NUM_LAYERS:-36}"
GROUP_SIZE="${GROUP_SIZE:-128}"
ROT_DUMP_TOKENS="${ROT_DUMP_TOKENS:-30000}"
MEAN_DUMP_TOKENS="${MEAN_DUMP_TOKENS:-60000}"
TRACE_NUM_EXAMPLES="${TRACE_NUM_EXAMPLES:-28}"
TRACE_MAX_NEW_TOKENS="${TRACE_MAX_NEW_TOKENS:-32768}"
TRACE_SEEDS="${TRACE_SEEDS:-1}"
TRACE_IO_LOG="${TRACE_IO_LOG:-${TRACE_RUN_DIR}/seed1/io_log.jsonl}"
TRACE_MAX_CHARS="${TRACE_MAX_CHARS:-}"
TRACE_NUM_THREADS="${TRACE_NUM_THREADS:-8}"
DUMP_TP_SIZE="${DUMP_TP_SIZE:-${TP_SIZE}}"
DUMP_GPUS="${DUMP_GPUS:-${KV2QUANT_GPUS}}"
DUMP_MEM_FRAC="${DUMP_MEM_FRAC:-0.8}"
DUMP_MAX_RUNNING="${DUMP_MAX_RUNNING:-32}"

mkdir -p "${ART}" "${CALIB_ROOT}"

log() { echo "[optr-artifacts $(date '+%F %T')] $*"; }

cleanup_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        pkill -TERM -P "${SERVER_PID}" 2>/dev/null || true
        sleep 2
        kill -KILL "${SERVER_PID}" 2>/dev/null || true
        pkill -KILL -P "${SERVER_PID}" 2>/dev/null || true
    fi
    SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

start_dump_server() {
    local dump_dir="$1"
    local dump_tokens="$2"
    local port="$3"
    local dist_port="$4"
    local log_file="$5"

    if [[ ! -d "${DUMP_ENGINE}/python/sglang" ]]; then
        echo "missing dump-enabled sglang at ${DUMP_ENGINE}; set DUMP_ENGINE=/path/to/sglang-dump-qkv" >&2
        exit 2
    fi
    mkdir -p "${dump_dir}" "$(dirname "${log_file}")"
    : > "${log_file}"

    local dump_pythonpath="${ROOT}/rotation/_disable_hf_kernels:${ROOT}/rotation/_dump_compat:${DUMP_ENGINE}/python:${PYTHONPATH:-}"
    local server_args=(
        --model-path "${MODEL}"
        --tensor-parallel-size "${DUMP_TP_SIZE}"
        --max-running-requests "${DUMP_MAX_RUNNING}"
        --max-queued-requests "${DUMP_MAX_QUEUED_REQUESTS:-64}"
        --page-size 128
        --chunked-prefill-size 4096
        --mem-fraction-static "${DUMP_MEM_FRAC}"
        --pp-max-micro-batch-size 32
        --kv-cache-dtype auto
        --prefill-attention-backend triton
        --decode-attention-backend triton
        --sampling-backend flashinfer
        --host 127.0.0.1
        --port "${port}"
        --dist-init-addr "127.0.0.1:${dist_port}"
        --trust-remote-code
        --disable-custom-all-reduce
        --disable-cuda-graph
        --watchdog-timeout 1800
    )
    if [[ -n "${DUMP_EXTRA_SERVER_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        server_args+=(${DUMP_EXTRA_SERVER_ARGS})
    fi

    log "starting dump server model=${MODEL} tp=${DUMP_TP_SIZE} gpus=${DUMP_GPUS} dump=${dump_dir} tokens=${dump_tokens}"
    DUMP_KVCACHE=true \
    DUMP_KVCACHE_TOKENS="${dump_tokens}" \
    DUMP_KVCACHE_DIR="${dump_dir}" \
    PYTHONPATH="${dump_pythonpath}" \
    CUDA_VISIBLE_DEVICES="${DUMP_GPUS}" \
        python -m sglang.launch_server "${server_args[@]}" >> "${log_file}" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 360); do
        if [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:${port}/health" 2>/dev/null || echo 000)" == 200 ]]; then
            log "dump server ready"
            return
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "dump server died; tail follows" >&2
            tail -80 "${log_file}" >&2 || true
            exit 1
        fi
        sleep 5
    done
    echo "dump server timeout; tail follows" >&2
    tail -80 "${log_file}" >&2 || true
    exit 1
}

run_trace() {
    log "running BF16 GPQA trace -> ${TRACE_RUN_DIR}"
    SEEDS="${TRACE_SEEDS}" \
    NUM_EXAMPLES="${TRACE_NUM_EXAMPLES}" \
    MAX_NEW_TOKENS="${TRACE_MAX_NEW_TOKENS}" \
    RUN_DIR_OVERRIDE="${TRACE_RUN_DIR}" \
    KV2QUANT_MODEL="${MODEL}" \
    KV2QUANT_MODEL_TAG="${MODEL_TAG}" \
    TP_SIZE="${TP_SIZE}" \
    KV2QUANT_GPUS="${KV2QUANT_GPUS}" \
        bash "${ROOT}/run.sh" bf16 gpqa
    [[ -f "${TRACE_IO_LOG}" ]] || { echo "missing trace io log ${TRACE_IO_LOG}" >&2; exit 1; }
}

run_rotation_dump() {
    local port="${ROT_DUMP_PORT:-31170}"
    local dist_port="${ROT_DUMP_DIST_PORT:-41170}"
    start_dump_server "${ROT_DUMP_DIR}" "${ROT_DUMP_TOKENS}" "${port}" "${dist_port}" "${CALIB_ROOT}/rotation_prefill/server.log"
    python "${ROOT}/eval/trace_tools/dump_gpqa_prompts.py" \
        --model "${MODEL}" \
        --base-url "http://127.0.0.1:${port}/v1" \
        --num-prompts "${ROT_NUM_PROMPTS:-198}" \
        --num-threads "${ROT_NUM_THREADS:-32}" \
        --temperature 0.6 --top-p 0.95 --top-k 20 \
        --max-tokens 1 \
        2>&1 | tee "${CALIB_ROOT}/rotation_prefill/dump_runner.log"
    cleanup_server
}

run_rotation() {
    log "computing OSCAR rotation from ${ROT_DUMP_DIR}"
    python "${ROOT}/rotation/compute_kv_rotation.py" \
        --dump-path "${ROT_DUMP_DIR}" \
        --output-dir "${ART}" \
        --head-dim "${HEAD_DIM}" \
        --chunk-id "${CHUNK_ID:-all}"
}

run_mean_dump() {
    [[ -f "${TRACE_IO_LOG}" ]] || { echo "missing trace io log ${TRACE_IO_LOG}; run trace first" >&2; exit 1; }
    local port="${MEAN_DUMP_PORT:-31171}"
    local dist_port="${MEAN_DUMP_DIST_PORT:-41171}"
    start_dump_server "${MEAN_DUMP_DIR}" "${MEAN_DUMP_TOKENS}" "${port}" "${dist_port}" "${CALIB_ROOT}/mean_decode/server.log"
    extra=()
    if [[ -n "${TRACE_MAX_CHARS}" ]]; then
        extra+=(--max-chars "${TRACE_MAX_CHARS}")
    fi
    python "${ROOT}/eval/trace_tools/dump_traces.py" \
        --model "${MODEL}" \
        --base-url "http://127.0.0.1:${port}/v1" \
        --io-log "${TRACE_IO_LOG}" \
        --num-traces "${TRACE_NUM_EXAMPLES}" \
        --num-threads "${TRACE_NUM_THREADS}" \
        "${extra[@]}" \
        2>&1 | tee "${CALIB_ROOT}/mean_decode/dump_runner.log"
    cleanup_server
}

run_mean() {
    log "computing K-Mean from ${MEAN_DUMP_DIR}"
    python "${ROOT}/rotation/compute_k_mean.py" \
        --dump "${MEAN_DUMP_DIR}" \
        --layers "${NUM_LAYERS}" \
        --out "${ART}/k_mean_decode.pt"
}

validate_artifacts() {
    python - "${ART}" "${NUM_LAYERS}" "${HEAD_DIM}" <<'PY'
import sys
from pathlib import Path
import torch

art = Path(sys.argv[1])
num_layers = int(sys.argv[2])
head_dim = int(sys.argv[3])
required = [
    "k_rotation_qqt_r_h_pbr.pt",
    "v_rotation_sst_r_h_pbr.pt",
    "k_mean_decode.pt",
]
for name in required:
    path = art / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
mean = torch.load(art / "k_mean_decode.pt", map_location="cpu")
if len(mean["layers"]) != num_layers:
    raise SystemExit(f"k_mean layers={len(mean['layers'])}, expected {num_layers}")
for layer_id, entry in mean["layers"].items():
    mu = entry["mu"]
    if mu.ndim != 2 or mu.shape[1] != head_dim:
        raise SystemExit(f"bad mu shape at layer {layer_id}: {tuple(mu.shape)}")
print(f"validated artifacts in {art}")
PY
}

case "${STEP}" in
    trace) run_trace ;;
    rotation-dump) run_rotation_dump ;;
    rotation) run_rotation ;;
    mean-dump) run_mean_dump ;;
    mean) run_mean ;;
    validate) validate_artifacts ;;
    all)
        run_trace
        run_rotation_dump
        run_rotation
        run_mean_dump
        run_mean
        validate_artifacts
        ;;
    *)
        echo "unknown step: ${STEP}" >&2
        exit 2
        ;;
esac
