#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export KV2QUANT_MODEL="Qwen/Qwen3-4B-Thinking-2507"
export KV2QUANT_MODEL_TAG="qwen3-4B-thinking-2507"
export NUM_LAYERS="36"
export TP_SIZE="${TP_SIZE:-8}"
export KV2QUANT_GPUS="${KV2QUANT_GPUS:-0,1,2,3,4,5,6,7}"

exec bash "${ROOT}/scripts/run_model.sh" "$@"
