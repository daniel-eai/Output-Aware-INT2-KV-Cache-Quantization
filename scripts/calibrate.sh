#!/usr/bin/env bash
# Calibrate OptR on top of a supplied OSCAR or QuaRot basis.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/config.sh"
kv2q_activate_env

MODEL="${KV2QUANT_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
MODEL_TAG="${KV2QUANT_MODEL_TAG:-qwen3-4B-thinking-2507}"
ART="${ART:-${ROOT}/artifacts/${MODEL_TAG}}"
DUMP_PATH="${DUMP_PATH:-${ROOT}/calibration/${MODEL_TAG}/rotation_prefill/qkv_dumps/gpqa}"
K_BASE="${K_BASE:-${ART}/k_rotation_qqt_r_h_pbr.pt}"
V_BASE="${V_BASE:-${ART}/v_rotation_sst_r_h_pbr.pt}"
K_MEAN="${K_MEAN:-${ART}/k_mean_decode.pt}"
OUT_DIR="${OUT_DIR:-${ART}/oscar_optr}"

for path in "${DUMP_PATH}" "${K_BASE}" "${V_BASE}" "${K_MEAN}"; do
  [[ -e "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 3; }
done

K_OUT="${OUT_DIR}/k_rotation.pt"
V_OUT="${OUT_DIR}/v_rotation.pt"
if [[ -f "${K_OUT}" && -f "${V_OUT}" && "${FORCE:-0}" != "1" ]]; then
  echo "[optr] reusing ${OUT_DIR}"
  exit 0
fi
if [[ -e "${OUT_DIR}" && "${FORCE:-0}" != "1" ]]; then
  echo "refusing to overwrite incomplete output: ${OUT_DIR}; set FORCE=1" >&2
  exit 4
fi
mkdir -p "${OUT_DIR}"

layer_args=()
[[ -n "${NUM_LAYERS:-}" ]] && layer_args=(--layers "${NUM_LAYERS}")

CUDA_VISIBLE_DEVICES="${CALIB_GPU:-0}" \
python "${ROOT}/optr/calibrate_optr.py" \
  --dump-path "${DUMP_PATH}" \
  --model-path "${MODEL}" \
  --k-oscar "${K_BASE}" \
  --v-oscar "${V_BASE}" \
  --k-mean "${K_MEAN}" \
  --k-out "${K_OUT}" \
  --v-out "${V_OUT}" \
  --group-size "${GROUP_SIZE:-128}" \
  --clip-k "${K_CLIP:-0.96}" \
  --clip-v "${V_CLIP:-0.92}" \
  --calib "${CALIB_CHUNKS:-2,8}" \
  --held "${HELDOUT_CHUNKS:-10,12}" \
  --prefix "${SINK_TOKENS:-64}" \
  --recent "${RECENT_TOKENS:-256}" \
  --qwin "${QUERY_WINDOW:-64}" \
  --steps "${STEPS:-80}" \
  --lr "${LR:-0.02}" \
  --lam "${LAMBDA_K:-1.0}" \
  --margin-k "${MARGIN_K:-0.0}" \
  --margin-v "${MARGIN_V:-0.0}" \
  "${layer_args[@]}"

validate_args=(
  --k-rotation "${K_OUT}"
  --v-rotation "${V_OUT}"
)
[[ -n "${NUM_LAYERS:-}" ]] && validate_args+=(--expected-layers "${NUM_LAYERS}")
python "${ROOT}/scripts/validate_rotation_artifact.py" "${validate_args[@]}"
