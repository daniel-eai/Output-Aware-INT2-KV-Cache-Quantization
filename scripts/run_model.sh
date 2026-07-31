#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${KV2QUANT_MODEL:?model wrapper must set KV2QUANT_MODEL}"
: "${KV2QUANT_MODEL_TAG:?model wrapper must set KV2QUANT_MODEL_TAG}"
: "${NUM_LAYERS:?model wrapper must set NUM_LAYERS}"

TASKS=(aime24 aime25 gpqa mbpp_plus lcb_v6)
MODES=(bf16 naive_int2 quarot quarot_optr oscar oscar_optr)

usage() {
    cat <<EOF
Usage: bash scripts/<model>.sh <command> [arguments]

Commands:
  prepare
      Build base, key-mean, identity, and Hadamard artifacts.

  calibrate <oscar|quarot|all>
      Build OptR artifacts from the selected initialization.

  eval <mode> <task>
      Run one benchmark.

  eval-all <mode>
      Run all supported benchmarks.

Modes:
  ${MODES[*]}

Tasks:
  ${TASKS[*]}
EOF
}

prepare_artifacts() {
    bash "${ROOT}/scripts/prepare.sh" all

    python "${ROOT}/scripts/make_fixed_rotations.py" \
        --output-dir "${ROOT}/artifacts/${KV2QUANT_MODEL_TAG}/fixed" \
        --num-layers "${NUM_LAYERS}" \
        --head-dim "${HEAD_DIM:-128}"

    mkdir -p "${ROOT}/artifacts/${KV2QUANT_MODEL_TAG}/quarot"
    cp \
        "${ROOT}/artifacts/${KV2QUANT_MODEL_TAG}/fixed/k_rotation_hadamard.pt" \
        "${ROOT}/artifacts/${KV2QUANT_MODEL_TAG}/quarot/k_rotation_hadamard.pt"
    cp \
        "${ROOT}/artifacts/${KV2QUANT_MODEL_TAG}/fixed/v_rotation_hadamard.pt" \
        "${ROOT}/artifacts/${KV2QUANT_MODEL_TAG}/quarot/v_rotation_hadamard.pt"
}

calibrate_oscar() {
    OUT_DIR="${ROOT}/artifacts/${KV2QUANT_MODEL_TAG}/oscar_optr" \
        bash "${ROOT}/scripts/calibrate.sh"
}

calibrate_quarot() {
    K_BASE="${ROOT}/artifacts/${KV2QUANT_MODEL_TAG}/quarot/k_rotation_hadamard.pt" \
    V_BASE="${ROOT}/artifacts/${KV2QUANT_MODEL_TAG}/quarot/v_rotation_hadamard.pt" \
    OUT_DIR="${ROOT}/artifacts/${KV2QUANT_MODEL_TAG}/quarot_optr" \
        bash "${ROOT}/scripts/calibrate.sh"
}

run_eval() {
    local mode="$1"
    local task="$2"
    bash "${ROOT}/run.sh" "${mode}" "${task}"
}

contains() {
    local candidate="$1"
    shift
    local value
    for value in "$@"; do
        [[ "${candidate}" == "${value}" ]] && return 0
    done
    return 1
}

command="${1:-help}"
case "${command}" in
    prepare)
        [[ "$#" -eq 1 ]] || { usage >&2; exit 2; }
        prepare_artifacts
        ;;
    calibrate)
        [[ "$#" -eq 2 ]] || { usage >&2; exit 2; }
        case "$2" in
            oscar) calibrate_oscar ;;
            quarot) calibrate_quarot ;;
            all)
                calibrate_oscar
                calibrate_quarot
                ;;
            *) usage >&2; exit 2 ;;
        esac
        ;;
    eval)
        [[ "$#" -eq 3 ]] || { usage >&2; exit 2; }
        contains "$2" "${MODES[@]}" || { usage >&2; exit 2; }
        contains "$3" "${TASKS[@]}" || { usage >&2; exit 2; }
        run_eval "$2" "$3"
        ;;
    eval-all)
        [[ "$#" -eq 2 ]] || { usage >&2; exit 2; }
        contains "$2" "${MODES[@]}" || { usage >&2; exit 2; }
        for task in "${TASKS[@]}"; do
            run_eval "$2" "${task}"
        done
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
