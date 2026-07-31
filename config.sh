#!/usr/bin/env bash
# Shared defaults for artifact preparation and evaluation.

# --- How to activate the Python environment that has the engine installed. ---
# By default, use a repository-local virtual environment when it exists.
# Override KV2QUANT_VENV or KV2QUANT_CONDA_ENV for another installation.
KV2QUANT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KV2QUANT_VENV="${KV2QUANT_VENV:-${KV2QUANT_ROOT}/.venv}"
kv2q_activate_env() {
    if [[ -n "${KV2QUANT_VENV:-}" && -f "${KV2QUANT_VENV}/bin/activate" ]]; then
        # shellcheck disable=SC1090
        source "${KV2QUANT_VENV}/bin/activate"
        return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        conda activate "${KV2QUANT_CONDA_ENV:-optr}" 2>/dev/null || true
    fi
}

kv2q_configure_cuda() {
    local nvcc_path=""

    if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        :
    elif nvcc_path="$(command -v nvcc 2>/dev/null)" && [[ -n "${nvcc_path}" ]]; then
        export CUDA_HOME="$(cd "$(dirname "${nvcc_path}")/.." && pwd)"
    elif [[ -x "/usr/local/cuda/bin/nvcc" ]]; then
        export CUDA_HOME="/usr/local/cuda"
    else
        return 1
    fi

    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
}

kv2q_require_cuda() {
    if ! kv2q_configure_cuda; then
        cat >&2 <<'EOF'
CUDA Toolkit was not found. OptR requires nvcc in PATH or CUDA_HOME/bin/nvcc.
Install a CUDA 12.x toolkit (or use an NVIDIA CUDA devel container) and retry.
EOF
        return 1
    fi
}

KV2QUANT_SYS_GCC="${KV2QUANT_SYS_GCC:-}"
kv2q_configure_build_env() {
    kv2q_require_cuda || return 1

    if [[ -n "${KV2QUANT_SYS_GCC}" && -x "${KV2QUANT_SYS_GCC}/bin/g++" ]]; then
        unset \
            GCC_EXEC_PREFIX COMPILER_PATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH \
            LIBRARY_PATH CPATH 2>/dev/null || true
        export CC="${KV2QUANT_SYS_GCC}/bin/gcc"
        export CXX="${KV2QUANT_SYS_GCC}/bin/g++"
        export CUDAHOSTCXX="${CXX}"
        export NVCC_PREPEND_FLAGS="-ccbin ${CXX}"
    elif ! command -v g++ >/dev/null 2>&1; then
        echo "A C++ compiler was not found. Install build-essential and retry." >&2
        return 1
    fi
}

# --- Model + hardware --------------------------------------------------------
KV2QUANT_MODEL="${KV2QUANT_MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
KV2QUANT_MODEL_TAG="${KV2QUANT_MODEL_TAG:-qwen3-4B-thinking-2507}"   # artifacts/<tag>/
KV2QUANT_GPUS="${KV2QUANT_GPUS:-0}"          # CUDA_VISIBLE_DEVICES
# Ampere (A40/A100) must use the triton backend; Hopper (H100) may use fa3.
KV2QUANT_PREFILL_BACKEND="${KV2QUANT_PREFILL_BACKEND:-triton}"
KV2QUANT_DECODE_BACKEND="${KV2QUANT_DECODE_BACKEND:-triton}"

if [[ -f "${KV2QUANT_ROOT}/config.local.sh" ]]; then
    # shellcheck disable=SC1091
    source "${KV2QUANT_ROOT}/config.local.sh"
fi
