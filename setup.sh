#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

source "${ROOT}/config.sh"
kv2q_configure_build_env

if ! python -c 'import ctypes; ctypes.CDLL("libnuma.so.1")'; then
    echo "libnuma.so.1 was not found. Install libnuma1 and retry." >&2
    exit 1
fi

python -m pip install -r requirements.txt
python -m pip install --no-build-isolation \
    -e engine/sglang-research/python

# EvalPlus declares optional model-provider clients as hard dependencies.
# OptR uses only its local MBPP+ oracle; its runtime dependencies are listed
# in requirements.txt.
python -m pip install --no-deps evalplus==0.3.1

echo "OptR setup complete."
