#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

for command in python curl; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "${command} was not found. Install the README prerequisites and retry." >&2
        exit 1
    fi
done

python - <<'PY'
import sys
import sysconfig
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit(
        f"OptR requires Python 3.11 or newer; found {sys.version.split()[0]}"
    )

header = Path(sysconfig.get_paths()["include"]) / "Python.h"
if not header.is_file():
    raise SystemExit(
        f"Python development headers were not found at {header}. "
        "Install python3-dev and retry."
    )
PY

source "${ROOT}/config.sh"
kv2q_configure_build_env

if ! python -c 'import ctypes; ctypes.CDLL("libnuma.so.1")'; then
    echo "libnuma.so.1 was not found. Install libnuma1 and retry." >&2
    exit 1
fi

python -m pip install -r requirements.txt
python -m pip install --no-build-isolation \
    -e engine/sglang-research/python

# Install only the local MBPP+ evaluator without optional provider clients.
python -m pip install --no-deps evalplus==0.3.1

PYTHONPATH="${ROOT}/rotation/_disable_hf_kernels:${ROOT}/rotation/_triton_per_rank:${ROOT}/engine/sglang-research/python:${PYTHONPATH:-}" \
python - <<'PY'
from eval.data_registry import validate_manifest
from sglang.QuantKernel import oscar_rotation_clip_int2_kv  # noqa: F401
from sglang.srt.server_args import ServerArgs  # noqa: F401

validate_manifest()
PY

echo "OptR setup complete."
