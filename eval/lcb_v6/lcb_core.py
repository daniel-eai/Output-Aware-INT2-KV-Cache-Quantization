#!/usr/bin/env python3
"""LiveCodeBench v6 (release_v6 increment = test6.jsonl, 175 problems) harness core:
data loading + decoding, official-style prompt, code extraction, sandboxed
execution (stdin + functional), pass@1. Follows lcb_runner semantics.
"""
from __future__ import annotations

import base64
import io
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "eval"))

from data_registry import dataset_entry, materialize_dataset

LCB_FILE = "test6.jsonl"  # release_v6 increment (2025-01..04), the standard v6 window
DEFAULT_SANDBOX_IMAGE = (
    "python:3.11.9-slim-bookworm@"
    "sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317"
)


class SandboxUnavailable(RuntimeError):
    """Raised when fail-closed LiveCodeBench grading cannot start."""


class _RestrictedUnpickler(pickle.Unpickler):
    """Decode primitive LCB payloads without allowing global object loading."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(
            f"global object loading is forbidden: {module}.{name}"
        )

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("persistent pickle IDs are forbidden")


def _validate_test_payload(value, depth=0):
    if depth > 32:
        raise ValueError("private test payload is nested too deeply")
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate_test_payload(item, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("private test dictionaries require string keys")
            _validate_test_payload(item, depth + 1)
        return
    raise ValueError(f"unsupported private test payload type: {type(value).__name__}")


def _decode_private_tests(raw):
    if not isinstance(raw, str):
        _validate_test_payload(raw)
        return raw
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        compressed = base64.b64decode(raw.encode("utf-8"), validate=True)
        primitive = _RestrictedUnpickler(
            io.BytesIO(zlib.decompress(compressed))
        ).load()
        decoded = json.loads(primitive) if isinstance(primitive, str) else primitive
    _validate_test_payload(decoded)
    return decoded


def load_problems():
    p = materialize_dataset("lcb_v6")
    probs = []
    with p.open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            pub = (
                json.loads(r["public_test_cases"])
                if isinstance(r["public_test_cases"], str)
                else r["public_test_cases"]
            )
            priv = _decode_private_tests(r["private_test_cases"])
            tests = list(pub) + list(priv)
            probs.append(
                {
                    "qid": r["question_id"],
                    "question": r["question_content"],
                    "starter": r.get("starter_code", "") or "",
                    "tests": tests,
                    "fn_name": (
                        json.loads(r["metadata"]).get("func_name")
                        if r.get("metadata") and r["metadata"].strip()
                        else None
                    ),
                }
            )
    expected = int(dataset_entry("lcb_v6")["expected_examples"])
    if len(probs) != expected:
        raise RuntimeError(f"LCB v6 has {len(probs)} problems, expected {expected}")
    if len({problem["qid"] for problem in probs}) != expected:
        raise RuntimeError("LCB v6 question IDs are not unique")
    return probs


# ----------------------------- prompt (official LCB code_generation) -----------
SYS = (
    "You are an expert Python programmer. You will be given a question (problem "
    "specification) and will generate a correct Python program that matches the "
    "specification and passes all tests."
)


def make_prompt(prob):
    q = prob["question"]
    if prob["starter"].strip():
        fmt = (
            "You will use the following starter code to write the solution to the "
            "problem and enclose your code within delimiters.\n```python\n"
            + prob["starter"].strip()
            + "\n```"
        )
    else:
        fmt = (
            "Read the inputs from stdin solve the problem and write the answer to "
            "stdout (do not directly test on the sample inputs). Enclose your code "
            "within delimiters as follows. Ensure that when the python program runs, "
            "it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            "```python\n# YOUR CODE HERE\n```"
        )
    return f"### Question:\n{q}\n\n### Format: {fmt}\n\n### Answer: (use the provided format with backticks)\n"


def extract_code(response: str) -> str:
    # last ```python ... ``` block (fallback: last ``` ``` block)
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return response.strip()


# ----------------------------- execution (one subprocess per problem) ---------
_DRIVER = r"""
import sys, io, json, ast, signal, math, collections, heapq, bisect, itertools, functools, re
from typing import *

class _TO(Exception): pass
def _alarm(s, f): raise _TO()
signal.signal(signal.SIGALRM, _alarm)

USER = json.loads(sys.stdin.readline())
TESTS = json.loads(sys.stdin.readline())
FUNCTIONAL = json.loads(sys.stdin.readline())
FN = json.loads(sys.stdin.readline())
PERTO = 8

def _norm(s):
    return "\n".join(l.rstrip() for l in str(s).strip("\n").splitlines()).strip()

def _eq(a, b):
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(_eq(x,y) for x,y in zip(a,b))
    if isinstance(a,float) or isinstance(b,float):
        try: return abs(float(a)-float(b))<1e-6
        except Exception: return False
    return a==b

def _base_globals():
    # provide the common imports LeetCode/competitive solutions assume (typing.*, etc.)
    g=dict(globals()); g.pop("USER",None); g.pop("TESTS",None)
    return g

def run_stdin(inp):
    g=_base_globals(); g["__name__"]="__main__"
    oi,oo=sys.stdin,sys.stdout
    sys.stdin=io.StringIO(inp); buf=io.StringIO(); sys.stdout=buf
    try:
        exec(compile(USER,"<sol>","exec"), g)
    finally:
        sys.stdin,sys.stdout=oi,oo
    return buf.getvalue()

def check_stdin(t):
    out=run_stdin(t["input"])
    if _norm(out)==_norm(t["output"]): return True
    try:
        a=out.split(); b=t["output"].split()
        return len(a)==len(b) and all(abs(float(x)-float(y))<1e-6 for x,y in zip(a,b))
    except Exception: return False

if FUNCTIONAL:
    _g=_base_globals(); _g["__name__"]="sol"
    exec(compile(USER,"<sol>","exec"), _g)
    Solution=_g["Solution"]
    def check_func(t):
        args=[json.loads(x) for x in t["input"].split("\n") if x.strip()!=""]
        res=getattr(Solution(),FN)(*args)
        try: exp=json.loads(t["output"])
        except Exception:
            try: exp=ast.literal_eval(t["output"])
            except Exception: exp=t["output"]
        return _eq(res,exp)

ok=True
for t in TESTS:
    signal.alarm(PERTO)
    try:
        good = check_func(t) if FUNCTIONAL else check_stdin(t)
    except Exception:
        good=False
    finally:
        signal.alarm(0)
    if not good:
        ok=False; break
print("PASS" if ok else "FAIL")
"""


def sandbox_runtime() -> str:
    runtime = os.environ.get("OPTR_LCB_SANDBOX_RUNTIME", "docker").strip()
    if runtime not in {"docker", "podman"}:
        raise SandboxUnavailable(
            "OPTR_LCB_SANDBOX_RUNTIME must be 'docker' or 'podman'"
        )
    executable = shutil.which(runtime)
    if executable is None:
        raise SandboxUnavailable(
            f"{runtime} is required for LiveCodeBench grading but was not found"
        )
    return executable


def sandbox_image() -> str:
    return os.environ.get("OPTR_LCB_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)


def container_command(runtime: str, workdir: Path) -> list[str]:
    """Build the fail-closed container command used for generated code."""
    return [
        runtime,
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        "4g",
        "--memory-swap",
        "4g",
        "--cpus",
        "1",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "--user",
        "65534:65534",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--workdir",
        "/workspace",
        "--interactive",
        "--volume",
        f"{workdir}:/workspace:ro",
        sandbox_image(),
        "python",
        "-I",
        "-S",
        "/workspace/driver.py",
    ]


def ensure_sandbox_available() -> None:
    runtime = sandbox_runtime()
    result = subprocess.run(
        [runtime, "info"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = f": {detail[-1]}" if detail else ""
        raise SandboxUnavailable(f"cannot access the container runtime{suffix}")


def _run_sandboxed_driver(payload: str, timeout: int):
    runtime = sandbox_runtime()
    with tempfile.TemporaryDirectory(prefix="optr-lcb-") as temporary:
        workdir = Path(temporary)
        workdir.chmod(0o755)
        driver = workdir / "driver.py"
        driver.write_text(_DRIVER, encoding="utf-8")
        driver.chmod(0o644)
        return subprocess.run(
            container_command(runtime, workdir),
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )


def _fn_from_starter(starter):
    m = re.search(r"def\s+(\w+)\s*\(", starter)
    return m.group(1) if m else None


def eval_problem(code, prob, max_tests=None, timeout=60):
    """Return True iff code passes all tests inside the isolated container."""
    if not code.strip():
        return False
    tests = prob["tests"]
    if max_tests:
        tests = tests[:max_tests]
    functional = bool(prob["starter"].strip())
    fn = prob["fn_name"] or (_fn_from_starter(prob["starter"]) if functional else None)
    payload = (
        "\n".join(
            [
                json.dumps(code),
                json.dumps(tests),
                json.dumps(functional),
                json.dumps(fn),
            ]
        )
        + "\n"
    )
    try:
        result = _run_sandboxed_driver(payload, timeout)
    except subprocess.TimeoutExpired:
        return False
    output = result.stdout.strip()
    if not output:
        if result.returncode in {125, 126, 127}:
            detail = result.stderr.strip().splitlines()
            suffix = f": {detail[-1]}" if detail else ""
            raise RuntimeError(f"LiveCodeBench sandbox failed to start{suffix}")
        return False
    verdict = output.splitlines()[-1].strip()
    if verdict not in {"PASS", "FAIL"}:
        return False
    return verdict == "PASS"
