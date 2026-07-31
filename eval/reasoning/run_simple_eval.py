#!/usr/bin/env python3
"""Run the supported GPQA, AIME, and MBPP+ evaluations against an
OpenAI-compatible SGLang server.

Usage:
  python run_simple_eval.py \
    --task gpqa \
    --model Qwen/Qwen3-8B \
    --base-url http://127.0.0.1:31060/v1 \
    --max-tokens 32768 \
    --temperature 1.0 --top-p 0.95 --top-k 40 \
    --n-repeats 1 \
    --output-dir <dir>
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EVAL_DIR = REPO / "eval"
SE_DIR = REPO / "eval" / "third_party" / "simple_evals"
assert SE_DIR.is_dir(), (
    f"missing vendored simple_evals at {SE_DIR}"
)
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(SE_DIR.parent))

from data_registry import dataset_entry, materialize_dataset


def _build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task", required=True, choices=["gpqa", "aime24", "aime25", "mbpp_plus"]
    )
    p.add_argument("--model", required=True, help="HF model id served by sglang")
    p.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument(
        "--num-examples",
        type=int,
        default=None,
        help="Restrict to N examples (default: all)",
    )
    p.add_argument(
        "--variant",
        default="diamond",
        choices=["diamond"],
        help="Pinned GPQA variant",
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=32,
        help="Client-side concurrency cap. simple-evals defaults to "
        "os.cpu_count() which on big pods spikes the server "
        "above its CUDA-graph batch capture limit and turns "
        "CUDA graph off (eager → 2–3× slower).",
    )
    p.add_argument("--system-message", default="You are a helpful assistant.")
    p.add_argument("--output-dir", required=True)
    return p


class SglangChatSampler:
    """Pass top_p and top_k to the sglang OpenAI-compat endpoint
    (simple_evals' own ChatCompletionSampler only sends temperature)."""

    image_format = "url"

    def __init__(
        self,
        model,
        base_url,
        api_key,
        system_message,
        temperature,
        top_p,
        top_k,
        max_tokens,
    ):
        from openai import OpenAI

        # Long per-request timeout: 32K-token reasoning generations under batched
        # decode can take >>10min (the SDK default), which would time out and make
        # our retry loop re-send the SAME prompt -> duplicate server work + slowdown
        # (and eventual mis-scoring if all retries are exhausted). Disable the SDK's
        # own retries (max_retries=0) and let the outer loop handle real failures.
        import os as _os

        _timeout = float(_os.environ.get("SGLANG_EVAL_HTTP_TIMEOUT", "3600"))
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=_timeout,
            max_retries=0,
        )
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_tokens = max_tokens

    def _pack_message(self, role, content):
        return {"role": str(role), "content": content}

    def _handle_text(self, text):
        return {"type": "text", "text": text}

    def __call__(self, message_list):
        from simple_evals.types import SamplerResponse
        import openai

        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    extra_body={"top_k": self.top_k},
                )
                content = resp.choices[0].message.content
                if content is None:
                    raise ValueError("empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": resp.usage},
                    actual_queried_message_list=message_list,
                )
            except openai.BadRequestError as e:
                print("Bad Request:", e, flush=True)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                backoff = 2**trial
                print(f"  sampler retry {trial} in {backoff}s: {e}", flush=True)
                time.sleep(backoff)
                trial += 1
                if trial > 8:
                    raise


class _Result:
    """Minimal stand-in for simple_evals.EvalResult (score + metrics)."""

    def __init__(self, score, metrics):
        self.score = score
        self.metrics = metrics


def _extract_boxed(text: str):
    """Return the content of the LAST balanced \\boxed{...} in text, or None."""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None


def _last_int(s):
    import re

    if s is None:
        return None
    m = re.findall(r"-?\d+", s.replace(",", ""))
    if not m:
        return None
    tok = m[-1]
    # AIME answers are integers 0-999; degenerate quant can spam 1000s of digits
    # -> guard the int() string-conversion limit (4300) and treat as wrong.
    if len(tok.lstrip("-")) > 6:
        return None
    return int(tok)


def _grade_aime(text, ex):
    """Integer exact match (AIME answers are integers 0-999)."""
    boxed = _extract_boxed(text)
    pred = _last_int(boxed) if boxed is not None else _last_int(text)
    try:
        gold = int(str(ex["answer"]).strip())
    except (ValueError, TypeError):
        gold = _last_int(str(ex["answer"]))
    return pred is not None and gold is not None and pred == gold


class AnswerMatchEval:
    """Generic reasoning-task evaluator: prompt -> generate -> extract -> grade.

    Uses the same map_with_progress concurrency + io_log path as GPQAEval.
    Returns score = mean correctness, metrics = {chars: mean response length}.
    """

    def __init__(self, examples, build_prompt, grade, num_examples, n_repeats):
        if num_examples:
            examples = examples[:num_examples]
        self.examples = list(examples) * max(n_repeats, 1)
        self.build_prompt = build_prompt
        self.grade = grade

    def __call__(self, sampler):
        from simple_evals import common as _c

        # Keep generation threaded and grading deterministic in the main thread.
        def fn(ex):
            msgs = [sampler._pack_message("user", self.build_prompt(ex))]
            r = sampler(msgs)
            return {"txt": r.response_text or "", "ex": ex}

        gen = _c.map_with_progress(fn, self.examples)
        res = [
            {
                "correct": 1.0 if self.grade(x["txt"], x["ex"]) else 0.0,
                "chars": float(len(x["txt"])),
            }
            for x in gen
        ]
        n = max(len(res), 1)
        score = sum(x["correct"] for x in res) / n
        chars = sum(x["chars"] for x in res) / n
        return _Result(score, {"chars": chars})


def _load_pinned_aime(name):
    from datasets import load_dataset

    path = materialize_dataset(name)
    ds = load_dataset("parquet", data_files={"train": str(path)}, split="train")
    rows = [{"problem": r["problem"], "answer": r["answer"]} for r in ds]
    expected = int(dataset_entry(name)["expected_examples"])
    if len(rows) != expected:
        raise RuntimeError(f"{name} has {len(rows)} examples, expected {expected}")
    return rows


def _load_aime24():
    return _load_pinned_aime("aime24")


def _load_aime25():
    return _load_pinned_aime("aime25")


_AIME_PROMPT = (
    "{problem}\n\nPlease reason step by step, and put your final "
    "answer (an integer between 0 and 999) within \\boxed{{}}."
)


def _extract_code(text):
    """Robust code extraction for THINKING models. Qwen3-Thinking emits
    <think>...</think> then the FULL function WITHOUT a ```python fence, so
    the chain-of-thought to the executor. Drop the think block, prefer a fenced
    block, then fall back to the first import/def/class statement."""
    import re

    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    m = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return max(m, key=len)
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(("def ", "import ", "from ", "class ")):
            return "\n".join(lines[i:])
    return text


class MBPPPlusEval:
    """MBPP+ pass@1 using EvalPlus' official oracle executor."""

    def __init__(self, num_examples=None, n_repeats=1):
        import importlib.metadata
        import os

        entry = dataset_entry("mbpp_plus")
        installed = importlib.metadata.version("evalplus")
        expected_version = entry["evalplus_version"]
        if installed != expected_version:
            raise RuntimeError(
                f"EvalPlus {installed} is installed; expected {expected_version}"
            )
        os.environ["MBPP_OVERRIDE_PATH"] = str(materialize_dataset("mbpp_plus"))

        from evalplus.data import (
            get_mbpp_plus,
            get_mbpp_plus_hash,
        )
        from evalplus.evaluate import get_groundtruth
        from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS

        self.dataset = "mbpp"
        problems = get_mbpp_plus()
        hashcode = get_mbpp_plus_hash()
        if hashcode != entry["content_md5"]:
            raise RuntimeError(
                f"MBPP+ hash mismatch: expected {entry['content_md5']}, got {hashcode}"
            )
        expected_examples = int(entry["expected_examples"])
        if len(problems) != expected_examples:
            raise RuntimeError(
                f"MBPP+ has {len(problems)} examples, expected {expected_examples}"
            )
        self.problems = list(problems.values())
        if num_examples:
            self.problems = self.problems[:num_examples]
        self.problems = self.problems * max(n_repeats, 1)
        self.expected_output = get_groundtruth(
            problems, hashcode, MBPP_OUTPUT_NOT_NONE_TASKS
        )

    def __call__(self, sampler):
        from evalplus.eval import PASS
        from evalplus.evaluate import check_correctness
        from simple_evals import common

        instr = (
            "Read the following function signature and docstring, and fully "
            "implement the function described. Your response should only "
            "contain the code for this function.\n"
        )

        def fn(item):
            idx, problem = item
            msg = [
                sampler._pack_message(role="user", content=instr + problem["prompt"])
            ]
            resp = sampler(msg).response_text or ""
            code = _extract_code(resp)
            solution = problem["prompt"] + "\n" + code
            try:
                result = check_correctness(
                    self.dataset,
                    idx,
                    problem,
                    solution,
                    self.expected_output[problem["task_id"]],
                    base_only=False,
                    fast_check=True,
                    identifier=f"{problem['task_id']}::{idx}",
                )
                base_ok = result["base"][0] == PASS
                plus_ok = result["plus"][0] == PASS
                return {
                    "base": 1.0 if base_ok else 0.0,
                    "plus": 1.0 if plus_ok else 0.0,
                    "pass": 1.0 if (base_ok and plus_ok) else 0.0,
                }
            except Exception:
                return {"base": 0.0, "plus": 0.0, "pass": 0.0}

        rows = common.map_with_progress(
            fn, list(enumerate(self.problems)), num_threads=None
        )
        n = max(len(rows), 1)
        base = sum(r["base"] for r in rows) / n
        plus = sum(r["plus"] for r in rows) / n
        score = sum(r["pass"] for r in rows) / n
        return _Result(
            score, {"base_pass@1": base, "plus_pass@1": plus, "pass@1": score}
        )


def build_answer_match_evaluator(task, num_examples, n_repeats):
    if task == "aime24":
        return AnswerMatchEval(
            _load_aime24(),
            lambda ex: _AIME_PROMPT.format(problem=ex["problem"]),
            _grade_aime,
            num_examples,
            n_repeats,
        )
    if task == "aime25":
        return AnswerMatchEval(
            _load_aime25(),
            lambda ex: _AIME_PROMPT.format(problem=ex["problem"]),
            _grade_aime,
            num_examples,
            n_repeats,
        )
    raise ValueError(f"no answer-match evaluator for {task}")


def main():
    args = _build_argparser().parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sampler = SglangChatSampler(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        system_message=args.system_message,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    # Cap simple-evals' map_with_progress concurrency. GPQAEval.__call__
    # calls common.map_with_progress(fn, examples) without a num_threads arg,
    # which would otherwise default to os.cpu_count(). Monkey-patch the default
    # so the server stays at <= cuda-graph-max-bs concurrent requests.
    from simple_evals import common as _se_common

    _orig_map = _se_common.map_with_progress

    def _patched_map(f, xs, num_threads=None, pbar=True):
        if num_threads is None:
            num_threads = args.num_threads
        return _orig_map(f, xs, num_threads=num_threads, pbar=pbar)

    _se_common.map_with_progress = _patched_map

    # Monkey-patch ANSWER_PATTERN_MULTICHOICE back to the permissive `\s*`
    # (matches newlines), instead of openai's newer `[ \t]*` which fails on
    # "Answer:\n<letter>" outputs that thinking models do produce.
    import re

    _RELAXED = r"(?i)Answer\s*:\s*([A-D])"
    _se_common.ANSWER_PATTERN_MULTICHOICE = _RELAXED
    # gpqa_eval.py captured the symbol at import time, so patch the eval
    # module too if already imported (defensive — actually imported below).
    try:
        import simple_evals.gpqa_eval as _gpqa

        _gpqa.ANSWER_PATTERN_MULTICHOICE = _RELAXED
    except ImportError:
        pass

    # I/O dump: capture every (prompt, response) pair to io_log.jsonl so the
    # framework-vs-server contribution to noise can be checked offline.
    _io_log_path = out / "io_log.jsonl"
    _io_log_f = open(_io_log_path, "w")
    _orig_call = SglangChatSampler.__call__
    import threading

    _io_lock = threading.Lock()

    def _logging_call(self, message_list):
        resp = _orig_call(self, message_list)
        try:
            with _io_lock:
                import json as _json

                _io_log_f.write(
                    _json.dumps(
                        {
                            "messages": message_list,
                            "response": resp.response_text,
                            "model": self.model,
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "top_k": self.top_k,
                            "max_tokens": self.max_tokens,
                        }
                    )
                    + "\n"
                )
                _io_log_f.flush()
        except Exception:
            pass
        return resp

    SglangChatSampler.__call__ = _logging_call

    if args.task == "gpqa":
        from simple_evals.gpqa_eval import GPQAEval

        gpqa_entry = dataset_entry("gpqa_diamond")
        evaluator = GPQAEval(
            n_repeats=args.n_repeats,
            variant=args.variant,
            num_examples=args.num_examples,
            data_path=materialize_dataset("gpqa_diamond"),
            expected_examples=int(gpqa_entry["expected_examples"]),
        )
    elif args.task in ("aime24", "aime25"):
        evaluator = build_answer_match_evaluator(
            args.task, args.num_examples, args.n_repeats
        )
    elif args.task == "mbpp_plus":
        import os as _os, tempfile as _tf

        _localtmp = f"/tmp/evalplus_mbpp_{_os.getpid()}"
        _os.makedirs(_localtmp, exist_ok=True)
        _os.environ["TMPDIR"] = _localtmp
        _tf.tempdir = _localtmp
        evaluator = MBPPPlusEval(
            num_examples=args.num_examples, n_repeats=args.n_repeats
        )
    else:
        raise ValueError(f"task {args.task} not wired up yet")

    manifest_name = "gpqa_diamond" if args.task == "gpqa" else args.task
    (out / "data_provenance.json").write_text(
        json.dumps(
            {
                "manifest": str(REPO / "eval" / "data_manifest.json"),
                "dataset": manifest_name,
                "entry": dataset_entry(manifest_name),
            },
            indent=2,
        )
        + "\n"
    )

    print(f"=== running {args.task} eval ===", flush=True)
    print(f"  model={args.model}  base_url={args.base_url}")
    print(f"  n_repeats={args.n_repeats}  num_examples={args.num_examples}")
    print(f"  temperature={args.temperature} top_p={args.top_p} top_k={args.top_k}")
    print(f"  max_tokens={args.max_tokens}", flush=True)
    t0 = time.time()
    result = evaluator(sampler)
    elapsed = time.time() - t0

    # simple_evals.EvalResult: top-line `score` lives on the dataclass attribute,
    # NOT inside `metrics`. Merge them so downstream sees a single dict.
    metrics = dict(result.metrics or {})
    if getattr(result, "score", None) is not None:
        metrics["score"] = float(result.score)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Pretty score table; downstream consumers grep
    # `^|\\s+<task>/score\\s+\\|` to pull the final number from eval.log.
    lines = [
        f"Evaluation results for {args.task} on {args.model}",
        "=" * 100,
        "+" + "-" * 20 + "+" + "-" * 24 + "+",
        "|       Metric         |         Value          |",
        "+" + "-" * 20 + "+" + "-" * 24 + "+",
    ]
    for k in sorted(metrics.keys()):
        v = metrics[k]
        try:
            v_str = f"{float(v):.6f}"
        except (TypeError, ValueError):
            v_str = str(v)
        lines.append(f"|   {args.task}/{k:<14s} | {v_str:>22s} |")
    lines.append("+" + "-" * 20 + "+" + "-" * 24 + "+")
    lines.append(f"(elapsed: {elapsed:.1f}s)")
    log_text = "\n".join(lines) + "\n"
    (out / "eval.log").write_text(log_text)
    print(log_text)


if __name__ == "__main__":
    main()
