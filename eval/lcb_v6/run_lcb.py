#!/usr/bin/env python3
"""LiveCodeBench v6 runner for the unified evaluation entry point.
Queries the sglang server (OpenAI API), extracts code, grades pass@1 via lcb_core,
writes <output-dir>/eval.log ("lcb_v6/score  <p>") and io_log.jsonl.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lcb_core as L


def generate(prob, args):
    import openai

    client = openai.OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
        timeout=args.request_timeout,
        max_retries=0,
    )
    msgs = [
        {"role": "system", "content": L.SYS},
        {"role": "user", "content": L.make_prompt(prob)},
    ]
    for trial in range(9):
        try:
            kwargs = {}
            if args.seed is not None:
                kwargs["seed"] = args.seed
            resp = client.chat.completions.create(
                model=args.model,
                messages=msgs,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                top_p=args.top_p,
                extra_body={"top_k": args.top_k},
                **kwargs,
            )
            c = resp.choices[0].message.content
            if c is None:
                raise ValueError("empty")
            return c
        except Exception as e:
            if trial >= 8:
                return f"__ERROR__ {e}"
            time.sleep(2**trial)
    return "__ERROR__"


def _grade_one(args_tuple):
    code, prob = args_tuple
    return L.eval_problem(code, prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="lcb_v6")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--request-timeout", type=float, default=7200.0)
    ap.add_argument("--n-repeats", type=int, default=1)
    ap.add_argument("--num-threads", type=int, default=16)
    ap.add_argument(
        "--grade-workers",
        type=int,
        default=4,
        help="Number of isolated grading containers to run concurrently",
    )
    ap.add_argument("--num-examples", type=int, default=None)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    L.ensure_sandbox_available()
    probs = L.load_problems()
    if args.num_examples:
        probs = probs[: args.num_examples]
    n = len(probs)
    print(
        f"=== running lcb_v6 eval ===\n  model={args.model} base_url={args.base_url}\n"
        f"  n_problems={n} max_tokens={args.max_tokens} T={args.temperature} "
        f"top_p={args.top_p} top_k={args.top_k} seed={args.seed} "
        f"request_timeout={args.request_timeout}",
        flush=True,
    )

    # 1) generate concurrently
    responses = [None] * n
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        futs = {ex.submit(generate, probs[i], args): i for i in range(n)}
        done = 0
        for fut in as_completed(futs):
            i = futs[fut]
            responses[i] = fut.result()
            done += 1
            if done % 20 == 0 or done == n:
                print(f"  generated {done}/{n}  [{time.time()-t0:.0f}s]", flush=True)

    codes = [L.extract_code(r) for r in responses]

    # 2) grade (process pool; each eval is one subprocess internally too -> modest workers)
    passed = [False] * n
    with ThreadPoolExecutor(max_workers=max(1, args.grade_workers)) as ex:
        futs = {ex.submit(_grade_one, (codes[i], probs[i])): i for i in range(n)}
        for fut in as_completed(futs):
            passed[futs[fut]] = fut.result()

    score = sum(passed) / n
    provenance = {
        "dataset": "lcb_v6",
        "entry": L.dataset_entry("lcb_v6"),
        "sandbox_runtime": Path(L.sandbox_runtime()).name,
        "sandbox_image": L.sandbox_image(),
    }
    with open(os.path.join(args.output_dir, "data_provenance.json"), "w") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")
    # io log
    with open(os.path.join(args.output_dir, "io_log.jsonl"), "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "qid": probs[i]["qid"],
                        "passed": bool(passed[i]),
                        "code": codes[i],
                        "response": responses[i],
                        "response_len": len(responses[i] or ""),
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "top_k": args.top_k,
                        "seed": args.seed,
                        "max_tokens": args.max_tokens,
                        "request_timeout": args.request_timeout,
                    }
                )
                + "\n"
            )
    # Machine-readable summary consumed by run.sh.
    with open(os.path.join(args.output_dir, "eval.log"), "w") as f:
        f.write(f"Evaluation results for lcb_v6 on {args.model}\n")
        f.write(f"lcb_v6/score   {score:.6f}\n")
        f.write(f"lcb_v6/n   {n}\n")
        f.write(f"lcb_v6/n_pass   {sum(passed)}\n")
    print(
        f"lcb_v6/score   {score:.6f}  ({sum(passed)}/{n})  [{time.time()-t0:.0f}s]",
        flush=True,
    )


if __name__ == "__main__":
    main()
