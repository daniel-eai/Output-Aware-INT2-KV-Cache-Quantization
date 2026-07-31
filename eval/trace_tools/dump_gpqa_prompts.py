#!/usr/bin/env python3
"""Send GPQA prompts through the calibration dump server."""

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
EVAL_DIR = REPO / "eval"
SE_DIR = REPO / "eval" / "third_party" / "simple_evals"
assert SE_DIR.is_dir(), f"missing {SE_DIR}"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(SE_DIR.parent))

from data_registry import dataset_entry, materialize_dataset

QUERY_TEMPLATE_MULTICHOICE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


def _build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--variant", default="diamond")
    p.add_argument("--num-prompts", type=int, default=198)
    p.add_argument("--num-threads", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p


def _build_prompts(num_prompts, variant, seed):
    import pandas

    if variant != "diamond":
        raise ValueError("only the pinned GPQA Diamond split is supported")
    df = pandas.read_csv(materialize_dataset("gpqa_diamond"))
    rows = [r.to_dict() for _, r in df.iterrows()]
    expected = int(dataset_entry("gpqa_diamond")["expected_examples"])
    if len(rows) != expected:
        raise RuntimeError(f"GPQA Diamond has {len(rows)} rows, expected {expected}")
    rng = random.Random(seed)
    if num_prompts < len(rows):
        rows = rng.sample(rows, num_prompts)
    out = []
    for r in rows:
        perm = rng.sample(range(4), 4)
        choices = [
            r["Correct Answer"],
            r["Incorrect Answer 1"],
            r["Incorrect Answer 2"],
            r["Incorrect Answer 3"],
        ]
        choices = [choices[i] for i in perm]
        body = {
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
            "Question": r["Question"],
        }
        out.append(QUERY_TEMPLATE_MULTICHOICE.format(**body))
    return out


def _send_one(client, model, prompt, temperature, top_p, top_k, max_tokens):
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body={"top_k": top_k},
        )
        return "ok"
    except Exception as e:
        return f"err: {e!r}"


def main():
    args = _build_argparser().parse_args()
    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    prompts = _build_prompts(args.num_prompts, args.variant, args.seed)
    print(
        f"[dump] sending {len(prompts)} GPQA-{args.variant} prompts "
        f"with max_tokens={args.max_tokens}",
        flush=True,
    )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        futs = [
            ex.submit(
                _send_one,
                client,
                args.model,
                prompt,
                args.temperature,
                args.top_p,
                args.top_k,
                args.max_tokens,
            )
            for prompt in prompts
        ]
        n_ok = n_err = 0
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r == "ok":
                n_ok += 1
            else:
                n_err += 1
                if n_err <= 5:
                    print(f"  prompt {i}: {r}", flush=True)
    print(
        f"[dump] done in {time.time()-t0:.1f}s ok={n_ok} err={n_err}",
        flush=True,
    )


if __name__ == "__main__":
    main()
