#!/usr/bin/env python3
"""Replay complete evaluation traces through the calibration dump server."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--io-log", required=True)
    p.add_argument("--num-traces", type=int, default=64)
    p.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Truncate each response to this many characters.",
    )
    p.add_argument("--num-threads", type=int, default=8)
    args = p.parse_args()

    from openai import OpenAI
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    client = OpenAI(
        base_url=args.base_url,
        api_key="EMPTY",
        timeout=600,
        max_retries=2,
    )

    rows = []
    with open(args.io_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            msgs = d.get("messages") or []
            resp = d.get("response") or ""
            if not msgs or not resp:
                continue
            if args.max_chars:
                resp = resp[: args.max_chars]
            prompt_text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            rows.append(prompt_text + resp)
            if len(rows) >= args.num_traces:
                break
    print(f"[dump_traces] sending {len(rows)} full traces for prefill-dump", flush=True)

    def _send(i_text):
        i, text = i_text
        try:
            client.completions.create(
                model=args.model,
                prompt=text,
                max_tokens=1,
                temperature=0.0,
            )
            return (i, True, None)
        except Exception as e:
            return (i, False, str(e)[:200])

    ok = 0
    with ThreadPoolExecutor(max_workers=args.num_threads) as ex:
        for i, good, err in ex.map(_send, enumerate(rows)):
            if good:
                ok += 1
            else:
                print(f"[dump_traces] trace {i} failed: {err}", flush=True)
    print(f"[dump_traces] done: {ok}/{len(rows)} traces prefilled", flush=True)


if __name__ == "__main__":
    main()
