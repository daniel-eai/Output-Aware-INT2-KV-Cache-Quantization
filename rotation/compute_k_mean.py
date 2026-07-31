#!/usr/bin/env python3
"""Compute per-layer key means from decode calibration dumps."""
import argparse
import glob
import os

import torch

torch.set_grad_enabled(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dump",
        required=True,
        help="decode-regime post-RoPE dump root (layer_*/k/*.pt)",
    )
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = {
        "format_version": 1,
        "note": "decode-calibrated per-(layer,kvhead) post-RoPE key channel mean",
        "layers": {},
    }
    for li in range(args.layers):
        ld = f"{args.dump}/layer_{li}/k"
        cs = sorted(
            glob.glob(ld + "/*.pt"), key=lambda p: int(os.path.basename(p)[:-3])
        )
        cs = [
            p for p in cs if int(os.path.basename(p)[:-3]) != 0
        ]  # skip 6-token warmup
        if not cs:
            raise FileNotFoundError(f"no k chunks in {ld}")
        ssum, n = None, 0
        for p in cs:
            k = torch.load(p, map_location="cpu").float()  # [T, kv_heads, head_dim]
            ssum = k.sum(0) if ssum is None else ssum + k.sum(0)
            n += k.shape[0]
        mu = (ssum / n).contiguous().float()  # [kv_heads, head_dim]
        out["layers"][li] = {"layer_id": li, "mu": mu}
        if li in (0, 9, 18, 27, 35):
            print(
                f"layer {li}: kv_heads={mu.shape[0]} head0 mean-norm={mu[0].norm():.2f} "
                f"spread={float(mu[0].max() - mu[0].min()):.2f}"
            )
    torch.save(out, args.out)
    print(f"saved {args.out} ({len(out['layers'])} layers)")


if __name__ == "__main__":
    main()
