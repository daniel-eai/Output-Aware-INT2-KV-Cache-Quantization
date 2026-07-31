#!/usr/bin/env python3
"""Calibrate OptR K/V rotation checkpoints."""
import argparse
import math

import torch

from optr_common import (
    DEV,
    WOCache,
    ensure_parent,
    head_layout,
    load_fp_sequences,
    load_headwise_rotation,
    load_layer_entry,
    load_layer_mu,
    load_model_geometry,
    parse_chunks,
    quant_int2_ste,
    summarize_ortho,
)
from kv_calib_utils import rotate_headwise, derotate_headwise


def k_eff_headwise(k, R_k, mu, clip_k, grp, prefix, recent):
    from kv_calib_utils import restore_exact_window

    kc = k - mu[None]
    kr = rotate_headwise(kc, R_k)
    kq = derotate_headwise(quant_int2_ste(kr, clip_k, grp), R_k)
    return restore_exact_window(kq, kc, prefix, recent), kc


def v_eff_headwise(v, R_v, clip_v, grp, prefix, recent):
    from kv_calib_utils import restore_exact_window

    vr = rotate_headwise(v, R_v)
    vq = derotate_headwise(quant_int2_ste(vr, clip_v, grp), R_v)
    return restore_exact_window(vq, v, prefix, recent)


def k_loss_by_head(seqs, R_k, mu, WO, clip_k, grp, prefix, recent, qwin, scale, lam):
    """Output-aware K loss per KV head: KL + lam * post-W_O attention-shift err."""
    kvh = R_k.shape[0]
    vals = [[] for _ in range(kvh)]
    for s in seqs:
        keff, kc = k_eff_headwise(s.k, R_k, mu, clip_k, grp, prefix, recent)
        L = keff.shape[0]
        g = head_layout(s.q, kvh)
        lo = max(1, L - qwin)
        m = torch.arange(L, device=DEV).unsqueeze(0) > torch.arange(
            lo, L, device=DEV
        ).unsqueeze(1)
        for h in range(kvh):
            kt = kc[:, h]
            ks = keff[:, h]
            vfp = s.v[:, h]
            hv = []
            for gi in range(g):
                j = h * g + gi
                qt = s.q[lo:L, j]
                Lt = ((qt @ kt.T) * scale).masked_fill(m, float("-inf"))
                Ls = ((qt @ ks.T) * scale).masked_fill(m, float("-inf"))
                pt = torch.softmax(Lt, -1)
                pq = torch.softmax(Ls, -1)
                kl = (
                    (pt * (pt.clamp_min(1e-12).log() - pq.clamp_min(1e-12).log()))
                    .sum(-1)
                    .mean()
                )
                # post-W_O output error from the attention-distribution change
                d_out = ((pq - pt) @ vfp) @ WO[j].T  # [qwin, hidden]
                hv.append(kl + lam * d_out.pow(2).mean())
            vals[h].append(torch.stack(hv).mean())
    return torch.stack([torch.stack(v).mean() for v in vals])


def v_loss_by_head(
    seqs, R_k, R_v, mu, WO, clip_k, clip_v, grp, prefix, recent, qwin, scale
):
    """Output-aware V loss per KV head: post-W_O value reconstruction under P_q."""
    kvh = R_v.shape[0]
    vals = [[] for _ in range(kvh)]
    for s in seqs:
        keff, _ = k_eff_headwise(s.k, R_k, mu, clip_k, grp, prefix, recent)
        veff = v_eff_headwise(s.v, R_v, clip_v, grp, prefix, recent)
        L = keff.shape[0]
        g = head_layout(s.q, kvh)
        lo = max(1, L - qwin)
        m = torch.arange(L, device=DEV).unsqueeze(0) > torch.arange(
            lo, L, device=DEV
        ).unsqueeze(1)
        for h in range(kvh):
            ks = keff[:, h]
            dv = veff[:, h] - s.v[:, h]  # value error [L, hd]
            hv = []
            for gi in range(g):
                j = h * g + gi
                qt = s.q[lo:L, j]
                Ls = ((qt @ ks.T) * scale).masked_fill(m, float("-inf"))
                pq = torch.softmax(Ls, -1)
                d_out = (pq @ dv) @ WO[j].T  # [qwin, hidden]
                hv.append(d_out.pow(2).mean())
            vals[h].append(torch.stack(hv).mean())
    return torch.stack([torch.stack(v).mean() for v in vals])


def learn_rotation(
    R_base, base_held, loss_fn_calib, loss_fn_held, steps, lr, margin, kvh, hd
):
    A = torch.zeros(kvh, hd, hd, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([A], lr=lr)
    best = base_held.detach().clone()
    bestR = R_base.detach().clone()
    for step in range(steps):
        opt.zero_grad()
        R = R_base @ torch.matrix_exp(A - A.transpose(-1, -2))
        loss = loss_fn_calib(R).mean()
        loss.backward()
        opt.step()
        if step % 20 == 0 or step == steps - 1:
            with torch.no_grad():
                R = R_base @ torch.matrix_exp(A - A.transpose(-1, -2))
                hb = loss_fn_held(R)
                imp = hb < best
                best = torch.where(imp, hb, best)
                bestR[imp] = R.detach()[imp]
    keep = best < base_held * (1 - margin)
    Rf = R_base.detach().clone()
    Rf[keep] = bestR[keep]
    return Rf, keep, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-path", required=True)
    ap.add_argument("--k-oscar", required=True, help="OSCAR K rotation (base)")
    ap.add_argument("--v-oscar", required=True, help="OSCAR V rotation (base)")
    ap.add_argument("--k-mean", required=True)
    ap.add_argument("--k-out", required=True)
    ap.add_argument("--v-out", required=True)
    ap.add_argument(
        "--model-path", default=None, help="HF model id or local snapshot parent"
    )
    ap.add_argument(
        "--hf-snapshot", default=None, help="Explicit HF snapshot directory"
    )
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--clip-k", type=float, default=0.96)
    ap.add_argument("--clip-v", type=float, default=0.92)
    ap.add_argument("--calib", default="2,8")
    ap.add_argument("--held", default="10,12")
    ap.add_argument("--prefix", type=int, default=64)
    ap.add_argument("--recent", type=int, default=256)
    ap.add_argument("--qwin", type=int, default=64)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--margin-k", type=float, default=0.0)
    ap.add_argument("--margin-v", type=float, default=0.0)
    a = ap.parse_args()

    torch.manual_seed(0)
    k_state = torch.load(a.k_oscar, map_location="cpu")
    v_state = torch.load(a.v_oscar, map_location="cpu")
    mean_state = torch.load(a.k_mean, map_location="cpu")
    geom = load_model_geometry(model_path=a.model_path, snap_dir=a.hf_snapshot)
    layers = a.layers or geom["layers"]
    wo = WOCache(snap_dir=geom["snapshot"])
    print(
        "OptR model geometry: "
        f"snapshot={geom['snapshot']} layers={layers} hidden={geom['hidden']} "
        f"q_heads={geom['q_heads']} kv_heads={geom['kv_heads']} head_dim={geom['head_dim']}",
        flush=True,
    )
    calib, held = parse_chunks(a.calib), parse_chunks(a.held)
    common_meta = {
        "variant": "sequential",
        "loss_objective": "hybrid",
        "lam": a.lam,
        "steps": a.steps,
        "lr": a.lr,
        "margin_k": a.margin_k,
        "margin_v": a.margin_v,
        "calib_chunks": a.calib,
        "held_chunks": a.held,
    }
    kout = {
        "format_version": 2,
        "objective": "optr_output_aware_k",
        "source_grouping": "kv_head",
        "layers": {},
        **common_meta,
    }
    vout = {
        "format_version": 2,
        "objective": "optr_output_aware_v",
        "source_grouping": "kv_head",
        "layers": {},
        **common_meta,
    }
    kept_k, kept_v = {}, {}
    print(
        f"{'L':>3s} {'kvh':>3s} {'Kbase':>10s} {'Kbest':>10s} "
        f"{'Kkeep':>6s} {'Vbase':>10s} {'Vbest':>10s} {'Vkeep':>6s}",
        flush=True,
    )

    for li in range(layers):
        mu = load_layer_mu(mean_state, li, DEV)
        kvh, hd = mu.shape
        Rk_base = load_headwise_rotation(k_state, li, kvh, hd, DEV)
        Rv_base = load_headwise_rotation(v_state, li, kvh, hd, DEV)
        WO = wo.heads(li, DEV)
        scale = 1.0 / math.sqrt(hd)
        cs = load_fp_sequences(a.dump_path, li, calib, a.prefix, a.recent, DEV)
        hs = load_fp_sequences(a.dump_path, li, held, a.prefix, a.recent, DEV)
        ev = load_layer_entry(k_state, li).get("eigenvalues")
        evv = load_layer_entry(v_state, li).get("eigenvalues")
        if not cs or not hs:
            kout["layers"][li] = {
                "layer_id": li,
                "rotation": Rk_base.cpu().float().contiguous(),
                "eigenvalues": ev,
                "kept_heads": [],
            }
            vout["layers"][li] = {
                "layer_id": li,
                "rotation": Rv_base.cpu().float().contiguous(),
                "eigenvalues": evv,
                "kept_heads": [],
            }
            print(f"{li:3d} {kvh:3d} {'skip':>10s}", flush=True)
            continue

        # K phase.
        with torch.no_grad():
            kbase = k_loss_by_head(
                hs,
                Rk_base,
                mu,
                WO,
                a.clip_k,
                a.group_size,
                a.prefix,
                a.recent,
                a.qwin,
                scale,
                a.lam,
            )
        Rk_final, keepk, kbest = learn_rotation(
            Rk_base,
            kbase,
            lambda R: k_loss_by_head(
                cs,
                R,
                mu,
                WO,
                a.clip_k,
                a.group_size,
                a.prefix,
                a.recent,
                a.qwin,
                scale,
                a.lam,
            ),
            lambda R: k_loss_by_head(
                hs,
                R,
                mu,
                WO,
                a.clip_k,
                a.group_size,
                a.prefix,
                a.recent,
                a.qwin,
                scale,
                a.lam,
            ),
            a.steps,
            a.lr,
            a.margin_k,
            kvh,
            hd,
        )

        # V phase, conditioned on the optimized K rotation.
        with torch.no_grad():
            vbase = v_loss_by_head(
                hs,
                Rk_final,
                Rv_base,
                mu,
                WO,
                a.clip_k,
                a.clip_v,
                a.group_size,
                a.prefix,
                a.recent,
                a.qwin,
                scale,
            )
        Rv_final, keepv, vbest = learn_rotation(
            Rv_base,
            vbase,
            lambda R: v_loss_by_head(
                cs,
                Rk_final,
                R,
                mu,
                WO,
                a.clip_k,
                a.clip_v,
                a.group_size,
                a.prefix,
                a.recent,
                a.qwin,
                scale,
            ),
            lambda R: v_loss_by_head(
                hs,
                Rk_final,
                R,
                mu,
                WO,
                a.clip_k,
                a.clip_v,
                a.group_size,
                a.prefix,
                a.recent,
                a.qwin,
                scale,
            ),
            a.steps,
            a.lr,
            a.margin_v,
            kvh,
            hd,
        )

        kh = torch.nonzero(keepk, as_tuple=False).reshape(-1).cpu().tolist()
        vh = torch.nonzero(keepv, as_tuple=False).reshape(-1).cpu().tolist()
        print(
            f"{li:3d} {kvh:3d} {float(kbase.mean()):10.4g} "
            f"{float(kbest.mean()):10.4g} {len(kh):3d}/{kvh:<2d} "
            f"{float(vbase.mean()):10.4g} {float(vbest.mean()):10.4g} "
            f"{len(vh):3d}/{kvh:<2d}",
            flush=True,
        )

        if kh:
            kept_k[li] = kh
        if vh:
            kept_v[li] = vh
        kout["layers"][li] = {
            "layer_id": li,
            "rotation": Rk_final.cpu().float().contiguous(),
            "eigenvalues": ev,
            "kept_heads": kh,
            "base_held": kbase.cpu().float(),
            "best_held": kbest.cpu().float(),
            "ortho_err": summarize_ortho(Rk_final),
        }
        vout["layers"][li] = {
            "layer_id": li,
            "rotation": Rv_final.cpu().float().contiguous(),
            "eigenvalues": evv,
            "kept_heads": vh,
            "base_held": vbase.cpu().float(),
            "best_held": vbest.cpu().float(),
            "ortho_err": summarize_ortho(Rv_final),
        }
    ensure_parent(a.k_out)
    torch.save(kout, a.k_out)
    torch.save(vout, a.v_out)
    print(
        f"\nOptR kept K {kept_k}\nOptR kept V {kept_v}\nsaved {a.k_out}\nsaved {a.v_out}"
    )


if __name__ == "__main__":
    main()
