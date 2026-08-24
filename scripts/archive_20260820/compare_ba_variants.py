"""Score cached BA variants per joint: reprojection (px) and OMC agreement (mm).

Reads cache/ba_variants/*.npz -- no solving. DLT is the shipped triangulation (t["mmc"]).
OMC comparison is alignment-free (step + excursion magnitudes), so no Kabsch is fitted and
a coherent shift can neither hide nor manufacture an effect.

    python scripts/compare_ba_variants.py --variants fix perjoint
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gnn_train as T                     # noqa: E402
import gnn_refiner as G                   # noqa: E402
import ba_refine as BA                    # noqa: E402

VAR = ROOT / "cache" / "ba_variants"


def _traj_metrics(P, valid, j):
    w = P[:, j].copy(); w[~valid[:, j]] = np.nan
    step = np.linalg.norm(np.diff(w, axis=0), axis=1)
    exc = np.linalg.norm(w - np.nanmedian(w, axis=0), axis=1)
    return step, exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["fix", "perjoint"])
    a = ap.parse_args()
    loaded = {}
    for v in a.variants:
        p = VAR / f"{v}.npz"
        if not p.exists():
            print(f"  missing {p} -- skipping", flush=True); continue
        z = np.load(p, allow_pickle=True); loaded[v] = dict(zip(z["ids"], z["traj"]))
    if not loaded:
        sys.exit("no variants found")
    print(f"variants: {list(loaded)} (+ dlt baseline)", flush=True)

    srcs = ["dlt"] + list(loaded)
    rp = {(s, n): [] for s in srcs for n in G.JOINTS}      # reprojection px
    om = {(s, n): [] for s in srcs for n in G.JOINTS}      # excursion err mm
    trials = T.load_clean(need_reproj=True)
    for i, t in enumerate(trials):
        tid = f"{t['part']}/{t['trial']}"
        if any(tid not in d for d in loaded.values()):
            continue
        S = BA._to_dev(t); C = t["uv"].shape[1]
        cand = {"dlt": t["mmc"], **{v: d[tid] for v, d in loaded.items()}}
        for s, P in cand.items():
            fin = np.isfinite(P).all(-1)
            X = torch.from_numpy(np.nan_to_num(P).astype(np.float32))[None].cuda()
            res = []
            for c in range(C):
                uvp, _ = G.project_torch(X, S["K"][c], S["dist"][c], S["R"][c], S["tt"][c])
                uvp = uvp.clamp(-1e6, 1e6)
                r = np.linalg.norm(uvp[0].cpu().numpy() - t["uv"][:, c], axis=-1)
                ok = t["uv_valid"][:, c] & fin & np.isfinite(r)
                res.append(np.where(ok, r, np.nan))
            R = np.nanmedian(np.stack(res, 1), axis=1)
            for j, n in enumerate(G.JOINTS):
                v = np.isfinite(R[:, j])
                if v.sum() >= 20:
                    rp[(s, n)].append(float(np.median(R[v, j])))
                _, eo = _traj_metrics(t["omc"], t["valid"], j)
                _, em = _traj_metrics(P, t["valid"], j)
                m = np.isfinite(eo) & np.isfinite(em)
                if m.sum() >= 20:
                    om[(s, n)].append(float(np.median(np.abs(em - eo)[m])))
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(trials)}]", flush=True)

    def col(d, s, n):
        v = d[(s, n)]
        return np.median(v) if v else np.nan
    print(f"\nREPROJECTION px (lower better)          |  OMC EXCURSION err mm (lower better)")
    hdr = " ".join(f"{s:>9s}" for s in srcs)
    print(f"{'joint':16s} {hdr}  | {hdr}")
    for n in G.JOINTS:
        r = " ".join(f"{col(rp,s,n):9.2f}" for s in srcs)
        o = " ".join(f"{col(om,s,n):9.2f}" for s in srcs)
        print(f"{n:16s} {r}  | {o}")
    print("\nDELTA vs DLT (negative = better):")
    print(f"{'joint':16s} " + " ".join(f"{v+' px':>10s} {v+' mm':>10s}" for v in loaded))
    for n in G.JOINTS:
        cells = []
        for v in loaded:
            cells.append(f"{col(rp,v,n)-col(rp,'dlt',n):+10.2f} {col(om,v,n)-col(om,'dlt',n):+10.2f}")
        print(f"{n:16s} " + " ".join(cells))


if __name__ == "__main__":
    main()
