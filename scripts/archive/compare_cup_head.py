"""Do OMC and cup-task AGREE on the cup->head signal, and on the phases it produces?

The only physical signal both truly share is CUP -> HEAD distance:
  * OMC:  QTM mocap cup  ->  QTM mocap head           (cache/qtm_omc/<stem>.json)
  * MMC:  refill cup     ->  yolo26s pose head-proxy   (nose/eye centroid)

  Both MMC inputs are cup-task's own detectors. The refill cup track
  (track3d_clean3d_refill) was produced by models/cup_clean3d_refill.pt -- VERIFIED
  md5-identical to pscale_1_clean3d_refill/best.pt, i.e. cup-task's cup model IS the one
  that made this cache. So MMC = cup-task cup + cup-task yolo head, vs OMC = fully mocap.

No wrist, no iMOVE segmenter, no MeTRAbs. We compute cup->head on BOTH sides, and run the SAME
simple rule on each, so any difference is the SIGNAL SOURCE, not the method:

  signal   : d(t) = |cup - head|, low-pass 4Hz
  transport window : cup speed crosses 80 mm/s (hysteresis)
  drink    : inside window, d < closest + 15%*(steady - closest)   [van Andel, self-normalising]

Reports:
  SIGNAL   : per-rep correlation of the two d(t) curves + median |Δ| after removing the
             constant offset (mocap head marker sits higher than the nose proxy, so a fixed
             offset is expected and not an error -- the SHAPE is what matters).
  PHASE    : drink onset/offset error (ms) and dwell-duration error, cup-task vs OMC, from the
             identical rule on each side's own signal.

    python scripts/compare_cup_head.py            # all reps with pose cache + qtm_omc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import segment, triangulate
from pipeline.kalman_3d import load_calibration

ROOT = Path(__file__).resolve().parents[1]
POSE = ROOT / "cache" / "pose_models"
OMC = ROOT / "cache" / "qtm_omc"
FPS = 60.0
DRINK_FRAC = 0.15
CUP_MOVE_THR = 80.0     # mm/s, transport-window gate (both sides, same value)
LP_HZ = 4.0


def _lp(x):
    v = np.isfinite(x)
    if v.sum() < 8:
        return x
    xi = x.copy()
    idx = np.flatnonzero(v)
    xi = np.interp(np.arange(len(x)), idx, x[idx])
    b, a = butter(2, LP_HZ / (FPS / 2))
    return filtfilt(b, a, xi)


def _speed(xyz):
    """Per-frame speed, NaN where either endpoint is missing. Do NOT interpolate across gaps
    -- interpolating over a despiked dropout stitches a fake teleport (the P07 14657 mm/s
    artifact). The segmenter masks NaN; so do we."""
    d = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * FPS
    both = np.isfinite(xyz[:-1]).all(1) & np.isfinite(xyz[1:]).all(1)
    out = np.full(len(xyz), np.nan)
    out[1:][both] = d[both]
    return out


def _runs(mask):
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def phases_from_cup_head(cup_xyz, d_cuphead):
    """Same rule on either side. Returns (transport_start, drink0, drink1, transport_end)."""
    d = _lp(d_cuphead)
    speed = _lp(_speed(cup_xyz))
    win = [(s, e) for s, e in _runs(speed > CUP_MOVE_THR) if e - s >= 5]
    if not win:
        return None
    ts, te = win[0][0], win[-1][1]
    in_win = np.zeros(len(d), bool); in_win[ts:te] = True
    fin = np.isfinite(d)
    steady = np.percentile(d[fin], 90)
    closest = np.percentile(d[fin], 5)
    near = d < closest + DRINK_FRAC * (steady - closest)
    dr = [(s, e) for s, e in _runs(in_win & near) if e - s >= 5]
    if not dr:
        return (ts, None, None, te)
    return (ts, dr[0][0], dr[-1][1], te)


REFILL = Path("/home/imove/Documents/object_tracking/experiments/drink_study/cache"
              "/track3d_clean3d_refill")


def _load_refill_cup(stem, T):
    """(T,3) cup from the finetuned detector's cached 3D track (RTS-smoothed), or None."""
    def norm(s):
        p = s.split("_"); return "_".join(p[1:]) if len(p) > 1 and p[0] == p[1] else s
    for f in REFILL.glob("*__clean3d_refill.json"):
        if norm(f.name.replace("__clean3d_refill.json", "")) == norm(stem):
            d = json.loads(f.read_text())
            cup = np.full((T, 3), np.nan)
            for fr in d["frames"]:
                if fr["fr"] < T and fr.get("rts"):
                    cup[fr["fr"]] = fr["rts"]
            return cup
    return None


def _load_omc(stem, T):
    f = OMC / f"{stem}.json"
    if not f.exists():
        return None, None
    d = json.loads(f.read_text())
    if d.get("head") is None:
        return None, None
    def arr(key):
        out = np.full((T, 3), np.nan)
        for i, p in enumerate(d[key]):
            if i < T and p is not None:
                out[i] = p
        return out
    return arr("cup"), arr("head")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-root", type=Path,
                    default=Path("/home/imove/Documents/object_tracking/data/calib"))
    ap.add_argument("--model", default="s")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "cache" / "cup_head_compare.json")
    a = ap.parse_args(argv)

    stems = sorted(p.parent.name for p in POSE.glob(f"*/yolo26{a.model}.2d.json"))
    stems = [s for s in stems if (OMC / f"{s}.json").exists()]
    print(f"comparable (pose + omc): {len(stems)}", flush=True)

    calib, results = {}, {}
    sig_corr, sig_absdiff, dwell_err, onset_err, offset_err = [], [], [], [], []
    spd_corr, gate80, gate150 = [], [], []
    for i, stem in enumerate(stems):
        part = stem.split("_")[0]
        cf = a.calib_root / part / "calibration.toml"
        if not cf.exists():
            continue
        if part not in calib:
            calib[part] = load_calibration(cf, target_size=(1920, 1080))
        cams = calib[part]
        per_cam = json.loads((POSE / stem / f"yolo26{a.model}.2d.json").read_text())
        n = max(len(v) for v in per_cam.values())

        # cup-task side: the finetuned cup detector's 3D track (research refill RTS, cached
        # for all reps -- no GPU) + our pose head-proxy (nose/eye centroid).
        ct_cup = _load_refill_cup(stem, n)
        if ct_cup is None:
            continue
        ct_head, _ = segment.track_confidence(
            triangulate.triangulate_target(per_cam, cams, triangulate._mouth_point, n))
        d_ct = np.linalg.norm(ct_cup - ct_head, axis=1)

        # OMC side: mocap cup + mocap head
        omc_cup, omc_head = _load_omc(stem, n)
        if omc_cup is None:
            continue
        d_omc = np.linalg.norm(omc_cup - omc_head, axis=1)

        # SIGNAL agreement: correlation of shape + offset-removed |Δ|
        both = np.isfinite(d_ct) & np.isfinite(d_omc)
        if both.sum() < 20:
            continue
        c = float(np.corrcoef(d_ct[both], d_omc[both])[0, 1])
        off = float(np.median(d_ct[both] - d_omc[both]))     # constant offset (nose vs head marker)
        absd = float(np.median(np.abs((d_ct[both] - off) - d_omc[both])))
        sig_corr.append(c); sig_absdiff.append(absd)

        # cup SPEED agreement (masked -- no interp across gaps, which would re-create the spike)
        s_ct, s_omc = _speed(ct_cup), _speed(omc_cup)
        ms = np.isfinite(s_ct) & np.isfinite(s_omc)
        spd_c = spd_g80 = spd_g150 = None
        if ms.sum() >= 20:
            spd_c = float(np.corrcoef(s_ct[ms], s_omc[ms])[0, 1])
            spd_g80 = float(((s_ct[ms] > 80) == (s_omc[ms] > 80)).mean())
            spd_g150 = float(((s_ct[ms] > 150) == (s_omc[ms] > 150)).mean())
            spd_corr.append(spd_c); gate80.append(spd_g80); gate150.append(spd_g150)

        # PHASE agreement: identical rule on each side's own signal
        ph_ct = phases_from_cup_head(ct_cup, d_ct)
        ph_omc = phases_from_cup_head(omc_cup, d_omc)
        rec = {"sig_corr": round(c, 3), "sig_absdiff_mm": round(absd, 1),
               "offset_mm": round(off, 0),
               "spd_corr": round(spd_c, 3) if spd_c is not None else None,
               "gate80_agree": round(spd_g80, 3) if spd_g80 is not None else None}
        if ph_ct and ph_omc and ph_ct[1] is not None and ph_omc[1] is not None:
            on = (ph_ct[1] - ph_omc[1]) / FPS * 1000
            off_ms = (ph_ct[2] - ph_omc[2]) / FPS * 1000
            dw = ((ph_ct[2] - ph_ct[1]) - (ph_omc[2] - ph_omc[1])) / FPS * 1000
            rec.update(drink_onset_err_ms=round(on, 0), drink_offset_err_ms=round(off_ms, 0),
                       dwell_err_ms=round(dw, 0))
            onset_err.append(on); offset_err.append(off_ms); dwell_err.append(dw)
        results[stem] = rec
        print(f"[{i+1}/{len(stems)}] {stem}  r={c:.2f}  |Δ|={absd:.0f}mm  "
              f"dwell_err={rec.get('dwell_err_ms','?')}ms", flush=True)

    a.out.write_text(json.dumps(results, indent=1))
    def stat(name, v):
        v = np.array(v)
        if len(v):
            print(f"  {name:22s} median {np.median(v):+7.1f}   |med| {np.median(np.abs(v)):6.1f}"
                  f"   n={len(v)}", flush=True)
    print(f"\n=== CUP->HEAD DISTANCE SIGNAL (n={len(sig_corr)}) ===", flush=True)
    print(f"  correlation (shape)   median {np.median(sig_corr):.3f}", flush=True)
    if spd_corr:
        print(f"\n=== CUP SPEED SIGNAL (n={len(spd_corr)}) ===", flush=True)
        print(f"  correlation           median {np.median(spd_corr):.3f}", flush=True)
        print(f"  gate 80mm/s agree     median {np.median(gate80)*100:.0f}% of frames",
              flush=True)
        print(f"  gate 150mm/s agree    median {np.median(gate150)*100:.0f}% of frames",
              flush=True)
    stat("offset-removed |Δ| mm", sig_absdiff)
    print(f"\n=== PHASES FROM THE SHARED CUP->HEAD RULE ===", flush=True)
    stat("drink onset err ms", onset_err)
    stat("drink offset err ms", offset_err)
    stat("drink dwell err ms", dwell_err)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
