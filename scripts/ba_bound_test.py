"""Does BOUNDING the solve recover BA's peak-vel gain (+4.9% on matched cycles) across ALL 1376
cycles, instead of only the 1136 it survived unbounded?

Unbounded distortion-aware BA improves peak-vel on the cycles it resolves (+7.4%->+4.9% vs pipeline
on the SAME cycles, VERIFIED paired) but blows up on ~17% of trials (runs off into the distortion
far-field, up to 650mm). Two bounds:
  anchor_mm_w : IN-SOLVE soft quadratic pull to the pipeline init (LM damping / trust region).
  fallback_mm : POST-HOC revert any joint-frame moved > this (or non-finite) to the pipeline point
                (DLT-as-backup -> can never do worse than the incumbent).
Compare peak-vel %err on ALL resolved cycles + cycle count (did we recover the dropped 240?).
"""
import sys, time, numpy as np
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts")
sys.path.insert(0, "/home/imove/Documents/cup-task")
sys.path.insert(0, "/tmp/claude-1000/-home-imove-Documents-object-tracking/25d11f20-b722-49a2-b616-7d5262e468ad/scratchpad")
import numpy as np, gnn_train as T, gnn_refiner as G, ba_refine as BA
from ba_selection_test import cycle_peakvels, wrist_sig
JOINTS = G.JOINTS
med = lambda a: float(np.nanmedian([z for z in a if np.isfinite(z)]))

trials = [t for t in T.load_clean(need_reproj=True) if t["part"] in ("P07","P08","P15","P17","P19")]
print(f"trials: {len(trials)}", flush=True)

configs = [
    ("unbounded",             dict()),
    ("fallback=80",           dict(fallback_mm=80.0)),
    ("anchor=0.001",          dict(anchor_mm_w=0.001)),
    ("anchor=0.003",          dict(anchor_mm_w=0.003)),
    ("anchor=0.001+fb=80",    dict(anchor_mm_w=0.001, fallback_mm=80.0)),
]
print(f"\n{'config':22s} {'peak-vel':>9s} {'cycles':>7s} {'reverted':>9s}", flush=True)
for name, kw in configs:
    t0 = time.time(); pv = []; nfb = 0
    print(f"  [{name}] starting {len(trials)} trials...", flush=True)
    for i, t in enumerate(trials):
        side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
        womc = t["omc"][:, wi].copy(); womc[~t["valid"][:, wi]] = np.nan
        Xr, info = BA.refine_trial_ba(t, 0.0, iters=60, **kw)
        nfb += info.get("n_fallback", 0)
        wb = wrist_sig(Xr, t, True)
        for f, v in cycle_peakvels(wb, womc):
            if np.isfinite(v):
                pv.append(v)
        if (i + 1) % 40 == 0:
            print(f"      [{name}] {i+1}/{len(trials)}  running peak-vel {med(pv):+.2f}%  "
                  f"({len(pv)} cyc, {nfb} reverted, {time.time()-t0:.0f}s)", flush=True)
    print(f"  >> {name:22s} {med(pv):+7.2f}% {len(pv):7d} cyc {nfb:9d} reverted  "
          f"({time.time()-t0:.0f}s)", flush=True)
print("\nPIPELINE baseline (savgol) on ALL 1376 cycles = +6.60% ; unbounded-BA on its 1136 = +4.86%.",
      flush=True)
print("DONE", flush=True)
