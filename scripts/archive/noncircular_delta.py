"""Non-circular tests of the 'bias' claim.

The identity  MMC_effect = true_effect + (bias_L - bias_R)  is VACUOUS (the OMC terms
cancel; it holds for any numbers). So is the sign-consistency of the artifact, since
artifact == MMC_effect - true_effect, which is ~ -true_effect whenever MMC misses the
effect -- and true_effect always points the same clinical way.

These three tests can each FAIL:
  A. SPLIT-HALF bias reproducibility WITHIN a side. Estimate bias on half the trials,
     predict the other half. Never touches the other side or the effect.
  B. Cross-participant ANATOMICAL bias consistency (does right-arm bias replicate?).
  C. SNR ceiling maxR = sig/sqrt(sig^2+err^2) per participant -- is P15/P17's low r
     instrument error, or just range restriction (small sig)?
"""
import sys, re, numpy as np
sys.path.insert(0, 'scripts')
from validate_cohort_delta import _trial, DELTA
rng = np.random.default_rng(0)

KEYS = ["max_trunk_displacement", "elbow_extension_reaching", "shoulder_flexion_reaching",
        "peak_velocity__max", "peak_velocity__p90", "peak_elbow_ang_vel__p90"]
PARTS = ["P14", "P15", "P17"]
AFF = {"P14": "left", "P15": "left", "P17": "right"}

store = {}
for part in PARTS:
    dd = DELTA / part / "dets"
    trials = sorted({re.sub(r"\.\d+\.pose\.json$", "", f.name).replace(f"delta_{part}_", "")
                     for f in dd.glob("*.pose.json")})
    nerr = 0
    for t in trials:
        m = re.search(r"_(R|L)_(unaffected|affected)$", t)
        if not m:
            continue
        side = "right" if m.group(1) == "R" else "left"
        try:
            r = _trial(part, t, side)
        except Exception:
            nerr += 1
            continue
        if r:
            store.setdefault((part, side), []).append(r)
    print(f"{part}: " + "  ".join(f"{s}={len(store.get((part,s),[]))}" for s in ("left","right"))
          + f"   (skipped {nerr})", flush=True)

def pairs(part, side, k):
    v = store.get((part, side), [])
    a = np.array([[o[k], m[k]] for o, m in v
                  if np.isfinite(o.get(k, np.nan)) and np.isfinite(m.get(k, np.nan))])
    return a

print("\n" + "="*100)
print("A. SPLIT-HALF BIAS REPRODUCIBILITY (within side; never sees the other side or the effect)")
print("   bias_A = mean(MMC-OMC) on half A. Does it predict half B's bias?")
print("   ratio = bias_B/bias_A ~ 1 => systematic.  |ratio| wild / sign flips => bias is NOISE.")
print("="*100)
print(f"{'measure':30}{'part':6}{'side':7}{'n':>4}{'biasA':>10}{'biasB':>10}{'ratio':>9}{'SD/sqrt(n)':>12}")
print("-"*100)
for k in KEYS:
    for part in PARTS:
        for side in ("left", "right"):
            a = pairs(part, side, k)
            if len(a) < 8:
                continue
            d = a[:, 1] - a[:, 0]
            rs = []
            for _ in range(400):
                idx = rng.permutation(len(d)); h = len(d)//2
                bA, bB = d[idx[:h]].mean(), d[idx[h:]].mean()
                rs.append(bB/bA if abs(bA) > 1e-9 else np.nan)
            ratio = np.nanmedian(rs)
            se = d.std(ddof=1)/np.sqrt(len(d))
            print(f"{k:30}{part:6}{side:7}{len(d):4d}{d[:len(d)//2].mean():10.2f}"
                  f"{d[len(d)//2:].mean():10.2f}{ratio:9.2f}{se:12.2f}")

print("\n" + "="*100)
print("B. CROSS-PARTICIPANT ANATOMICAL BIAS  (landmark story predicts SAME value everywhere)")
print("="*100)
print(f"{'measure':30}{'side':8}" + "".join(f"{p:>12}" for p in PARTS))
print("-"*100)
for k in KEYS:
    for side in ("left", "right"):
        row = f"{k:30}{side:8}"
        for part in PARTS:
            a = pairs(part, side, k)
            row += f"{(a[:,1]-a[:,0]).mean():12.2f}" if len(a) >= 4 else f"{'--':>12}"
        print(row)

print("\n" + "="*100)
print("C. SNR CEILING: is low r instrument error, or RANGE RESTRICTION (small OMC sig)?")
print("   maxR = sig/sqrt(sig^2+err^2).  gotR near maxR => instrument is fine, measure is flat.")
print("="*100)
print(f"{'measure':30}{'part':6}{'side':7}{'n':>4}{'OMCsig':>9}{'err':>9}{'maxR':>7}{'gotR':>7}{'slope':>8}")
print("-"*100)
from scipy.stats import pearsonr
for k in KEYS:
    for part in PARTS:
        for side in ("left", "right"):
            a = pairs(part, side, k)
            if len(a) < 6 or a[:,0].std() < 1e-9 or a[:,1].std() < 1e-9:
                continue
            sig = a[:, 0].std(ddof=1)
            err = (a[:, 1] - a[:, 0]).std(ddof=1)
            maxR = sig/np.sqrt(sig**2 + err**2)
            gotR = pearsonr(a[:, 0], a[:, 1])[0]
            slope = np.polyfit(a[:, 0], a[:, 1], 1)[0]
            print(f"{k:30}{part:6}{side:7}{len(a):4d}{sig:9.2f}{err:9.2f}{maxR:7.2f}{gotR:7.2f}{slope:8.2f}")
