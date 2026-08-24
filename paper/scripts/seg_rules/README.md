# Reach-onset and settle boundary rules — the 2026-08-24 investigation

Asks whether the two weakest phase boundaries should change. **Conclusion: no.** Both candidates
cost more than they buy. Full record in `paper/VERIFY.md`; this directory holds the code and the
result tables so the numbers are reproducible rather than merely reported.

## What prompted it

The shipped reach onset fires at **47% of peak wrist speed** and the settle at **52%** — i.e.
mid-motion, not at the start and end of movement. That costs 200 ms and 450 ms of agreement with
`unger2024`'s boundaries and truncates total movement time by 7.5%.

## The headline finding

**Boundary-agreement metrics do not predict measure quality.** A velocity onset at 10% of peak looks
*strictly better* on every boundary statistic — reference agreement 200 -> 33 ms, firing speed
47% -> 11.7%, cross-system agreement unchanged, no failures — and it **destroys interjoint
coordination** (0.571 -> 0.333). Never tune a boundary without re-scoring the measures.

## Scripts

| script | what it does |
|---|---|
| `sweep_boundary_rules.py` | Sweeps four rule families (absolute position, position relative to the reach distance, wrist speed, wrist speed projected on a fixed reach direction) x thresholds x hold, on cached channels. Emits `seg_rule_sweep.csv`. |
| `compare_measures.py` | Compares the twelve movement-quality measures across scored CSVs. Emits `seg_rule_measures.csv`. |

## Reproducing

```bash
conda activate object_tracking
python paper/scripts/seg_rules/sweep_boundary_rules.py        # boundary metrics, cached channels only

# the measure comparison needs the scorer re-run per variant (~3 min each)
for cfg in "pos:end:SHIPPED" "speed:speed:SPEED010" "speed:end:ONSETONLY" "pos:speed:SETTLEONLY"; do
  IFS=: read on se tag <<< "$cfg"
  OT_SEG_ONSET=$on OT_SEG_SETTLE=$se python scripts/score_own_phases.py \
      --anat12 out/scoring/anat12_wv1wa1_theta.csv \
      --out out/scoring/score_own_phases_$tag.csv
done
python paper/scripts/seg_rules/compare_measures.py
```

`OT_SEG_ONSET` / `OT_SEG_SETTLE` / `OT_SEG_PEAK_FRAC` are read by `scripts/seg_sequential.py` and
**default to the shipped `pos` / `end`**, so nothing changes unless they are set.

## Why the settle cannot be fixed

Frame error = (inter-system signal disagreement) / (derivative of the signal at the threshold).
Asking "has motion stopped" forces the threshold into the region where that derivative vanishes, so
every criterion is either early-and-biased or late-and-noisy. Measured: the slope at the settle
crossing is 601 mm/s^2 against 1453 at onset, and 71% of trials cross a 5%-of-peak speed line more
than once after release. The 52%-of-peak firing speed is the *reason* the shipped boundary is
reproducible, not a defect. Positional precision is not the limit — the two systems agree on
distance-to-final-position to 2.2 mm.
