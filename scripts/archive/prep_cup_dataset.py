"""Turn the reject-then-fill label pool into an ultralytics YOLO-seg dataset (train/val split).

Split at TRIAL granularity (never frame-random): adjacent frames of one trial are near-duplicates,
so a frame-random split leaks val into train and inflates the score. Val = hold out ~VAL_FRAC of
the (participant, trial) groups, stratified per participant so every participant appears in val.

    python scripts/prep_cup_dataset.py --pool data/delta_cup_cohort --val-frac 0.15
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path


def group_key(stem):
    # <part>_<trial...>_cam_<c>_fNNNNN  -> (part, trial) ; camera & frame stripped
    m = re.match(r"(P\d+)_(.+)_cam_\d+_f\d+$", stem)
    return (m.group(1), m.group(2)) if m else (stem, stem)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/delta_cup_cohort")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-parts", nargs="+", default=None,
                    help="participant-HOLDOUT split: these go to TRAIN, all others to VAL. "
                         "(cross-participant generalization test; overrides trial-stratified split)")
    ap.add_argument("--train-count", type=int, default=0,
                    help="cap TRAIN to EXACTLY this many images (deterministic subsample, seeded) "
                         "for a reproducible pipeline. Errors if the pool has fewer.")
    ap.add_argument("--only-parts", nargs="+", default=None,
                    help="restrict the whole pool to these participants (for within-participant "
                         "experiments)")
    ap.add_argument("--train-trials", nargs="+", default=None,
                    help="these TRIAL names -> train, all other trials (in the restricted pool) "
                         "-> val. For the 1-trial-generalization test.")
    ap.add_argument("--out-prefix", default="", help="prefix for cup.yaml/train.txt/val.txt names")
    a = ap.parse_args(argv)
    pool = Path(a.pool).resolve()
    imgs = sorted((pool / "images").glob("*.jpg"))
    if a.only_parts:
        op = set(a.only_parts)
        imgs = [p for p in imgs if group_key(p.stem)[0] in op]
    if not imgs:
        raise SystemExit(f"no images in {pool}/images")

    # group images by (participant, trial); assign whole groups to train/val
    by_part = defaultdict(list)
    groups = defaultdict(list)
    for p in imgs:
        k = group_key(p.stem)
        groups[k].append(p)
    for k in groups:
        by_part[k[0]].append(k)
    rng = random.Random(a.seed)
    train, val = [], []
    if a.train_trials:
        tt = set(a.train_trials)
        for k, ps in groups.items():
            dst = train if k[1] in tt else val
            dst.extend(str(x) for x in ps)
        print(f"train-trials: {sorted(tt)}  ({len(train)} train imgs, "
              f"{len(set(k[1] for k in groups)) - len(tt)} val trials)")
    elif a.train_parts:
        tp = set(a.train_parts)
        for part, ks in sorted(by_part.items()):
            dst = train if part in tp else val
            for k in ks:
                dst.extend(str(x) for x in groups[k])
        print(f"participant-holdout: train={sorted(tp)}  val={sorted(set(by_part)-tp)}")
    else:
        for part, ks in sorted(by_part.items()):
            rng.shuffle(ks)
            nval = max(1, round(len(ks) * a.val_frac))
            vset = set(ks[:nval])
            for k in ks:
                (val if k in vset else train).extend(str(x) for x in groups[k])
    rng.shuffle(train); rng.shuffle(val)
    if a.train_count:
        if len(train) < a.train_count:
            raise SystemExit(f"pool has only {len(train)} train imgs < requested {a.train_count}; "
                             f"rebuild with a lower --vid-stride for more surplus")
        train = train[:a.train_count]      # deterministic (seeded shuffle above)
        print(f"capped TRAIN to exactly {a.train_count}")
    pre = a.out_prefix
    (pool / f"{pre}train.txt").write_text("\n".join(train) + "\n")
    (pool / f"{pre}val.txt").write_text("\n".join(val) + "\n")
    yaml = (f"path: {pool}\n"
            f"train: {pre}train.txt\n"
            f"val: {pre}val.txt\n"
            f"names:\n  0: cup\n")
    (pool / f"{pre}cup.yaml").write_text(yaml)
    parts = sorted(by_part)
    print(f"pool {pool}: {len(imgs)} imgs, {len(groups)} (part,trial) groups over {parts}")
    print(f"  train {len(train)} imgs / val {len(val)} imgs  ({len(val)/(len(train)+len(val))*100:.0f}% val)")
    print(f"  wrote {pool}/{pre}cup.yaml, {pre}train.txt, {pre}val.txt")


if __name__ == "__main__":
    main()
