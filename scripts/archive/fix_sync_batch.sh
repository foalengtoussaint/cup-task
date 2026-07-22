#!/usr/bin/env bash
# Batch sync-repair: for each (participant, desynced camera), download the uncut locally,
# motion-energy align-recut ALL trials at full fps, and re-detect (pose+cup) on the re-cuts.
# Validated on P13 cam2: cup consensus 4cams 48% -> +fixed-cam2 59% (vs +broken-cam2 32%).
# NO deletes -- re-cut clips + uncut + dets all kept in cache/delta/<P>/.
set -u
G="/run/user/1000/gvfs/smb-share:server=nslliappl01.lli.local,share=research_analyzed_dataset"
V="$G/DELTA/DELTA/DATA/data_newStruc"
CT=/home/imove/Documents/cup-task
source /home/imove/miniconda3/etc/profile.d/conda.sh; conda activate object_tracking
cd "$CT"

# (participant camera) pairs with a clean desync WITHIN cams 1-5 (from cam_quality audits)
PAIRS="${*:-P13:2 P19:2 P10:2}"

for pc in $PAIRS; do
  P="${pc%%:*}"; CAM="${pc##*:}"
  D="$CT/cache/delta/$P"
  echo "#### $P cam$CAM start $(date +%H:%M:%S)"
  # 1. download uncut locally if not present (~33s, sequential read ~32MB/s)
  loc="$D/uncut/cam${CAM}.mp4"
  if [ ! -f "$loc" ]; then
    mkdir -p "$D/uncut"
    src="$V/$P/01_Measurement/04_Video/01_Uncut/cam${CAM}.mp4"
    [ -f "$src" ] || { echo "  no uncut on share: $src"; continue; }
    echo "  downloading uncut cam$CAM ..."; cp "$src" "$loc"
  fi
  # 2. align-recut ALL trials at full fps (thumbnails cached in cache/delta/_thumbs)
  python scripts/delta_recut.py --part "$P" --cam "$CAM" --align --ref-cam 3 --all-trials \
    2>&1 | grep -E 'located|LOW-Q|wrote|Error' | sed "s|^|  |"
  # 3. re-detect pose+cup on every re-cut clip (cam only -> ~3s each)
  echo "  re-detecting cam$CAM re-cuts ..."
  n=0
  for f in "$D"/recut/delta_${P}_*.${CAM}.mp4; do
    [ -f "$f" ] || continue
    PRE=$(basename "$f" | sed "s|\.${CAM}\.mp4||")
    python scripts/detect_rep_batched.py "$D/recut" "$PRE" -o "$D/recut_dets" \
      --pose-model models/yolo26s-pose.pt --cup-model models/cup_clean3d_refill.pt \
      --batch 32 --device 0 2>&1 | grep -E 'Error|Traceback' | sed "s|^|    |"
    n=$((n+1))
  done
  echo "#### $P cam$CAM DONE $(date +%H:%M:%S)  recut+detected $n trials"
done
echo "SYNC-FIX BATCH COMPLETE $(date +%H:%M:%S)"
