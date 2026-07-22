#!/usr/bin/env bash
# Import cams 1-5 for the 5-camera DELTA participants and detect pose+cup, so cam_quality_delta
# can confirm their low cameras are synced+calibrated (the premise of the "just use cams 1-5,
# skip fix_slicing" plan). NO rm -- we keep the staged videos (lesson from bigrun.sh).
# P07/P08/P10/P11/P12 are already study-validated (Pose2Sim .trc exists); P13 is the real unknown.
set -u
G="/run/user/1000/gvfs/smb-share:server=nslliappl01.lli.local,share=research_analyzed_dataset"
V="$G/DELTA/DELTA/DATA/data_newStruc"
OMC="$G/DELTA/iDrink/OMC_data_newStruct/Data"
CT=/home/imove/Documents/cup-task
source /home/imove/miniconda3/etc/profile.d/conda.sh; conda activate object_tracking
cd "$CT"
PARTS="${*:-P10 P13}"
NTRIAL=6
for P in $PARTS; do
  VD="$V/$P/01_Measurement/04_Video/03_Cut/drinking"
  CB="$V/$P/01_Measurement/04_Video/05_Calib_before"
  D="$CT/cache/delta/$P"
  mkdir -p "$D/staged" "$D/dets" "$D/c3d" "$D/calib"
  # calib: source name varies (Calib_full_P..toml / P.._calibration.toml) -> canonical name
  cal=$(ls "$CB"/*.toml 2>/dev/null | head -1)
  [ -n "$cal" ] && cp -n "$cal" "$D/calib/${P}_calibration.toml" && echo "#### $P calib: $(basename "$cal")"
  echo "#### $P start $(date +%H:%M:%S)"
  n=0
  for f in "$VD/cam1"/*.mp4; do
    [ $n -ge $NTRIAL ] && break
    t=$(basename "$f" .mp4)
    [ -f "$OMC/$P/c3d/$t.c3d" ] || { echo "  no c3d: $t"; continue; }
    cp -n "$OMC/$P/c3d/$t.c3d" "$D/c3d/" 2>/dev/null
    PRE="delta_${P}_${t}"
    [ -f "$D/dets/${PRE}.2.pose.json" ] && { n=$((n+1)); continue; }
    ok=1
    for cam in 1 2 3 4 5; do
      dst="$D/staged/${PRE}.${cam}.mp4"
      [ -f "$dst" ] && continue
      src="$VD/cam$cam/$t.mp4"
      [ -f "$src" ] || { ok=0; break; }
      if [ "$cam" = "1" ]; then
        ffmpeg -y -loglevel error -i "$src" -vf scale=1920:1080 -c:v libx264 -crf 18 \
               -preset veryfast "$dst" </dev/null || ok=0
      else cp -n "$src" "$dst" || ok=0; fi
    done
    [ "$ok" = "1" ] || { echo "  SKIP $t (missing cam)"; continue; }
    python scripts/detect_rep_batched.py "$D/staged" "$PRE" -o "$D/dets" \
      --pose-model models/yolo26s-pose.pt --cup-model models/cup_clean3d_refill.pt \
      --batch 32 --device 0 2>&1 | grep -E 'done:' | sed "s|^|  $t |"
    n=$((n+1))
  done
  echo "#### $P DONE $(date +%H:%M:%S)  dets=$(ls $D/dets/*.pose.json 2>/dev/null | wc -l)"
done
echo "IMPORT+DETECT COMPLETE $(date +%H:%M:%S)"
