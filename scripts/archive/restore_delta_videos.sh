#!/usr/bin/env bash
# Restore the DELTA staged videos for P15/P17/P19 from the SMB share.
#
# WHY THIS EXISTS: bigrun.sh (2026-07-16) ended each trial with
#     rm -f "$D/staged/${PRE}".*.mp4   # "free the video once detected; dets are what we need"
# That was wrong. Within a day we needed the pixels again (COCO yolo26x-seg teacher for the cup
# finetune) and they were gone. It also violated the standing rule: NEVER auto-delete experiment
# data -- new metrics routinely need the originals. It saved ~16GB against 651GB free, i.e. it
# bought nothing. Do not add an `rm` to this script.
#
# Resumable: cp -n / skip-if-exists, so re-running costs nothing for what's already local.
# cam1 is 720p on the share while its calib says 1080p -> upscale (same as bigrun.sh did).
set -u
G="/run/user/1000/gvfs/smb-share:server=nslliappl01.lli.local,share=research_analyzed_dataset/DELTA"
CT=/home/imove/Documents/cup-task
PARTS="${*:-P17 P19 P15}"        # P17/P19 first: they are the held-out eval participants
JOBS=6                            # measured: 6 streams ~7.9MB/s vs 5.2 serial; gvfs is the cap

for P in $PARTS; do
  V="$G/DELTA/DATA/data_newStruc/$P/01_Measurement/04_Video/03_Cut/drinking"
  D="$CT/cache/delta/$P"
  mkdir -p "$D/staged"
  n=0; tot=$(ls "$V/cam1"/*.mp4 2>/dev/null | wc -l)
  echo "#### $P start $(date +%H:%M:%S)  ($tot trials)"
  while IFS= read -r f; do
    t=$(basename "$f" .mp4)
    PRE="delta_${P}_${t}"
    n=$((n+1))
    for cam in 1 2 3 4 5 6 7 8 9 10; do
      dst="$D/staged/${PRE}.${cam}.mp4"
      [ -f "$dst" ] && continue
      src="$V/cam$cam/$t.mp4"
      [ -f "$src" ] || continue
      if [ "$cam" = "1" ]; then
        ffmpeg -y -loglevel error -i "$src" -vf scale=1920:1080 -c:v libx264 -crf 18 \
               -preset veryfast "$dst" </dev/null || echo "  ffmpeg FAIL $t cam1"
      else
        cp -n "$src" "$dst" || echo "  cp FAIL $t cam$cam" &
      fi
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    done
    wait
    [ $((n % 5)) -eq 0 ] && echo "  $P $n/$tot  $(du -sh "$D/staged" 2>/dev/null | cut -f1)  $(date +%H:%M:%S)"
  done < <(ls "$V/cam1"/*.mp4 2>/dev/null)
  echo "#### $P DONE $(date +%H:%M:%S)  clips=$(ls "$D/staged"/*.mp4 2>/dev/null | wc -l)  $(du -sh "$D/staged" | cut -f1)"
done
echo "RESTORE COMPLETE $(date +%H:%M:%S)"
