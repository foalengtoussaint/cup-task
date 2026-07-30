#!/usr/bin/env bash
# Bring NEW valid participants into cup-task: pull video+c3d from the SMB share, run detections,
# run UETrack cup tracks. Selection (done separately): valid calibration (RMS<6 in the study CSV)
# AND has AutoMQ OMC. Result: P10 P12 P13 P14 P25 (P16/18/20/21/22 excluded = no OMC; P24 = no OMC).
#
# Fully RESUMABLE: every stage skips work whose output already exists (cp -n, skip-if-json,
# cache_uetrack_tracks is idempotent). Safe to re-run after an interruption.
#
#   bash scripts/add_participants.sh                 # all five
#   bash scripts/add_participants.sh P10 P12          # a subset
set -u
G="/run/user/1000/gvfs/smb-share:server=nslliappl01.lli.local,share=research_analyzed_dataset/DELTA"
VID="$G/DELTA/DATA/data_newStruc"
OMC="$G/iDrink/OMC_data_newStruct/Data"
CT=/home/imove/Documents/cup-task
PARTS="${*:-P10 P12 P13 P14 P25}"
JOBS=6

cd "$CT" || exit 1
echo "#################### add_participants START $(date '+%F %T') ####################"
echo "parts: $PARTS   share reachable: $([ -d "$VID" ] && echo yes || echo NO)"
[ -d "$VID" ] || { echo "SHARE NOT MOUNTED — abort"; exit 1; }

for P in $PARTS; do
  echo; echo "======================== $P  $(date '+%T') ========================"
  V="$VID/$P/01_Measurement/04_Video/03_Cut/drinking"
  CB="$VID/$P/01_Measurement/04_Video/05_Calib_before"
  D="$CT/cache/delta/$P"
  mkdir -p "$D/staged" "$D/c3d" "$D/dets" "$D/calib"

  # ---- 0. calibration TOML (only if missing) ----
  if [ -z "$(ls "$D/calib/"*.toml 2>/dev/null)" ]; then
    cal=$(ls "$CB/"*.toml 2>/dev/null | head -1)
    [ -n "$cal" ] && cp -n "$cal" "$D/calib/${P}_calibration.toml" && echo "[$P] calib: $(basename "$cal")"
  fi
  echo "[$P] calib: $(ls "$D/calib/"*.toml 2>/dev/null | wc -l) toml"

  # ---- 1. OMC c3d ----
  nc=0
  for c in "$OMC/$P/c3d/"*.c3d; do
    [ -f "$c" ] || continue
    cp -n "$c" "$D/c3d/" 2>/dev/null && nc=$((nc+1))
  done
  echo "[$P] c3d: $(ls "$D/c3d/"*.c3d 2>/dev/null | wc -l) local"

  # ---- 2. staged video (cam1 720p->1080p upscale, rest copy) ----
  tot=$(ls "$V/cam1/"*.mp4 2>/dev/null | wc -l); n=0
  echo "[$P] pulling video: $tot trials x cam 1-5"
  while IFS= read -r f; do
    t=$(basename "$f" .mp4); PRE="delta_${P}_${t}"; n=$((n+1))
    for cam in 1 2 3 4 5; do            # cam 1-5 only (the shared 5-cam rig; 6-10 not needed)
      dst="$D/staged/${PRE}.${cam}.mp4"; [ -f "$dst" ] && continue
      src="$V/cam$cam/$t.mp4"; [ -f "$src" ] || continue
      if [ "$cam" = "1" ]; then
        ffmpeg -y -loglevel error -i "$src" -vf scale=1920:1080 -c:v libx264 -crf 18 \
               -preset veryfast "$dst" </dev/null || echo "  ffmpeg FAIL $t cam1"
      else
        cp -n "$src" "$dst" || echo "  cp FAIL $t cam$cam" &
      fi
      while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n; done
    done
    wait
    [ $((n % 10)) -eq 0 ] && echo "  [$P] pull $n/$tot  $(du -sh "$D/staged" 2>/dev/null | cut -f1)  $(date '+%T')"
  done < <(ls "$V/cam1/"*.mp4 2>/dev/null)
  echo "[$P] video DONE: $(ls "$D/staged/"*.mp4 2>/dev/null | wc -l) clips  $(du -sh "$D/staged" 2>/dev/null | cut -f1)"

  # ---- 3. detections (cup + pose) per rep, skip reps already fully detected ----
  echo "[$P] detections $(date '+%T')"
  reps=$(ls "$D/staged/"*.mp4 2>/dev/null | sed -E 's#.*/(delta_'"$P"'_[^.]*(_[^.]*)*)\.[0-9]+\.mp4#\1#' | sort -u)
  nr=$(echo "$reps" | grep -c .); k=0
  for rep in $reps; do
    k=$((k+1))
    # already done if every staged cam for this rep has a .cup.json
    ncam=$(ls "$D/staged/${rep}."*.mp4 2>/dev/null | wc -l)
    ndet=$(ls "$D/dets/${rep}."*.cup.json 2>/dev/null | wc -l)
    if [ "$ndet" -ge "$ncam" ] && [ "$ncam" -gt 0 ]; then continue; fi
    python3 scripts/detect_rep_batched.py "$D/staged" "$rep" -o "$D/dets" --batch 32 \
      2>&1 | grep -vE 'Warning|warn' | sed "s/^/  [$P det $k\/$nr] /"
  done
  echo "[$P] detections DONE: $(ls "$D/dets/"*.cup.json 2>/dev/null | wc -l) cup jsons"

  # ---- 4. UETrack cup tracks (idempotent; skips P13 which already has them) ----
  echo "[$P] UETrack $(date '+%T')"
  python3 scripts/cache_uetrack_tracks.py --parts "$P" 2>&1 \
    | grep -vE 'Warning|warn|MULTI_MODAL|Update interval|Update threshold|USE_NLP' | sed "s/^/  [$P uet] /"

  echo "[$P] ALL DONE $(date '+%T')  tracks=$(ls "$CT/cache/tracks_uetrack/${P}__"*.json 2>/dev/null | wc -l)"
done

echo; echo "#################### add_participants FINISHED $(date '+%F %T') ####################"
for P in $PARTS; do
  echo "  $P: $(ls "$CT/cache/delta/$P/staged/"*.mp4 2>/dev/null|wc -l) clips, $(ls "$CT/cache/delta/$P/dets/"*.cup.json 2>/dev/null|wc -l) dets, $(ls "$CT/cache/delta/$P/c3d/"*.c3d 2>/dev/null|wc -l) c3d, $(ls "$CT/cache/tracks_uetrack/${P}__"*.json 2>/dev/null|wc -l) uetrack"
done
