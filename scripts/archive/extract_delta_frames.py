"""Pre-extract DELTA frames as LETTERBOXED 640 BGR (matches prep_view's geometric output) to a flat
memmap cache. The letterbox depends only on native (W,H) — deterministic — so caching the letterboxed
canvas is lossless for geometry, and ~15GB not ~93GB (raw native). The loader's fast path reads this
canvas and computes P_in / kp_in from the (known) native size, no native pixels needed.
Run: python scripts/extract_delta_frames.py
"""
import sys, os
sys.path.insert(0, '/home/imove/Documents/cup-task/scripts')
sys.path.insert(0, '/home/imove/Documents/cup-task')
import numpy as np, cv2
import compare_pose_omc_delta as H
import results_v3_delta as R
from pipeline.mv3d.imgproc import letterbox_image
H.use_good_cams()

PART = 'P07'
TRIALS = [f'trial_{i}_L_unaffected' for i in [10, 11, 12, 13, 14, 15]]
IMSZ = 640
OUT = '/home/imove/Documents/cup-task/cache/delta/_frames_lb640'   # letterboxed 640 BGR
os.makedirs(OUT, exist_ok=True)


def extract(part, trial):
    calib = R._calib(part)
    for c in calib:
        vid = H.DELTA/part/'staged'/f'delta_{part}_{trial}.{c.split("_")[1]}.mp4'
        if not vid.exists():
            continue
        out = f'{OUT}/{part}_{trial}_{c}.npy'
        if os.path.exists(out):
            print(f'  {trial} {c}: exists, skip', flush=True); continue
        cap = cv2.VideoCapture(str(vid))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        arr = np.empty((n, IMSZ, IMSZ, 3), np.uint8); got = 0
        for f in range(n):
            ok, im = cap.read()
            if not ok:
                break
            canvas, _, _, _ = letterbox_image(im, IMSZ)   # BGR letterboxed (geometry-consistent)
            arr[f] = canvas; got = f + 1
        cap.release()
        np.save(out, arr[:got])
        print(f'  {trial} {c}: {got} frames -> {os.path.getsize(out)//1024//1024}MB', flush=True)


if __name__ == '__main__':
    for tr in TRIALS:
        print(f'{tr}...', flush=True); extract(PART, tr)
    print('DONE letterboxed frame extraction', flush=True)
