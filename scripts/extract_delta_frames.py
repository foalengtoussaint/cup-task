"""Pre-extract DELTA staged-video frames to a flat on-disk cache (640x640 RGB uint8 .npy per trial/cam).

WHY: the CDRNet train loop was decode-bound — cv2 seek+decode of H.264 per frame per cam (~tens of
ms x5 cams x every step). YOLO trains fast because it reads pre-decoded flat images. This does the
same: decode ONCE to a contiguous (n,640,640,3) uint8 array per (trial,cam); training then mem-maps
and indexes instantly. One-time cost; ~ n*640*640*3 bytes/cam (~600MB/trial/5cam, kept on disk).

Only the frames a trial actually uses (valid + within range) are needed, but we dump all n for reuse.
Run: python scripts/extract_delta_frames.py
"""
import sys, os
sys.path.insert(0, '/home/imove/Documents/cup-task/scripts')
sys.path.insert(0, '/home/imove/Documents/cup-task')
import numpy as np, cv2
import compare_pose_omc_delta as H
import results_v3_delta as R
H.use_good_cams()

PART = 'P07'
TRIALS = [f'trial_{i}_L_unaffected' for i in [10, 11, 12, 13, 14, 15]]
IMSZ = 640
OUT = '/home/imove/Documents/cup-task/cache/delta/_frames640'
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
        arr = np.empty((n, IMSZ, IMSZ, 3), np.uint8)
        got = 0
        for f in range(n):                              # sequential read = fast (no seeking)
            ok, im = cap.read()
            if not ok:
                break
            arr[f] = cv2.cvtColor(cv2.resize(im, (IMSZ, IMSZ)), cv2.COLOR_BGR2RGB)
            got = f + 1
        cap.release()
        np.save(out, arr[:got])
        print(f'  {trial} {c}: {got} frames -> {os.path.getsize(out)//1024//1024}MB', flush=True)


if __name__ == '__main__':
    for tr in TRIALS:
        print(f'{tr}...', flush=True)
        extract(PART, tr)
    print('DONE frame extraction', flush=True)
