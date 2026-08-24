"""Diagnostic overlay: on ONE camera's clip, draw the YOLO wrist keypoint vs the reprojected 3-cam
consensus (pipeline) 3D wrist, per frame, to SEE the confidently-wrong detection.

  GREEN  = YOLO detected wrist (where the detector says it is), radius scaled, conf printed
  CYAN   = reprojected pipeline 3D wrist (where 3-cam consensus says it should be)
  line   = the gap; RED if reproj residual > 30px (bad keypoint), else yellow
  HUD    = frame, residual px, YOLO conf, n cams

Usage: overlay_wrist_diag.py --part P19 --trial trial_63_R_affected --cam cam_3
"""
import sys, argparse, json, glob
from pathlib import Path
import numpy as np, cv2, torch
sys.path.insert(0, "/home/imove/Documents/cup-task/scripts"); sys.path.insert(0, "/home/imove/Documents/cup-task")
import gnn_refiner as G, gnn_train as T
JOINTS = G.JOINTS
STAGED = "/home/imove/Documents/cup-task/cache/delta/{p}/staged/delta_{p}_{tr}.{n}.mp4"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="P19"); ap.add_argument("--trial", default="trial_63_R_affected")
    ap.add_argument("--cam", default="cam_3")
    a = ap.parse_args()
    t = [x for x in T.load_clean(need_reproj=True) if x["part"] == a.part and x["trial"] == a.trial][0]
    side = t["side"]; wi = JOINTS.index(f"{side}_wrist")
    cams = list(np.load(f"/home/imove/Documents/cup-task/cache/delta/gnn_pairs/{a.part}/{a.trial}.reproj.npz",
                        allow_pickle=True)["cams"])
    ci = [str(c) for c in cams].index(a.cam)
    camnum = a.cam.split("_")[1]
    clip = STAGED.format(p=a.part, tr=a.trial, n=camnum)
    print(f"cam {a.cam} = sidecar idx {ci}; clip {clip}", flush=True)
    # reproject the pipeline 3D wrist into this camera, every frame
    K, dist, R, tt = t["K"][ci], t["dist"][ci], t["R"][ci], t["t"][ci]
    X = torch.from_numpy(np.nan_to_num(t["mmc"][:, wi])[None, :, None].astype(np.float32))
    uvp, _ = G.project_torch(X, torch.from_numpy(K[None].astype(np.float32)),
                             torch.from_numpy(dist[None].astype(np.float32)),
                             torch.from_numpy(R[None].astype(np.float32)),
                             torch.from_numpy(tt[None].astype(np.float32)))
    reproj = uvp[0, :, 0].numpy()                                  # (T,2) consensus wrist in this cam
    yolo = t["uv"][:, ci, wi]                                      # (T,2) detected wrist
    conf = t["uv_conf"][:, ci, wi]; validc = t["uv_valid"][:, ci, wi]
    resid = np.linalg.norm(reproj - yolo, axis=1)
    cap = cv2.VideoCapture(clip)
    W = int(cap.get(3)); Hh = int(cap.get(4)); n = int(cap.get(7))
    out = f"/home/imove/Documents/cup-task/out/gnn/overlay_{a.part}_{a.trial}_{a.cam}.mp4"
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, Hh))
    f = 0
    while True:
        ok, im = cap.read()
        if not ok or f >= len(yolo):
            break
        y = yolo[f]; rp = reproj[f]; r = resid[f]
        if validc[f] and np.isfinite(y).all():
            bad = r > 30
            cv2.line(im, tuple(y.astype(int)), tuple(rp.astype(int)),
                     (0, 0, 255) if bad else (0, 255, 255), 2)
            cv2.circle(im, tuple(y.astype(int)), 9, (0, 255, 0), 2)          # YOLO = green
            cv2.circle(im, tuple(rp.astype(int)), 7, (255, 255, 0), -1)      # consensus = cyan
            cv2.putText(im, f"conf {conf[f]:.2f}", tuple((y + [12, 0]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            col = (0, 0, 255) if bad else (0, 255, 0)
            cv2.putText(im, f"f{f}  resid {r:4.0f}px", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        else:
            cv2.putText(im, f"f{f}  wrist not detected", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 165, 255), 2)
        vw.write(im); f += 1
    cap.release(); vw.release()
    m = validc & np.isfinite(resid)
    print(f"wrote {out}  ({f} frames)", flush=True)
    print(f"resid on this cam: median {np.median(resid[m]):.0f}px  frac>30px {100*np.mean(resid[m]>30):.0f}%",
          flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
