"""Cache YOLO's per-frame per-axis wrist SIGMA (+ wrist px in NATIVE resolution) for the DELTA
speed-metric trials, so the sigma-DLT / sigma-RTS ablations run CPU-only afterward.

Rigor: we re-project the sigma-patched YOLO wrist from 640-letterbox back to NATIVE px, so the sigma
track triangulates with the IDENTICAL native 2D + _load_calib_mm P as triangulate_target -> the ONLY
difference from the standard jitter-DLT is the per-axis sigma weighting (clean ablation). We also cache
the native wrist px itself so sigma-DLT and jitter-DLT consume the same 2D.

Cache: cache/wrist_sigma/<part>_<trial>.npz with, per good cam:
  <cam>_uv  (n,2) native px wrist (nan where undetected/below conf)
  <cam>_sig (n,2) per-axis sigma in (0,1)   (nan where undetected)
One GPU pass over 12 trials (~10 min). Reuses the loader's letterbox params for the inverse map.
"""
import sys; sys.path.insert(0,'/home/imove/Documents/cup-task/scripts'); sys.path.insert(0,'/home/imove/Documents/cup-task')
import os, numpy as np, torch, cv2
from pathlib import Path
from ultralytics import YOLO
import compare_pose_omc_delta as H
from pipeline.mv3d.yolo_sigma import enable_sigma, split_sigma
from pipeline.mv3d.imgproc import letterbox_image, letterbox_params
H.use_good_cams()
dev='cuda'; IMSZ=640
COCO17=['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder',
        'left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip',
        'left_knee','right_knee','left_ankle','right_ankle']
OUT=Path('/home/imove/Documents/cup-task/cache/wrist_sigma'); OUT.mkdir(parents=True,exist_ok=True)
y=YOLO('models/yolo26s-pose.pt'); m=y.model.to(dev).eval(); nk=m.model[-1].nk; enable_sigma(y)

TRIALS={'P07':([f'trial_{t}_L_unaffected' for t in range(10,16)],'left'),
        'P08':([f'trial_{t}_R_unaffected' for t in range(10,16)],'right')}

def sigma_native(bgr, wr, W, H_):
    """native BGR frame -> (wrist_native_px (2,), sigma (2,)) or (nan,nan)."""
    canvas,r,dw,dh = letterbox_image(bgr, IMSZ)
    t=torch.from_numpy(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB)).permute(2,0,1).float()[None].to(dev)/255.
    with torch.no_grad():
        raw=m(t); raw=raw[0] if isinstance(raw,(tuple,list)) else raw
    det=raw[0]; c=det[:,4]
    if c.max()<0.25: return np.array([np.nan,np.nan]),np.array([np.nan,np.nan])
    b=c.argmax(); k,s=split_sigma(det,nk,34)
    kxy=k[b].view(17,3)[wr,:2].cpu().numpy()         # 640-letterbox px
    sxy=s[b].view(17,2)[wr].cpu().numpy()
    if not np.isfinite(kxy).all(): return np.array([np.nan,np.nan]),np.array([np.nan,np.nan])
    # inverse letterbox: native = (lb - pad)/scale
    nx=(kxy[0]-dw)/r; ny=(kxy[1]-dh)/r
    return np.array([nx,ny]), sxy

for part,(trials,side) in TRIALS.items():
    wr=COCO17.index(f'{side}_wrist')
    for trial in trials:
        outp=OUT/f'{part}_{trial}.npz'
        if outp.exists():
            print(f"{part} {trial}: cached, skip",flush=True); continue
        d=H.DELTA/part
        good=H.GOOD_CAMS.get(part) if H.GOOD_CAMS else None
        vids={}
        import glob as _g
        for pj in sorted(_g.glob(str(d/H.DETS_SUBDIR/f"*{trial}*.pose.json"))):
            cam='cam_'+Path(pj).name.split('.')[1]
            if good and cam not in good: continue
            mp=d/'staged'/f'delta_{part}_{trial}.{cam.split("_")[1]}.mp4'
            if mp.exists(): vids[cam]=str(mp)
        if not vids:
            print(f"{part} {trial}: no videos",flush=True); continue
        store={}
        for cam,mp in vids.items():
            cap=cv2.VideoCapture(mp); frames=[]
            while True:
                ok,im=cap.read()
                if not ok: break
                frames.append(im)
            cap.release()
            W,Hh=frames[0].shape[1],frames[0].shape[0]
            uv=np.full((len(frames),2),np.nan); sg=np.full((len(frames),2),np.nan)
            for i,im in enumerate(frames):
                u,s=sigma_native(im,wr,W,Hh); uv[i]=u; sg[i]=s
            store[f'{cam}_uv']=uv; store[f'{cam}_sig']=sg
            print(f"  {part} {trial} {cam}: {len(frames)}fr det {np.isfinite(uv[:,0]).mean()*100:.0f}%",flush=True)
        np.savez(outp,**store)
        print(f"{part} {trial}: SAVED {outp.name}",flush=True)
print("done",flush=True)
