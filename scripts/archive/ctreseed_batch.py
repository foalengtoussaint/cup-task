"""CoTracker-reseed (re-anchor from YOLO every 30fr, stays in the accurate pre-drift regime) as a 2D
keypoint smoother -> triangulate -> speed. Does its speed beat YOLO / match flow? P07+P08 vs OMC.
Reseed tracks cached in cache/pointtrack/*__ctreseed30.npy."""
import sys, numpy as np, cv2, torch
from pathlib import Path
sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import compare_pose_omc_delta as H, flow_velocity_probe as F
from pipeline import pose_smooth
from pipeline.kalman_3d import triangulate_dlt
ROOT=Path('.'); H.use_good_cams()
_M=[None]; CROP=384; RESEED=30
def M():
    if _M[0] is None: _M[0]=torch.hub.load("facebookresearch/co-tracker","cotracker3_online").cuda().eval()
    return _M[0]
def track_reseed(clip,wrist_px):
    ck=ROOT/'cache/pointtrack'/f'{clip.stem}__ctreseed30.npy'
    if ck.exists():
        v=np.load(ck)
        if len(v)==len(wrist_px): return v
    m=M(); step=m.step; T=len(wrist_px); out=np.full((T,2),np.nan); good=np.flatnonzero(np.isfinite(wrist_px).all(1))
    if len(good)<8: np.save(ck,out); return out
    cap=cv2.VideoCapture(str(clip)); frames=[]
    while True:
        ok,im=cap.read()
        if not ok: break
        frames.append(im)
    cap.release(); Ww,Hh=1920,1080; t=int(good[0])
    while t<T:
        if not np.isfinite(wrist_px[t]).all():
            nx=good[good>t]
            if len(nx)==0: break
            t=int(nx[0]); continue
        sx,sy=wrist_px[t]; we=min(T,t+RESEED)
        x0=int(np.clip(sx-CROP//2,0,Ww-CROP)); y0=int(np.clip(sy-CROP//2,0,Hh-CROP))
        cf=[frames[f][y0:y0+CROP,x0:x0+CROP,::-1] for f in range(t,we)]; L=len(cf)
        if L<=step:      # window too short to run the streaming loop -> keep YOLO for these frames
            for k in range(L):
                if np.isfinite(wrist_px[t+k]).all(): out[t+k]=wrist_px[t+k]
            t=we; continue
        q=torch.tensor([[[0.,float(sx-x0),float(sy-y0)]]]).float().cuda()
        win=[]; tr=None; first=True
        with torch.no_grad():
            for i in range(L):
                win.append(cf[i])
                if i%step==0 and i!=0:
                    ch=torch.from_numpy(np.stack(win[-2*step:])).permute(0,3,1,2)[None].float().cuda()
                    tr,_=m(video_chunk=ch,is_first_step=first,queries=q); first=False
            ch=torch.from_numpy(np.stack(win[-((L-1)%step)-step-1:])).permute(0,3,1,2)[None].float().cuda()
            tr,_=m(video_chunk=ch,is_first_step=first,queries=q)
        if tr is None:
            for k in range(L):
                if np.isfinite(wrist_px[t+k]).all(): out[t+k]=wrist_px[t+k]
            t=we; continue
        tk=tr[0,:,0].cpu().numpy(); torch.cuda.empty_cache()
        for k in range(L): out[t+k]=[tk[k,0]+x0,tk[k,1]+y0]
        t=we
    np.save(ck,out); return out
def ct_speed(part,trial,joint,cams,n,px):
    ct={}
    for cam in cams:
        clip=H.DELTA/part/'staged'/f'delta_{part}_{trial}.{cam.split("_")[1]}.mp4'
        if clip.exists() and cam in px: ct[cam]=track_reseed(clip,px[cam])
    X=np.full((n,3),np.nan)
    for f in range(n):
        obs={c:ct[c][f] for c in ct if np.isfinite(ct[c][f]).all()}
        if len(obs)>=2: X[f]=triangulate_dlt([cams[c] for c in obs],[np.array(obs[c]) for c in obs])
    return H._speed(X)
def flow_sp(part,trial,joint,cams,n):
    px=F.load_wrist_px(part,trial,joint); flow={}
    for c in px:
        f=ROOT/'cache/flow_vel'/f'delta_{part}_{trial}.{c.split("_")[1]}__pyrlk.npy'
        if f.exists() and c in cams: flow[c]=np.load(f)
    sp=np.full(n,np.nan)
    for fr in range(n):
        op,opv={},{}
        for c in flow:
            if fr<len(px[c]) and np.isfinite(px[c][fr]).all() and fr<len(flow[c]) and np.isfinite(flow[c][fr]).all():
                op[c]=px[c][fr]; opv[c]=px[c][fr]+flow[c][fr]
        if len(op)<2: continue
        Xp=triangulate_dlt([cams[c] for c in op],[np.array(op[c]) for c in op]); Xv=triangulate_dlt([cams[c] for c in opv],[np.array(opv[c]) for c in opv])
        sp[fr]=np.linalg.norm(np.array(Xv)-np.array(Xp))*60.
    return sp
def err(s,o):
    a,b=H._lp(s),H._lp(o); mk=np.isfinite(a)&np.isfinite(b); return np.median(np.abs(a[mk]-b[mk])) if mk.sum()>20 else np.nan
TR={"P07":([f"trial_{i}_L_unaffected" for i in range(10,16)],"left"),"P08":([f"trial_{i}_R_unaffected" for i in range(10,16)],"right")}
agg={k:[] for k in ['yolo','smoothnet','flow','ctreseed']}
for part,(trials,side) in TR.items():
    for trial in trials:
        joint=f"{side}_wrist"; mmc,n=H._load_mmc(part,trial); omc=H._load_omc(part,trial,n); lag,_=H._find_lag(mmc[joint],omc[joint])
        o=H._speed(F._shift(omc[joint],lag)); cams=H._load_calib_mm(part)
        if part in H.GOOD_CAMS and H.GOOD_CAMS[part]: cams={c:v for c,v in cams.items() if c in H.GOOD_CAMS[part]}
        px=F.load_wrist_px(part,trial,joint)
        snt=pose_smooth.smooth_track([{"frame":k,"X":(None if not np.isfinite(p).all() else list(p))} for k,p in enumerate(mmc[joint])])
        sp_sn=H._speed(np.array([t["X"] if t["X"] else [np.nan]*3 for t in snt]))
        agg['yolo'].append(err(H._speed(mmc[joint]),o)); agg['smoothnet'].append(err(sp_sn,o))
        agg['flow'].append(err(flow_sp(part,trial,joint,cams,n),o)); agg['ctreseed'].append(err(ct_speed(part,trial,joint,cams,n,px),o))
        print(f"  {part}_{trial.split('_')[1]}: yolo {agg['yolo'][-1]:.0f} sn {agg['smoothnet'][-1]:.0f} flow {agg['flow'][-1]:.0f} ct {agg['ctreseed'][-1]:.0f}",flush=True)
print(f"\n{'method':10} {'speed |err| mm/s':>16}")
for k in ['yolo','smoothnet','flow','ctreseed']: print(f"{k:10} {np.median(agg[k]):14.1f}")
