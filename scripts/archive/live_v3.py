"""LIVE v3 rig: batched pose + detect-once UETrack cup + per-frame 3D, with the speed levers ON.

The old live loop (object_tracking/live_track.py) is a DIFFERENT pipeline: YOLO-seg cup EVERY frame,
one predict PER CAMERA (5 cams = 10 single-image forwards per cycle), fp32. This one runs the v3
pipeline that the offline results use -- YOLO-pose + detect-once UETrack -- and applies every lever
measured in out/speed/:

  L1 BATCH across cameras   ONE pose forward + ONE UETrack forward per rig-frame.
                            MEASURED: pose 5cam 12.42ms batched vs 5x4.02=20.1ms sequential.
  L2 fp16 on both nets      pose: build+fuse fp32 then cast the AutoBackend (predict(half=True) is a
                            NO-OP on a .pt in ultralytics 8.4.49). UETrack: autocast (verified to
                            actually run fp16 -- conv_out dtype checked, not assumed).
                            MEASURED: pose infer 7.65->5.04ms, UETrack 10.96->7.51ms at 5 cams.
  L3 kill the CPU tax       (a) GPU letterbox: MEASURED A LOSS, now opt-in (--gpu-letterbox). Doing
                            the resize on device means uploading the full 1080p batch (~31MB/frame at
                            5 cams) vs the 640x384 tensor ultralytics uploads after its CPU resize
                            (~1.1MB): pose 12.70->17.49ms fp32, 10.62->13.06ms fp16.
                            (b) crop-then-convert for UETrack (update(bgr=True)): convert five
                            224x224 crops, not five 1080p frames (~3.1ms -> ~0.05ms).
                            (c) optional torch.compile on both forwards (--compile).

The BGR->RGB lesson is baked in: a numpy `[:, :, ::-1].copy()` costs 32.5ms per 5-cam rig-frame and
was 25ms of what looked like "UETrack is the bottleneck". Never do the flip on a full frame.

SOURCES (so this is testable with no rig attached):
    --source replay --part P07 --trial trial_10_L_unaffected     staged clips, wall-clock free
    --source v4l2 --devices 0 4 8 12                             cv2.VideoCapture per camera
    --source zmq                                                 cam_server streams (object_tracking)

    python scripts/live_v3.py --source replay --part P07 --frames 300
    python scripts/live_v3.py --source replay --part P07 --no-fp16 --no-crop-convert  # ablate
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

POSE_W = ROOT / "models/yolo26n-pose.pt"
CUP_W = ROOT / "models/cup_clean3d_refill.pt"
IMGSZ = 640
COCO17 = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
          "left_wrist", "right_wrist", "left_hip", "right_hip",
          "left_knee", "right_knee", "left_ankle", "right_ankle"]
JOINTS = ["right_wrist", "right_elbow", "right_shoulder",
          "left_wrist", "left_elbow", "left_shoulder", "nose"]


# ----------------------------------------------------------------------------- sources
class ReplaySource:
    """Staged DELTA clips as a stand-in rig: yields one BGR frame per camera per tick."""

    def __init__(self, part, trial=None):
        import cache_uetrack_tracks as CU
        import gpu_decode
        d = CU._staged_dir(part)
        if trial is None:
            trial = sorted({Path(p).name.split(".")[0].replace(f"delta_{part}_", "")
                            for p in glob.glob(str(d / f"delta_{part}_*.mp4"))})[0]
        paths = sorted(glob.glob(str(d / f"delta_{part}_{trial}.*.mp4")))
        self.cams = [f"cam_{Path(p).name.split('.')[1]}" for p in paths]
        self.gens = [gpu_decode.frames(p) for p in paths]
        self.part, self.trial = part, trial

    def read(self):
        out = [next(g, None) for g in self.gens]
        return None if any(o is None for o in out) else out

    def close(self):
        pass


class V4L2Source:
    def __init__(self, devices, width=1920, height=1080):
        import cv2
        self.caps = []
        for d in devices:
            c = cv2.VideoCapture(int(d))
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            c.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.caps.append(c)
        self.cams = [f"cam_{i}" for i in range(len(devices))]

    def read(self):
        out = []
        for c in self.caps:
            ok, f = c.read()
            if not ok:
                return None
            out.append(f)
        return out

    def close(self):
        for c in self.caps:
            c.release()


class ZmqSource:
    """cam_server streams. Lives in the object_tracking repo, imported lazily/optionally."""

    def __init__(self, host="127.0.0.1", config=None, ot_root="/home/imove/Documents/object_tracking"):
        sys.path.insert(0, ot_root)
        import zmq
        from recording.cam_streams import load_streams
        streams = load_streams(config)
        ctx = zmq.Context()
        self.socks = {}
        for name, port in streams.items():
            s = ctx.socket(zmq.SUB)
            s.connect(f"tcp://{host}:{port}")
            s.setsockopt_string(zmq.SUBSCRIBE, "")
            s.setsockopt(zmq.CONFLATE, 1)          # live: newest frame only, never a backlog
            self.socks[name] = s
        self.cams = list(self.socks)
        self._zmq = zmq

    def read(self):
        import cv2
        out = []
        for s in self.socks.values():
            buf = s.recv()
            arr = np.frombuffer(buf, np.uint8)
            f = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if f is None:
                return None
            out.append(f)
        return out

    def close(self):
        for s in self.socks.values():
            s.close()


# ----------------------------------------------------------------------------- levers
def gpu_letterbox(frames_bgr, imgsz, dtype):
    """L3a: BGR uint8 frames -> (N,3,H,W) RGB float tensor on GPU, resize+convert+normalise on device.

    Pads BOTTOM/RIGHT only (not centred like ultralytics) so inverting is a plain divide by `s` with
    no offset -- fewer chances to put a keypoint in the wrong place. Returns (tensor, s).
    """
    import torch
    import torch.nn.functional as F
    h, w = frames_bgr[0].shape[:2]
    x = torch.from_numpy(np.ascontiguousarray(np.stack(frames_bgr))).cuda(non_blocking=True)
    x = x.permute(0, 3, 1, 2).flip(1)                     # BGR->RGB on device
    x = x.to(dtype).div_(255.0)
    s = imgsz / max(h, w)
    nh, nw = int(round(h * s)), int(round(w * s))
    x = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    ph, pw = (-nh) % 32, (-nw) % 32                       # stride-32 pad, bottom/right
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), value=114 / 255.0)
    return x, s


def pose_model(fp16, warm, compile_it=False):
    import torch
    from ultralytics import YOLO
    m = YOLO(str(POSE_W)).to("cuda:0")
    m.predict(warm, imgsz=IMGSZ, device=0, verbose=False)          # build + fuse in fp32
    if fp16:
        m.predictor.model.model.half()
        m.predictor.model.fp16 = True
        assert next(m.predictor.model.model.parameters()).dtype == torch.float16
    if compile_it:
        m.predictor.model.model = torch.compile(m.predictor.model.model,
                                                mode="reduce-overhead")
    return m


def cup_seeder(fp16, warm):
    import torch
    from ultralytics import YOLO
    m = YOLO(str(CUP_W)).to("cuda:0")
    m.predict(warm, imgsz=IMGSZ, device=0, verbose=False)
    if fp16:
        m.predictor.model.model.half()
        m.predictor.model.fp16 = True
        assert next(m.predictor.model.model.parameters()).dtype == torch.float16
    return m


# ----------------------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["replay", "v4l2", "zmq"], default="replay")
    ap.add_argument("--part", default="P07")
    ap.add_argument("--trial", default=None)
    ap.add_argument("--devices", nargs="+", default=["0", "4", "8", "12"])
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--gpu-letterbox", action="store_true",
                    help="MEASURED LOSS -- opt-in only. Resizing on device means uploading the full "
                         "1080p uint8 batch (~31MB/rig-frame at 5 cams) instead of the 640x384 "
                         "tensor ultralytics uploads after its CPU resize (~1.1MB). Costs +4.8ms "
                         "on pose in fp32 (12.70->17.49ms), +2.4ms in fp16. Kept for the record.")
    ap.add_argument("--no-crop-convert", action="store_true",
                    help="ablate L3b (convert 224x224 crops instead of full frames)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-batch", action="store_true",
                    help="ablation: one forward PER CAMERA, like the old live_track.py")
    ap.add_argument("--save", type=Path, default=None, help="npz of the 3D tracks")
    a = ap.parse_args()

    import cv2
    import torch
    from pipeline import triangulate as T
    from pipeline.consensus import consensus3
    from pipeline.pose_keypoints import MIN_KP_CONF
    from uetrack_wrap import UETrackBatch
    import compare_pose_omc_delta as H
    import results_v3_delta as R

    fp16 = not a.no_fp16
    dtype = torch.float16 if fp16 else torch.float32
    cv2.setNumThreads(4)

    if a.source == "replay":
        src = ReplaySource(a.part, a.trial)
        H.use_good_cams()
        calib_all = R._calib(a.part)
    elif a.source == "v4l2":
        src = V4L2Source(a.devices)
        calib_all = R._calib(a.part)
    else:
        src = ZmqSource()
        calib_all = R._calib(a.part)
    cams = [c for c in src.cams if c in calib_all]
    if len(cams) < 3:
        raise SystemExit(f"need >=3 calibrated cams, got {cams} (calib has {sorted(calib_all)})")
    idx = [src.cams.index(c) for c in cams]
    calib = {c: calib_all[c] for c in cams}
    print(f"live_v3: source={a.source} cams={cams}  fp16={fp16} gpu_letterbox="
          f"{a.gpu_letterbox} crop_convert={not a.no_crop_convert} compile={a.compile} "
          f"batch={not a.no_batch}", flush=True)

    warm = [np.zeros((64, 64, 3), np.uint8)]
    pm = pose_model(fp16, warm, a.compile)
    cm = cup_seeder(fp16, warm)
    btrk = UETrackBatch(len(cams))
    if a.compile:
        btrk.net.forward_encoder = torch.compile(btrk.net.forward_encoder)
        btrk.net.forward_decoder = torch.compile(btrk.net.forward_decoder)
    ac = (torch.autocast("cuda", dtype=torch.float16) if fp16
          else torch.autocast("cuda", enabled=False))

    seeded = set()
    st = dict(grab=0.0, pose=0.0, xfer=0.0, cup=0.0, tri=0.0)
    joints3d = {j: [] for j in JOINTS}
    cup3d = []
    prev = None
    f = 0
    t_start = time.time()
    fps_win = []

    while f < a.frames:
        t0 = time.perf_counter()
        got = src.read()
        if got is None:
            break
        batch = [got[i] for i in idx]
        st["grab"] += time.perf_counter() - t0

        # ---------------- pose (L1 batch + L2 fp16 + L3a gpu letterbox) ----------------
        t0 = time.perf_counter()
        if (not a.gpu_letterbox):
            rs = (pm.predict(batch, imgsz=IMGSZ, device=0, verbose=False) if not a.no_batch
                  else [pm.predict([b], imgsz=IMGSZ, device=0, verbose=False)[0] for b in batch])
            scale = None
        else:
            if a.no_batch:
                rs = []
                for b in batch:
                    x, scale = gpu_letterbox([b], IMGSZ, dtype)
                    rs.append(pm.predict(x, imgsz=IMGSZ, device=0, verbose=False)[0])
            else:
                x, scale = gpu_letterbox(batch, IMGSZ, dtype)
                rs = pm.predict(x, imgsz=IMGSZ, device=0, verbose=False)
        torch.cuda.synchronize()
        st["pose"] += time.perf_counter() - t0

        # ONE GPU->CPU transfer for the whole rig-frame. The obvious per-camera version --
        # int(r.boxes.conf.argmax()) + .cpu() on xy + .cpu() on conf -- is THREE device syncs per
        # camera (15 per rig-frame at 5 cams), each stalling the pipeline immediately after the
        # batched forward, i.e. re-serialising the batch we just built. uetrack_wrap.py:140
        # documents the same trap costing ~50% of its per-frame CPU time. Index with the argmax
        # TENSOR (no sync), stack, and pull xy+conf across in a single cat.
        t0x = time.perf_counter()
        kxy_l, kcf_l, have = [], [], []
        for r in rs:
            if r.boxes is not None and len(r.boxes) and r.keypoints is not None:
                i = r.boxes.conf.argmax()                     # stays on device
                kxy_l.append(r.keypoints.xy[i].float())
                kcf_l.append(r.keypoints.conf[i].float())
                have.append(True)
            else:
                have.append(False)
        if kxy_l:
            KXY = torch.stack(kxy_l)                          # (n,17,2)
            KCF = torch.stack(kcf_l)                          # (n,17)
            flat = torch.cat([KXY.reshape(len(kxy_l), -1), KCF], 1).cpu().numpy()
            KXY = flat[:, :34].reshape(-1, 17, 2)
            KCF = flat[:, 34:]
        per_cam_kps = {}
        k = 0
        for ci, c in enumerate(cams):
            kps = {}
            if have[ci]:
                kxy, kcf = KXY[k], KCF[k]
                k += 1
                if scale is not None:
                    kxy = kxy / scale      # undo the GPU letterbox (bottom/right pad => no offset)
                for j, nm in enumerate(COCO17):
                    if nm in JOINTS and kcf[j] >= MIN_KP_CONF:
                        kps[nm] = np.array([kxy[j, 0], kxy[j, 1]], float)
            per_cam_kps[c] = kps
        st["xfer"] += time.perf_counter() - t0x

        # ---------------- cup: detect-once seed, then batched track (L1+L2+L3b) ----------------
        t0 = time.perf_counter()
        if len(seeded) < len(cams):
            need = [i for i in range(len(cams)) if i not in seeded]
            dets = cm.predict([batch[i] for i in need], imgsz=IMGSZ, device=0, verbose=False)
            for i, r in zip(need, dets):
                if r.boxes is not None and len(r.boxes):
                    k = int(r.boxes.conf.argmax())
                    x1, y1, x2, y2 = r.boxes.xyxy[k].float().cpu().numpy()
                    rgb = cv2.cvtColor(batch[i], cv2.COLOR_BGR2RGB)   # once per camera, at seed
                    btrk.init(i, rgb, [x1, y1, x2 - x1, y2 - y1])
                    seeded.add(i)
        if seeded:
            if a.no_crop_convert:
                rgbs = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in batch]
                with ac:
                    boxes = btrk.update([rgbs[i] if i in seeded else None
                                         for i in range(len(cams))])
            else:
                with ac:
                    boxes = btrk.update([batch[i] if i in seeded else None
                                         for i in range(len(cams))], bgr=True)
        else:
            boxes = [None] * len(cams)
        torch.cuda.synchronize()
        st["cup"] += time.perf_counter() - t0

        # ---------------- 3D, this frame only (live: no future frames) ----------------
        t0 = time.perf_counter()
        for j in JOINTS:
            obs = {c: per_cam_kps[c][j] for c in cams if j in per_cam_kps[c]}
            if len(obs) >= 2:
                ks = list(obs)
                X, kept, rp = T.robust_triangulate([calib[c] for c in ks], [obs[c] for c in ks])
                joints3d[j].append(X if X is not None else [np.nan] * 3)
            else:
                joints3d[j].append([np.nan] * 3)
        cobs = {}
        for i, c in enumerate(cams):
            b = boxes[i]
            if b is not None:
                cobs[c] = (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)
        if len(cobs) >= 2:
            X, kept, _ = consensus3(cobs, calib, prev=prev)
            cup3d.append(X if X is not None else [np.nan] * 3)
            if X is not None:
                prev = X
        else:
            cup3d.append([np.nan] * 3)
        st["tri"] += time.perf_counter() - t0

        f += 1
        fps_win.append(time.perf_counter())
        if f % 30 == 0:
            w = fps_win[-30:]
            inst = 29.0 / (w[-1] - w[0])
            print(f"  [{f:4d}] {inst:5.1f} rig-fps inst | mean {f/(time.time()-t_start):5.1f} | "
                  f"grab {st['grab']/f*1e3:5.2f} pose {st['pose']/f*1e3:5.2f} "
                  f"xfer {st['xfer']/f*1e3:4.2f} cup {st['cup']/f*1e3:5.2f} tri {st['tri']/f*1e3:5.2f} | "
                  f"seeded {len(seeded)}/{len(cams)}", flush=True)

    src.close()
    dur = time.time() - t_start
    tot_ms = sum(st.values()) / max(f, 1) * 1e3
    print(f"\n===== live_v3: {f} rig-frames, {len(cams)} cams, {dur:.1f}s =====", flush=True)
    print(f"  {'stage':10s}{'ms/frame':>10}{'% of frame':>12}", flush=True)
    for k in ("grab", "pose", "xfer", "cup", "tri"):
        ms = st[k] / max(f, 1) * 1e3
        print(f"  {k:10s}{ms:10.2f}{ms/tot_ms*100:11.0f}%", flush=True)
    print(f"  {'TOTAL':10s}{tot_ms:10.2f}   -> {1000/tot_ms:5.1f} rig-fps "
          f"({1000/max(tot_ms-st['grab']/max(f,1)*1e3, 1e-9):5.1f} excluding frame-grab)", flush=True)

    cov = {j: float(np.mean(np.isfinite(np.array(joints3d[j])).all(1))) for j in JOINTS}
    cupa = np.array(cup3d, float)
    print(f"\n  3D coverage: cup {np.mean(np.isfinite(cupa).all(1)):.0%}, "
          + ", ".join(f"{j.split('_')[-1]} {cov[j]:.0%}" for j in JOINTS), flush=True)
    print(f"  PROCESSING CHECK: frames {f}/{a.frames}, seeded {len(seeded)}/{len(cams)}, "
          f"cup non-finite {int(np.sum(~np.isfinite(cupa).all(1)))}", flush=True)

    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.save, cup=cupa, **{f"j_{j}": np.array(joints3d[j], float) for j in JOINTS},
                 timings=np.array(str({k: v / max(f, 1) for k, v in st.items()})))
        print(f"  wrote {a.save}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
