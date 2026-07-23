"""Render the staggered cloud tracker on a video, annotated so it ties back to the session's findings.

WHAT YOU SEE (one camera view + info panel), and which finding each maps to:
  * DOTS = the per-camera PyrLK tracks (the 2D tracking that is the real state; 3D is rebuilt each
    frame). Colour = track AGE: red=young(<10) -> yellow -> green -> blue=old(>=40). This is the
    SETTLING / maturity effect -- young (red) tracks jitter, old (blue) are clean. Staggered
    generations mean all ages coexist.
  * DOT RING: filled = RANSAC inlier (used in the fit) ; hollow = rejected. RANSAC is the implicit
    quality filter (rejects young/jittery tracks 20x more).
  * WHITE CIRCLE = the detected keypoint; dots outside the 30px ANCHOR are culled (you see them
    vanish).
  * CYAN cross = the tracked cloud CENTROID reprojected (what the speed is measured from).
  * GREEN arrow = the temporal-informed ROTATION AXIS (the jitter-separated real spin).
  * PANEL: cohort age (frames since reseed), n tracks / inliers, linear speed (median-step) vs OMC,
    angular speed, and a rolling speed trace vs OMC truth.

    python scripts/render_cloud_track.py --part P07 --trial trial_11_L_unaffected --cam cam_3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="P07")
    ap.add_argument("--trial", default="trial_11_L_unaffected")
    ap.add_argument("--cam", default="cam_3", help="which camera view to draw")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import cv2
    import compare_pose_omc_delta as H
    import results_v3_delta as R
    import cup_flow_probe as C
    import cloud_rotation_truth as CR
    from cup_task.cloud_track import surface_seed, _project, _visible
    from cup_task.cloud_velocity import kabsch_ransac
    from cup_task.kalman_3d import triangulate_dlt
    from scipy.spatial.transform import Rotation

    H.use_good_cams()
    LK = dict(winSize=(21, 21), maxLevel=3,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    FB = 1.0
    part, trial, drawcam = a.part, a.trial, a.cam
    calib = R._calib(part)
    side = "left" if part == "P07" else "right"

    mmc, n = H._load_mmc(part, trial)
    omc = H._load_omc(part, trial, n)
    lag, _ = H._find_lag(mmc[f"{side}_wrist"], omc[f"{side}_wrist"])
    so = H._lp(H._speed(R._shift(R._omc_cup(part, trial, n), lag)))          # OMC cup speed truth
    mk_, _ = CR.omc_cup_markers(part, trial, n, 60.)
    wo = H._lp(CR.omc_angular_speed(mk_, 60.)) if mk_ is not None else np.full(n, np.nan)
    px = C.cup_px(part, trial, n)
    cup3 = R._smooth_joint(R._cup_v3(part, trial, calib, n))
    cams = [c for c in px if c in calib]

    caps = {c: cv2.VideoCapture(str(H.DELTA / part / "staged"
                                    / f"delta_{part}_{trial}.{c.split('_')[1]}.mp4")) for c in cams}
    W = int(caps[drawcam].get(cv2.CAP_PROP_FRAME_WIDTH))
    Hh = int(caps[drawcam].get(cv2.CAP_PROP_FRAME_HEIGHT))
    PANEL = 320
    out = a.out or str(ROOT / "out" / f"cloud_track_{part}_{trial}_{drawcam}.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W + PANEL, Hh))

    def age_color(age):
        # red(young) -> yellow -> green -> blue(old)
        t = min(age / 40.0, 1.0)
        if t < 0.33:   # red->yellow
            f = t / 0.33; return (0, int(255 * f), 255)
        if t < 0.66:   # yellow->green
            f = (t - 0.33) / 0.33; return (0, 255, int(255 * (1 - f)))
        f = (t - 0.66) / 0.34; return (int(255 * f), int(255 * (1 - f)), 0)  # green->blue

    PXs = {c: {} for c in cams}; born = {}; pg = {}; nid = 0; prev3 = {}
    cohort_age = 0
    Vhist = []                      # rotation vectors for the temporal axis
    slin = np.full(n, np.nan); sang = np.full(n, np.nan)
    trace = []                      # (frame, tracker_speed, omc_speed) for the rolling plot

    for f in range(n):
        gray = {}; frame_bgr = None
        for c, cap in caps.items():
            ok, im = cap.read()
            if ok:
                gray[c] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                if c == drawcam:
                    frame_bgr = im
        if frame_bgr is None:
            break
        if len(gray) < 2 or not np.isfinite(cup3[f]).all():
            pg = gray
            canvas = np.zeros((Hh, W + PANEL, 3), np.uint8); canvas[:, :W] = frame_bgr
            vw.write(canvas); continue

        # advance PyrLK
        for c in cams:
            if c not in gray or c not in pg or not PXs[c]:
                continue
            ids = list(PXs[c]); p0 = np.array([PXs[c][i] for i in ids], np.float32).reshape(-1, 1, 2)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(pg[c], gray[c], p0, None, **LK)
            p0b, st2, _ = cv2.calcOpticalFlowPyrLK(gray[c], pg[c], p1, None, **LK)
            fb = np.linalg.norm(p0.reshape(-1, 2) - p0b.reshape(-1, 2), axis=1)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < FB)
            PXs[c] = {i: p1[j, 0] for j, i in enumerate(ids) if good[j]}
        # anchor cull
        for c in cams:
            kp = px[c][f] if f < len(px[c]) and np.isfinite(px[c][f]).all() else None
            if kp is not None:
                PXs[c] = {i: p for i, p in PXs[c].items() if np.linalg.norm(p - kp) <= 30.}
        cohort_age += 1
        # staggered reseed every 10 frames (never full reset here -> ages coexist)
        if f % 10 == 0:
            for X in surface_seed(cup3[f], 40., 95., 24, seed=f):
                iid = nid; nid += 1; born[iid] = f
                for c in cams:
                    if not _visible(calib[c], X, cup3[f]):
                        continue
                    u = _project(calib[c], X)
                    if u is not None and 5 <= u[0] < gray[c].shape[1] - 5 and 5 <= u[1] < gray[c].shape[0] - 5:
                        PXs[c][iid] = u
        # lift
        cur3 = {}
        for i in set().union(*[set(PXs[c]) for c in cams]):
            obs = [(c, PXs[c][i]) for c in cams if i in PXs[c]]
            if len(obs) < 2:
                continue
            X = triangulate_dlt([calib[c] for c, _ in obs], [np.asarray(p) for _, p in obs])
            if X is not None and np.isfinite(X).all():
                cur3[i] = X
        common = [i for i in cur3 if i in prev3 and (f - born[i]) >= 10]
        inl_ids = set()
        cen = None
        if len(common) >= 8:
            A = np.array([prev3[i] for i in common]); B = np.array([cur3[i] for i in common])
            Rr, tt, mk = kabsch_ransac(A, B, 5.)
            if Rr is not None and mk.sum() >= 6:
                inl_ids = {common[j] for j in np.flatnonzero(mk)}
                slin[f] = np.median(np.linalg.norm(B[mk] - A[mk], axis=1)) * 60          # median-step
                Vhist.append(Rotation.from_matrix(Rr).as_rotvec() * 60)
                cen = B[mk].mean(0)                                                       # centroid 3D
        prev3 = cur3; pg = gray

        # temporal-informed rotation axis + magnitude
        axis2d = None; angval = np.nan
        if len(Vhist) >= 1:
            seg = np.array(Vhist[-5:]); ax = seg.mean(0); nax = np.linalg.norm(ax)
            if nax > 1e-9 and np.isfinite(Vhist[-1]).all():
                angval = abs(Vhist[-1] @ (ax / nax)); sang[f] = angval
                if cen is not None:
                    tip = _project(calib[drawcam], cen + ax / nax * 60)
                    base = _project(calib[drawcam], cen)
                    if tip is not None and base is not None:
                        axis2d = (base, tip)

        # ---- DRAW ----
        canvas = np.zeros((Hh, W + PANEL, 3), np.uint8)
        canvas[:, :W] = frame_bgr
        # anchor keypoint
        kp = px[drawcam][f] if f < len(px[drawcam]) and np.isfinite(px[drawcam][f]).all() else None
        if kp is not None:
            cv2.circle(canvas, (int(kp[0]), int(kp[1])), 30, (255, 255, 255), 1)
            cv2.circle(canvas, (int(kp[0]), int(kp[1])), 3, (255, 255, 255), -1)
        # tracks in this camera, colour by age, filled=inlier
        for i, p in PXs[drawcam].items():
            age = f - born.get(i, f)
            col = age_color(age)
            xy = (int(p[0]), int(p[1]))
            if i in inl_ids:
                cv2.circle(canvas, xy, 4, col, -1)
            else:
                cv2.circle(canvas, xy, 4, col, 1)
        # centroid + rotation axis
        if cen is not None:
            bc = _project(calib[drawcam], cen)
            if bc is not None:
                cv2.drawMarker(canvas, (int(bc[0]), int(bc[1])), (255, 255, 0),
                               cv2.MARKER_CROSS, 18, 2)
        if axis2d is not None:
            (bx, by), (tx, ty) = axis2d
            cv2.arrowedLine(canvas, (int(bx), int(by)), (int(tx), int(ty)), (0, 255, 0), 2, tipLength=0.3)

        # ---- PANEL ----
        pnl = canvas[:, W:]
        pnl[:] = (25, 25, 25)
        y = 30
        def line(txt, c=(230, 230, 230), dy=26, sz=0.5):
            nonlocal y
            cv2.putText(pnl, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, sz, c, 1, cv2.LINE_AA); y += dy
        line(f"{part} {trial.split('_')[1]}  {drawcam}", (120, 200, 255), 30, 0.6)
        line(f"frame {f}/{n}")
        line(f"cohort age (since reseed): {cohort_age}", (150, 220, 150))
        ntr = len(PXs[drawcam]); nin = len(inl_ids)
        line(f"tracks(view) {ntr}   inliers(3D) {nin}")
        y += 6
        line("SPEED  (mm/s)", (200, 200, 120), 24, 0.55)
        tv = slin[f] if np.isfinite(slin[f]) else float('nan')
        ov = so[f] if np.isfinite(so[f]) else float('nan')
        line(f"  tracker (median-step): {tv:6.0f}", (0, 255, 255))
        line(f"  OMC truth           : {ov:6.0f}", (0, 200, 0))
        y += 6
        line("ROTATION  (rad/s)", (200, 200, 120), 24, 0.55)
        line(f"  tracker(temporal-ax): {angval:5.2f}", (0, 255, 0))
        line(f"  OMC truth           : {wo[f] if np.isfinite(wo[f]) else float('nan'):5.2f}", (0, 200, 0))
        y += 6
        line("track AGE colour:", (200, 200, 120), 22, 0.5)
        for lab, ag in (("young <10", 3), ("mid", 22), ("old >=40", 45)):
            cv2.circle(pnl, (22, y - 5), 5, age_color(ag), -1)
            cv2.putText(pnl, lab, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
            y += 20
        cv2.putText(pnl, "filled=RANSAC inlier  hollow=rejected", (12, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA); y += 24
        cv2.putText(pnl, "cyan +=cloud centroid  green->=spin axis", (12, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA); y += 26

        # rolling speed trace
        trace.append((tv, ov))
        th = 110; tw = PANEL - 24; ty0 = Hh - th - 12
        cv2.rectangle(pnl, (12, ty0), (12 + tw, ty0 + th), (60, 60, 60), 1)
        cv2.putText(pnl, "speed trace (cyan=trk, green=OMC)", (12, ty0 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        seg = trace[-tw:]
        mx = 800.
        for xi, (tvv, ovv) in enumerate(seg):
            if np.isfinite(ovv):
                cv2.circle(pnl, (12 + xi, int(ty0 + th - min(ovv, mx) / mx * th)), 1, (0, 200, 0), -1)
            if np.isfinite(tvv):
                cv2.circle(pnl, (12 + xi, int(ty0 + th - min(tvv, mx) / mx * th)), 1, (0, 255, 255), -1)

        # ZOOMED CUP INSET (top-left of the video region): 4x crop around the keypoint so the
        # individual tracks / inlier rings / centroid / spin axis are actually readable.
        if kp is not None:
            Z = 90                                        # half-size of the crop in original px
            zx0, zy0 = int(kp[0]) - Z, int(kp[1]) - Z
            zx0 = max(0, min(zx0, W - 2 * Z)); zy0 = max(0, min(zy0, Hh - 2 * Z))
            crop = canvas[zy0:zy0 + 2 * Z, zx0:zx0 + 2 * Z].copy()
            if crop.shape[0] == 2 * Z and crop.shape[1] == 2 * Z:
                big = cv2.resize(crop, (360, 360), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(big, (0, 0), (359, 359), (0, 200, 255), 2)
                cv2.putText(big, "cup 4x", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 200, 255), 2, cv2.LINE_AA)
                canvas[8:368, 8:368] = big

        vw.write(canvas)
        if f % 100 == 0:
            print(f"  frame {f}/{n}", flush=True)

    vw.release()
    for c in caps.values():
        c.release()
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
