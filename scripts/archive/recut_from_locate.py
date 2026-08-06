"""Re-cut shuffled trials at the PIXEL-EXACT positions found by cut_placement_audit (NCC locate).

Why not recut_from_audit.py: that path re-refines each guess with `align_trial` (motion-energy), which
WANDERS on the quasi-periodic drink task (the documented failure). cut_placement_audit already located
each cut clip in its own uncut by same-camera NCC (0.998 vs 0.986 bg floor) -- a pixel-exact position
that motion energy can't improve on. So we PARSE those positions from the audit log and ffmpeg-extract
straight from the (local) uncut. No motion-align, no re-decode, geometry-verified afterwards separately.

Parses lines like:
    trial_10_L_unaffected: ref@ 668.80s(ncc 0.999)  cam2@ 668.10s(ncc 0.998)  misplacement  +1.80s
    cam2: session offset -2.5s vs cam3 ...
-> CORRECT recut position = ref@ + session_offset  (NOT cam2@, which is where the SHUFFLED clip
   currently sits -- extracting there just re-encodes the same wrong footage). ref@ is where the
   reference cam's clip sits; adding the constant session offset gives where the SAME drink rep sits
   in the suspect's uncut. (Bug fixed 2026-08-05: v1 parsed cam{N}@ and re-cut the identical wrong rep.)

Originals preserved in work/rejected_preRecut/ (never deleted). Output overwrites the shuffled clip at
work/clips/delta_<part>_<trial>.<cam>.mp4 so the existing detect/track pipeline picks it up unchanged.

    python scripts/recut_from_locate.py --part P13 --cam 2 --log out/recut/locate_all_P13.log [--dry]
"""
from __future__ import annotations
import argparse, re, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "archive"))
from cut_placement_audit import resolve_uncut          # noqa: E402
from delta_recut import SHARE, duration                # noqa: E402

CT = Path(__file__).resolve().parents[1]
# only accept a confident NCC placement; 0.99 is well above the 0.986 background floor
NCC_MIN = 0.99


def parse_positions(log_path, cam):
    """{trial: correct_t_seconds} = ref@ + session_offset, for confident REF locates.

    The CORRECT cut position is where the reference cam's clip sits (ref@, located pixel-exact by NCC
    in the ref uncut) PLUS the constant session offset between the two cameras' clocks. We gate on the
    REF ncc (the ref locate is what anchors the position); the cam{N}@ field is only used to know the
    offset was measured, NOT as the cut position (that was the v1 bug -- it's the shuffled position)."""
    text = Path(log_path).read_text()
    # session offset for this cam: "  camN: session offset +2.0s vs cam3 ..."
    mo = re.search(rf"^\s*cam{cam}:\s*session offset\s*([+-][\d.]+)s", text, re.M)
    if mo is None:
        raise SystemExit(f"no session-offset line for cam{cam} in {log_path}")
    session_off = float(mo.group(1))
    # per trial: ref@ <t>(ncc <f>) ... cam{cam}@ ... (only lines that also have this cam's locate)
    pat = re.compile(
        rf"^\s*(trial_\S+):\s*ref@\s*([\d.]+)s\(ncc\s*([\d.]+)\).*?cam{cam}@", re.M)
    out, low = {}, []
    for m in pat.finditer(text):
        trial, t_ref, ncc_ref = m.group(1), float(m.group(2)), float(m.group(3))
        if ncc_ref >= NCC_MIN:
            out[trial] = t_ref + session_off        # CORRECT position (not the shuffled cam@ position)
        else:
            low.append((trial, ncc_ref))
    print(f"  cam{cam}: session offset {session_off:+.2f}s -> recut at ref@ + offset", flush=True)
    return out, low


def recut(part, cam, log_path, dry=False):
    usus = resolve_uncut(part, cam)
    if usus is None or not Path(usus).exists():
        raise SystemExit(f"{part} cam{cam}: no uncut at {usus}")
    pos, low = parse_positions(log_path, cam)
    print(f"{part} cam{cam}: {len(pos)} confident positions ({len(low)} low-ncc skipped) from {log_path}",
          flush=True)
    if not pos:
        raise SystemExit("no positions parsed -- check the log/cam")

    clips = CT / "cache" / "delta" / part / "work" / "clips"
    pre = CT / "cache" / "delta" / part / "work" / "rejected_preRecut"
    pre.mkdir(parents=True, exist_ok=True)
    vd = Path(SHARE) / part / "01_Measurement" / "04_Video" / "03_Cut" / "drinking"

    done = miss = 0
    for trial, t in sorted(pos.items()):
        # duration from the REF cam's cut clip (the canonical trial length)
        rc = vd / "cam3" / f"{trial}.mp4"
        if not rc.exists():
            print(f"  {trial}: no ref clip for duration, skip", flush=True); miss += 1; continue
        dur = duration(rc)
        dst = clips / f"delta_{part}_{trial}.{cam}.mp4"
        if dst.exists() and not (pre / dst.name).exists():
            (pre / dst.name).write_bytes(dst.read_bytes())          # preserve original ONCE
        if dry:
            print(f"  {trial}: cut cam{cam}@ {t:.3f}s  dur {dur:.2f}s -> {dst.name}", flush=True)
            done += 1; continue
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-i", str(usus),
                        "-t", f"{dur:.3f}", "-vf", "scale=1920:1080",
                        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-an", str(dst)],
                       check=True)
        done += 1
        if done % 20 == 0:
            print(f"    recut {done}/{len(pos)}", flush=True)
    print(f"\nPROCESSING CHECK: recut {done} clips, {miss} no-ref-dur, {len(low)} low-ncc-skipped",
          flush=True)
    print(f"originals preserved in {pre}", flush=True)
    print("DONE", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True)
    ap.add_argument("--cam", type=int, required=True)
    ap.add_argument("--log", required=True, help="a cut_placement_audit --n-trials 999 log")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    recut(a.part, a.cam, a.log, a.dry)


if __name__ == "__main__":
    main()
