"""
Control for _eval_fullref.py: scores the UNENHANCED low-light input against
the ground truth, plus the pre-fix (broken, blur-only SSIF) variant.

Without this baseline the Table 2 comparison is uninterpretable -- a method
must at minimum beat "do nothing" to have shown anything.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _eval_fullref import PART2, load, metrics, velol_pairs
from ssif_enhance import box_mean, local_variance, apply_clahe


def enhance_broken(bgr, r=19, kappa=2.0, eps=0.001, levels=2,
                   detail_weights=(1.5, 1.5), clip_limit=2.0, tile_grid=(8, 8)):
    """Pre-fix implementation: |phi| = raw variance, so SSIF ~ box mean."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v01 = v.astype(np.float64) / 255.0

    details, sp = [], v01
    for _ in range(levels):
        mu, var = local_variance(sp, r)
        ratio = var / (var + eps)
        term = 0.5 * var * ratio                     # <-- the bug
        alpha = term + np.sqrt(term * term + 4.0 * kappa * eps * ratio)
        sn = mu + alpha * (sp - mu)
        details.append(sp - sn)
        sp = sn

    v_enh = apply_clahe(sp, clip_limit, tile_grid).copy()
    for w, d in zip(detail_weights, details):
        v_enh += w * d
    v8 = (np.clip(v_enh, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(cv2.merge([h, s, v8]), cv2.COLOR_HSV2BGR)


def run(fn, label, max_side, limit):
    labels = {p.stem: p for p in (PART2 / "Label").iterdir() if p.is_file()}
    scenes = sorted([d for d in PART2.iterdir() if d.is_dir() and d.name != "Label"],
                    key=lambda p: int(p.name))
    if limit:
        scenes = scenes[:limit]
    mp, msim = [], []
    for i, sc in enumerate(scenes, 1):
        gt = load(labels[sc.name], max_side)
        if gt is None:
            continue
        ps, ss = [], []
        for f in sorted(sc.iterdir()):
            im = load(f, max_side)
            if im is None:
                continue
            p, s = metrics(fn(im), gt)
            ps.append(p); ss.append(s)
        if ps:
            mp.append(np.mean(ps)); msim.append(np.mean(ss))
        if i % 50 == 0:
            print(f"    ... {label} part2 {i}/{len(scenes)}", flush=True)
    print(f"  Part2 {label:12s} n={len(mp):3d}: PSNR {np.mean(mp):.3f}  SSIM {np.mean(msim):.3f}", flush=True)

    syn, cap = velol_pairs(limit)
    for name, pairs in (("Syn", syn), ("Cap", cap)):
        ps, ss = [], []
        for lo, gp in pairs:
            im, gt = load(lo, max_side), load(gp, max_side)
            if im is None or gt is None:
                continue
            p, s = metrics(fn(im), gt)
            ps.append(p); ss.append(s)
        print(f"  VE-LOL-{name} {label:12s} n={len(ps):4d}: PSNR {np.mean(ps):.3f}  SSIM {np.mean(ss):.3f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    ms = a.max_side or None
    print("=== CONTROL: unenhanced input vs ground truth ===", flush=True)
    run(lambda x: x, "identity", ms, a.limit)
    print("\n=== CONTROL: pre-fix (blur-only) SSIF ===", flush=True)
    run(enhance_broken, "broken", ms, a.limit)
