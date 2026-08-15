"""
Full-reference evaluation (PSNR / SSIM) of the SSIF enhancement against the
Part2 subset and VE-LOL-L, for comparison with Table 2 of:

  Y. Demir, N.H. Kaplan, "Low-light image enhancement based on
  sharpening-smoothing image filter", Digital Signal Processing 138 (2023).

LIEQA (the third metric in Table 2) is not implemented: it is a learned
perceptual metric from ref. [33] with no public reference implementation
bundled here, so only PSNR and SSIM are reported.

Protocol notes (the paper underspecifies these; choices are stated so the
numbers are interpretable rather than silently tuned):
  - Part2: each scene dir holds a multi-exposure sequence and Label/ holds
    the ground truth. The paper says the subset "consists of 229 images"
    with "multi-exposure sequences for each image" but does not say which
    exposure is the input. We evaluate every exposure and report both the
    per-scene mean over exposures and the per-scene best exposure.
  - VE-LOL-L: 1000 synthetic (Syn train+test) and 500 real (Cap train+test)
    per the paper's Sec. 4.1. Pairing is by filename (Syn) or by
    low->normal name substitution (Cap).
  - SSIM is computed on the grayscale image with the standard Wang et al.
    setup (11x11 Gaussian, sigma=1.5, gaussian_weights, data_range=255).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

sys.path.insert(0, str(Path(__file__).parent))
from ssif_enhance import enhance

PART2 = Path("datasets/Dataset_Part2/Dataset_Part2")
VELOL = Path("datasets/VE-LOL-L")


def metrics(out_bgr: np.ndarray, gt_bgr: np.ndarray) -> tuple[float, float]:
    if out_bgr.shape != gt_bgr.shape:
        gt_bgr = cv2.resize(gt_bgr, (out_bgr.shape[1], out_bgr.shape[0]), interpolation=cv2.INTER_AREA)
    p = psnr_fn(gt_bgr, out_bgr, data_range=255)
    g_out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)
    g_gt = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
    s = ssim_fn(g_gt, g_out, data_range=255, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return float(p), float(s)


def load(path: Path, max_side: int | None) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if max_side and max(img.shape[:2]) > max_side:
        sc = max_side / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    return img


def eval_part2(kw: dict, max_side: int | None, limit: int | None):
    labels = {p.stem: p for p in (PART2 / "Label").iterdir() if p.is_file()}
    scenes = sorted([d for d in PART2.iterdir() if d.is_dir() and d.name != "Label"], key=lambda p: int(p.name))
    if limit:
        scenes = scenes[:limit]

    mean_p, mean_s, best_p, best_s = [], [], [], []
    for i, sc in enumerate(scenes, 1):
        gt = load(labels[sc.name], max_side)
        if gt is None:
            print(f"  skip scene {sc.name}: unreadable label", file=sys.stderr)
            continue
        ps, ss = [], []
        for f in sorted(sc.iterdir()):
            im = load(f, max_side)
            if im is None:
                continue
            p, s = metrics(enhance(im, **kw), gt)
            ps.append(p)
            ss.append(s)
        if not ps:
            continue
        mean_p.append(np.mean(ps))
        mean_s.append(np.mean(ss))
        best_p.append(np.max(ps))
        best_s.append(np.max(ss))
        if i % 25 == 0:
            print(f"    ... {i}/{len(scenes)} scenes", flush=True)
    return {
        "n": len(mean_p),
        "psnr_mean_exp": float(np.mean(mean_p)),
        "ssim_mean_exp": float(np.mean(mean_s)),
        "psnr_best_exp": float(np.mean(best_p)),
        "ssim_best_exp": float(np.mean(best_s)),
    }


def eval_pairs(pairs, kw: dict, max_side: int | None, tag: str):
    ps, ss = [], []
    for i, (lo, gt_p) in enumerate(pairs, 1):
        im, gt = load(lo, max_side), load(gt_p, max_side)
        if im is None or gt is None:
            continue
        p, s = metrics(enhance(im, **kw), gt)
        ps.append(p)
        ss.append(s)
        if i % 200 == 0:
            print(f"    ... {tag} {i}/{len(pairs)}", flush=True)
    return {"n": len(ps), "psnr": float(np.mean(ps)), "ssim": float(np.mean(ss))}


def velol_pairs(limit: int | None):
    syn, cap = [], []
    for split in ("train", "test"):
        d_lo = VELOL / "VE-LOL-L-Syn" / f"VE-LOL-L-Syn-Low_{split}"
        d_gt = VELOL / "VE-LOL-L-Syn" / f"VE-LOL-L-Syn-Normal_{split}"
        for f in sorted(d_lo.iterdir()):
            g = d_gt / f.name
            if g.exists():
                syn.append((f, g))
        d_lo = VELOL / "VE-LOL-L-Cap-Full" / f"VE-LOL-L-Cap-Low_{split}"
        d_gt = VELOL / "VE-LOL-L-Cap-Full" / f"VE-LOL-L-Cap-Normal_{split}"
        for f in sorted(d_lo.iterdir()):
            g = d_gt / f.name.replace("low", "normal")
            if g.exists():
                cap.append((f, g))
    if limit:
        syn, cap = syn[:limit], cap[:limit]
    return syn, cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-side", type=int, default=1024, help="downscale long edge (0=off, default 1024)")
    ap.add_argument("--limit", type=int, default=None, help="cap items per dataset (smoke tests)")
    ap.add_argument("--omega", type=float, default=1.5)
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--clip-limit", type=float, default=2.0)
    ap.add_argument("--skip-part2", action="store_true")
    a = ap.parse_args()

    kw = dict(r=19, kappa=a.kappa, eps=0.001, levels=2,
              detail_weights=[a.omega, a.omega], clip_limit=a.clip_limit)
    ms = a.max_side or None
    print(f"params: kappa={a.kappa} omega={a.omega} clip_limit={a.clip_limit} max_side={ms}\n")

    if not a.skip_part2:
        print("Part2 subset:", flush=True)
        r = eval_part2(kw, ms, a.limit)
        print(f"  scenes={r['n']}")
        print(f"  mean-over-exposures : PSNR {r['psnr_mean_exp']:.3f}  SSIM {r['ssim_mean_exp']:.3f}")
        print(f"  best-exposure       : PSNR {r['psnr_best_exp']:.3f}  SSIM {r['ssim_best_exp']:.3f}")
        print(f"  paper (Table 2)     : PSNR 16.713  SSIM 0.735\n")

    syn, cap = velol_pairs(a.limit)
    print("VE-LOL-L:", flush=True)
    rs = eval_pairs(syn, kw, ms, "syn")
    rc = eval_pairs(cap, kw, ms, "cap")
    n = rs["n"] + rc["n"]
    comb_p = (rs["psnr"] * rs["n"] + rc["psnr"] * rc["n"]) / n
    comb_s = (rs["ssim"] * rs["n"] + rc["ssim"] * rc["n"]) / n
    print(f"  Syn  n={rs['n']:4d} : PSNR {rs['psnr']:.3f}  SSIM {rs['ssim']:.3f}")
    print(f"  Cap  n={rc['n']:4d} : PSNR {rc['psnr']:.3f}  SSIM {rc['ssim']:.3f}")
    print(f"  ALL  n={n:4d} : PSNR {comb_p:.3f}  SSIM {comb_s:.3f}")
    print(f"  paper (Table 2)  : PSNR 13.140  SSIM 0.388")


if __name__ == "__main__":
    main()
