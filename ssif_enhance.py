"""
Low-light image enhancement via multi-scale Smoothing-Sharpening Image Filter
(SSIF) decomposition + CLAHE, following:

Y. Demir, N.H. Kaplan, "Low-light image enhancement based on
sharpening-smoothing image filter", Digital Signal Processing 138 (2023) 104054.

Pipeline (paper Section 3, Table 1):
  1. RGB -> HSV, operate on the V channel only.
  2. Multi-scale SSIF decomposition of V into L detail layers + one
     approximation layer (eqs. 7-10).
  3. CLAHE on the final approximation layer (eq. 6, via cv2's CLAHE).
  4. Detail layers amplified by per-level weights and summed with the
     CLAHE'd approximation layer (eq. 11).
  5. HSV -> RGB with the enhanced V and the original H, S.

SSIF filter (eqs. 1-5) is implemented in its self-guided form (guidance
image G = input I), which collapses the guided-filter cross terms to plain
local variance:
    alpha_k = (var_k/2)*r_k + sqrt{ (var_k/2)^2 * r_k^2 + 4*kappa*eps*r_k }
    r_k     = var_k / (var_k + eps)
    J(q)    = mu_k + alpha_k * (I(q) - mu_k)
computed with box filters over a (2r+1)x(2r+1) patch.

Note: the paper lists a fourth SSIF parameter, Scale (s), but its formula
was not recoverable from the source text (see the OCR notes in the
accompanying .md); the paper itself reports s has a negligible effect on
PSNR, so it is omitted here rather than guessed at.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


def box_mean(img: np.ndarray, r: int) -> np.ndarray:
    k = 2 * r + 1
    return cv2.boxFilter(img, ddepth=-1, ksize=(k, k), normalize=True, borderType=cv2.BORDER_REFLECT)


def local_variance(img: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    mu = box_mean(img, r)
    mean_sq = box_mean(img * img, r)
    var = np.clip(mean_sq - mu * mu, 0.0, None)
    return mu, var


def ssif_filter(img: np.ndarray, r: int, kappa: float, eps: float) -> np.ndarray:
    """Self-guided smoothing-sharpening filter (eqs. 1-5, G = I)."""
    mu, var = local_variance(img, r)
    ratio = var / (var + eps)
    term = 0.5 * var * ratio
    alpha = term + np.sqrt(term * term + 4.0 * kappa * eps * ratio)
    return mu + alpha * (img - mu)


def multiscale_decompose(v: np.ndarray, levels: int, r: int, kappa: float, eps: float):
    """Returns (detail_layers, approximation) per eqs. (7)-(10)."""
    details = []
    s_prev = v
    for _ in range(levels):
        s_next = ssif_filter(s_prev, r, kappa, eps)
        details.append(s_prev - s_next)
        s_prev = s_next
    return details, s_prev


def apply_clahe(img01: np.ndarray, clip_limit: float, tile_grid: tuple[int, int]) -> np.ndarray:
    u8 = np.clip(img01, 0.0, 1.0)
    u8 = (u8 * 255.0).round().astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    out = clahe.apply(u8)
    return out.astype(np.float64) / 255.0


def enhance(
    bgr: np.ndarray,
    r: int = 19,
    kappa: float = 2.0,
    eps: float = 0.001,
    levels: int = 2,
    detail_weights: list[float] | None = None,
    clip_limit: float = 2.0,
    tile_grid: tuple[int, int] = (8, 8),
) -> np.ndarray:
    if detail_weights is None:
        detail_weights = [1.5] * levels
    if len(detail_weights) != levels:
        raise ValueError(f"expected {levels} detail weights, got {len(detail_weights)}")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v01 = v.astype(np.float64) / 255.0

    details, approx = multiscale_decompose(v01, levels, r, kappa, eps)
    approx_enh = apply_clahe(approx, clip_limit, tile_grid)

    v_enh = approx_enh.copy()
    for w, d in zip(detail_weights, details):
        v_enh += w * d

    v_enh_u8 = (np.clip(v_enh, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    hsv_enh = cv2.merge([h, s, v_enh_u8])
    return cv2.cvtColor(hsv_enh, cv2.COLOR_HSV2BGR)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="path to the low-light input image")
    p.add_argument("output", help="path to write the enhanced image")
    p.add_argument("--radius", type=int, default=19, help="SSIF patch radius r (default: 19)")
    p.add_argument("--kappa", type=float, default=2.0, help="SSIF kappa, >1 sharpens, <1 smooths (default: 2.0)")
    p.add_argument("--eps", type=float, default=0.001, help="SSIF epsilon regularizer (default: 0.001)")
    p.add_argument("--levels", type=int, default=2, help="number of decomposition levels L (default: 2)")
    p.add_argument(
        "--detail-weights",
        type=str,
        default=None,
        help="comma-separated per-level detail amplification weights, e.g. 1.5,1.5 (default: 1.5 per level)",
    )
    p.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)")
    p.add_argument("--tile-grid", type=str, default="8,8", help="CLAHE tile grid size, e.g. 8,8 (default: 8,8)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"error: could not read image '{args.input}'", file=sys.stderr)
        return 1

    weights = (
        [float(x) for x in args.detail_weights.split(",")]
        if args.detail_weights
        else [1.5] * args.levels
    )
    tile_grid = tuple(int(x) for x in args.tile_grid.split(","))

    out = enhance(
        bgr,
        r=args.radius,
        kappa=args.kappa,
        eps=args.eps,
        levels=args.levels,
        detail_weights=weights,
        clip_limit=args.clip_limit,
        tile_grid=tile_grid,
    )

    if not cv2.imwrite(args.output, out):
        print(f"error: could not write image '{args.output}'", file=sys.stderr)
        return 1

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
