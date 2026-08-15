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
image G = input I):
    alpha_k = r_k/2 + sqrt{ (r_k/2)^2 + 4*kappa*eps*r_k }
    r_k     = var_k / (var_k + eps)
    J(q)    = mu_k + alpha_k * (I(q) - mu_k)
computed with box filters over a (2r+1)x(2r+1) patch.

In eq. (5) |phi_k| is the patch covariance of the original and guidance
images, normalized by the guidance variance sigma^2_k -- i.e. the usual
guided-filter gain cov(I,G)/var(G), which is identically 1 when G = I.
Hence the leading term above is r_k/2 rather than var_k*r_k/2. Using the
raw (unnormalized) covariance instead caps alpha at ~0.19 for V in [0,1],
which degenerates SSIF into a plain box mean filter: it never sharpens,
and kappa becomes inert (varying it over 0.1..20 moves the output by
<= 17/255). That contradicts the paper's own statement in Sec. 3.1 that
kappa>1 sharpens, kappa<1 smooths, and kappa=1 applies no filtering.

Caveat: the source text is ambiguous about whether phi is the raw or the
normalized covariance; the normalized reading is inferred from dimensional
analysis of eq. (5), from deriving it out of eq. (4), and from being the
only reading consistent with the kappa semantics above. Ref. [22]
(Deng et al., IEEE OJSP 2 (2021) 119-135) is the definitive source.

Parameters NOT specified by the paper (chosen here, not reproduced):
  - detail_weights (omega_j in eq. 11): the paper calls this an "arbitrary
    coefficient" / "predefined enhancement coefficient" and never gives a
    value.
  - clip_limit: this is OpenCV's per-tile normalized clip limit, NOT the
    paper's beta from eq. (6). Eq. (6) is not implementable as printed --
    M*N/100 for a 512x512 image yields beta ~ 2621*(s_max-1)*alpha/100,
    which is nonsensical, almost certainly an OCR corruption of a per-tile
    pixel count. Table 1 lists beta as an input anyway.
  - Self-guiding (G = I): a reading of eq. (7), which writes SSIF(I) with
    no separate guidance image; the paper does not state it explicitly.
  - Scale (s): the paper lists it as a fourth SSIF parameter but its
    formula is not recoverable from the source text. The paper reports s
    has a negligible effect on PSNR, so it is omitted rather than guessed.
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


def ssif_filter(
    img: np.ndarray,
    r: int,
    kappa: float,
    eps: float,
    guide: np.ndarray | None = None,
) -> np.ndarray:
    """Guided smoothing-sharpening filter (eqs. 1-5); self-guided when guide is None."""
    if guide is None:
        guide = img

    v_mean, guide_var = local_variance(guide, r)
    mu = box_mean(img, r)
    # phi_k: patch covariance of input and guidance, normalized by the
    # guidance variance (eq. 5). Identically 1 in the self-guided case.
    cov = box_mean(img * guide, r) - mu * v_mean
    # Normalize by guide_var alone, not guide_var + eps: eq. (5) applies the
    # eps regularization separately via ratio, so folding it in here too
    # would double-count it. phi is then identically 1 when G = I.
    phi = np.divide(cov, guide_var, out=np.ones_like(cov), where=guide_var > 0)

    ratio = guide_var / (guide_var + eps)
    term = 0.5 * np.abs(phi) * ratio
    alpha = term + np.sqrt(term * term + 4.0 * kappa * eps * ratio)
    return mu + np.sign(cov) * alpha * (guide - v_mean)


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
