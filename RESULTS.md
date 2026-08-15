# Reproduction notes: full-reference evaluation vs. Table 2

Evaluation of this repository's SSIF implementation against Table 2 of:

> Y. Demir, N.H. Kaplan, "Low-light image enhancement based on
> sharpening-smoothing image filter", *Digital Signal Processing* 138 (2023) 104054.

Scripts: [`_eval_fullref.py`](_eval_fullref.py) (method), [`_eval_baseline.py`](_eval_baseline.py) (controls).
Enhancement: [`ssif_enhance.py`](ssif_enhance.py) at commit `82b77be` on branch `fix-ssif-alpha`.

**Bottom line: Table 2 did not reproduce.** The pipeline structure matches the
paper and the implementation beats a do-nothing baseline everywhere, but the
reported PSNR/SSIM values were not recovered. Several of the causes are
attributable to underspecification in the paper rather than to this code, and
they are separated out below.

---

## 1. Results

Parameters: `kappa=2, r=19, eps=0.001, L=2, omega=1.5, clip_limit=2.0`, long
edge downscaled to 1024 px. `kappa/r/eps/L` are the paper's (Sec. 4.2);
`omega` and `clip_limit` are **not** given by the paper (see §3.2).

### Part2 subset — 229 scenes, all exposures

| | PSNR | SSIM |
|---|---|---|
| Unenhanced input (control) | 12.502 | 0.510 |
| **This implementation, mean over exposures** | **13.503** | **0.549** |
| **This implementation, best exposure per scene** | **17.707** | **0.687** |
| Paper, Table 2 | 16.713 | 0.735 |

### VE-LOL-L — 1000 Syn + 500 Cap pairs

| | PSNR | SSIM |
|---|---|---|
| Unenhanced input (control) | 10.598 | 0.393 |
| — Syn only | 11.481 | 0.490 |
| — Cap only | 8.832 | 0.200 |
| **This implementation, all 1500** | **11.529** | **0.426** |
| — Syn only | 12.343 | 0.515 |
| — Cap only | 9.901 | 0.250 |
| Paper, Table 2 | 13.140 | 0.388 |

### Gain over the do-nothing control

The one check that does not depend on any unstated parameter:

| Split | ΔPSNR | ΔSSIM |
|---|---|---|
| Part2 (mean over exposures) | +1.001 | +0.039 |
| VE-LOL-Syn | +0.862 | +0.025 |
| VE-LOL-Cap | +1.069 | +0.050 |
| VE-LOL-L combined | +0.931 | +0.033 |

Positive on every split.

### How much of the gain comes from the eq. (5) fix?

A second control — the pre-fix, blur-only SSIF (§3.3) — isolates the
correction's contribution from CLAHE's. Full dataset, same protocol:

| Split | identity | pre-fix (blur-only) | fixed |
|---|---|---|---|
| Part2, PSNR | 12.502 | 12.993 | 13.503 |
| Part2, SSIM | 0.510 | 0.532 | 0.549 |
| VE-LOL-Syn, PSNR | 11.481 | 12.150 | 12.343 |
| VE-LOL-Syn, SSIM | 0.490 | 0.506 | 0.515 |
| VE-LOL-Cap, PSNR | 8.832 | 9.573 | 9.901 |
| VE-LOL-Cap, SSIM | 0.200 | 0.248 | 0.250 |

Attributing the total gain over identity:

| Split | CLAHE etc. (identity→pre-fix) | eq. (5) fix (pre-fix→fixed) | fix's share |
|---|---|---|---|
| Part2 | +0.491 dB / +0.022 | +0.510 dB / +0.017 | 51% PSNR, 44% SSIM |
| VE-LOL-Syn | +0.669 dB / +0.016 | +0.193 dB / +0.009 | 22% PSNR, 36% SSIM |
| VE-LOL-Cap | +0.741 dB / +0.048 | +0.328 dB / +0.002 | 31% PSNR, 4% SSIM |

The correction is monotonically positive on every split and every metric, and
on Part2 it accounts for about half the total improvement. This is independent
empirical support for the normalized reading of `|phi_k|` in §3.3: that reading
is not merely the one consistent with the paper's stated `kappa` semantics, it
also measurably outperforms the raw reading against ground truth on 229 scenes
and 1500 pairs. It is not proof — ref. [22] remains the authority — but it is
the strongest evidence obtainable without the primary source.

The effect is much weaker on VE-LOL-Cap SSIM (+0.002), where both variants sit
near the floor (~0.25); that split is dominated by whatever the enhancement
cannot fix, so it discriminates poorly between them.

---

## 2. Comparison to the paper

**Part2 — bracketed, not missed.** The paper's 16.713 falls *between* the two
readings of its own protocol: 13.503 (mean over exposures) and 17.707 (best
exposure per scene). Best-exposure overshoots the paper's PSNR while
undershooting its SSIM. Because the paper does not state which exposure is the
input, this row cannot be scored as a match or a mismatch — the protocol is
underdetermined and the choice is worth 4.2 dB.

**VE-LOL-L — a real gap.** 11.529 vs. 13.140 PSNR, with no protocol ambiguity
that plausibly closes 1.6 dB. This row is genuinely unreproduced.

**The SSIM anomaly is in the paper, not this code.** On VE-LOL-L the measured
SSIM (0.426) *exceeds* the paper's reported 0.388 — and so does the unenhanced
input (0.393). Every method in that row of Table 2 scores 0.375–0.449, i.e. at
or below doing nothing. That row does not demonstrate enhancement by SSIM for
any of the seven methods compared, including the paper's own. This is a
property of the published numbers and is independent of anything here.

---

## 3. Possible reasons for the discrepancy

Ordered by how much they could plausibly account for, and by whether they are
fixable from the paper text.

### 3.1 Protocol ambiguity in Part2 (not resolvable from the paper)

Sec. 4.1 says the subset "consists of 229 images" with "multi-exposure
sequences for each image", but never says which exposure is the low-light
input. Each scene dir here holds 5–9 exposures spanning roughly 8× in mean
brightness. Plausible readings — darkest exposure, all exposures averaged,
the exposure closest to some target level — differ by over 4 dB. No choice is
more defensible than the others from the text, so both bounds are reported
rather than the one that lands nearest the paper.

### 3.2 Two required parameters are never stated (not resolvable)

- **`omega_j`** (detail amplification, eq. 11) is called only "an arbitrary
  coefficient" / "a predefined enhancement coefficient". No value appears
  anywhere in the paper. `1.5` here is a choice.
- **`clip_limit`** — eq. (6) defines `beta` from image size and crop factor,
  but Table 1 lists `beta` as an *input*, and the crop factor `alpha` is never
  given either. Moreover eq. (6) is not implementable as printed: `M*N/100`
  for a 512×512 image yields `beta ~ 2621*(s_max-1)*alpha/100`, which is
  nonsensical and is almost certainly an OCR corruption of a per-tile pixel
  count. The `2.0` used here is OpenCV's per-tile normalized limit, a
  different quantity from the paper's `beta`.

Table 2 is therefore partly a test of parameters the paper does not supply. A
gap is expected and is not evidence about the algorithm. **An `omega` /
`clip_limit` sweep against Part2 has not been run**; if no setting reaches
16.713 / 0.735, that would point at the paper rather than at these unknowns.

### 3.3 Ambiguity in eq. (5) (resolved by inference, unconfirmed)

`|phi_k|` is the patch covariance of input and guidance. The source text does
not make clear whether it is the raw or the variance-normalized covariance.
This implementation uses the normalized form (`cov(I,G)/var(G)`, identically 1
when self-guided), inferred from dimensional analysis of eq. (5), from
deriving it out of eq. (4), and from it being the only reading consistent with
the paper's own Sec. 3.1 statement that `kappa>1` sharpens, `kappa<1` smooths,
and `kappa=1` applies no filtering.

The raw reading — what this repo did before `82b77be` — caps `alpha` at ~0.19
for V in [0,1], collapsing SSIF into a plain box mean filter that never
sharpens and in which `kappa` is inert (sweeping 0.1..20 moved the output by
≤ 17/255). See the commit message for the measurements.

**This has not been checked against the primary source.** Ref. [22] (Deng et
al., *IEEE OJSP* 2 (2021) 119–135) defines `phi_k` properly and should be
consulted before citing any of this. The normalized reading does, however,
beat the raw one against ground truth on every split — see the attribution
table in §1.

### 3.4 Resolution (deviation introduced here)

Part2 originals are up to 3648×5472 and were downscaled to a 1024 px long edge
for tractability. This shifts absolute PSNR — SSIF is a fixed-radius local
filter (r=19), so downscaling changes its effective spatial support relative
to image content. Direction and size of the effect are unmeasured. A
native-resolution run is the obvious next check.

### 3.5 Unmodelled parts of the method

- **Scale parameter `s=0.1`** (Sec. 4.2) is omitted entirely — its formula is
  not recoverable from the text. The paper reports its effect on PSNR is
  negligible, so this is likely minor, but it is a genuine missing piece.
- **Self-guiding (`G = I`)** is an assumption. Eq. (7) writes `SSIF(I)` with no
  separate guidance image, but the paper never states this explicitly.

### 3.6 Metric implementation details

SSIM is computed on grayscale with the standard Wang et al. setup (11×11
Gaussian, sigma=1.5, `data_range=255`). The paper does not specify grayscale
vs. per-channel, window size, or whether PSNR is computed on RGB or Y. These
choices routinely move SSIM by several hundredths and PSNR by a few tenths of
a dB — enough to matter at the margins here, though not enough to explain the
VE-LOL-L PSNR gap on their own.

---

## 4. Not verified

- **LIEQA**, the third metric in Table 2, is not implemented. It is a learned
  perceptual metric from ref. [33] with no reference implementation available
  here. No substitute was used.
- **Table 3** (no-reference metrics: NIQE, NLIEE) was not attempted.
- The comparison methods in Table 2 (SRIE, LIME, Robust Retinex, DeepUPE,
  EnlightenGAN, Zero-DCE) were not re-run; only the "Proposed" column is at
  issue here.

---

## 5. Reproducing

```bash
pip install scikit-image opencv-python numpy

python _eval_fullref.py --max-side 1024        # method (~25 min)
python _eval_baseline.py --max-side 1024       # controls (~35 min)
python _eval_fullref.py --limit 5              # smoke test (~40 s)
```

Datasets expected at `datasets/Dataset_Part2/Dataset_Part2/` (229 scene dirs +
`Label/`) and `datasets/VE-LOL-L/` (`VE-LOL-L-Syn/`, `VE-LOL-L-Cap-Full/`).
Both were verified complete: all 229 Part2 scenes pair with a label (dir `200`
is `.PNG`, not `.JPG`), and VE-LOL-L holds exactly the 1000 Syn + 500 Cap
pairs described in Sec. 4.1.
