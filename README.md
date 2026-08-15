# Low-Light Image Enhancement via SSIF + CLAHE

Python reimplementation of:

> Y. Demir, N.H. Kaplan, "Low-light image enhancement based on
> sharpening-smoothing image filter", *Digital Signal Processing* 138 (2023) 104054.

The method converts to HSV, decomposes the V channel with a multi-scale
Smoothing-Sharpening Image Filter (SSIF), applies CLAHE to the final
approximation layer, amplifies the detail layers, and recombines.

## Usage

```bash
pip install opencv-python numpy scikit-image

python ssif_enhance.py input.png output.png
python ssif_enhance.py input.png output.png --kappa 2.0 --radius 19 --detail-weights 1.5,1.5
```

## Files

| File | Purpose |
|---|---|
| [`ssif_enhance.py`](ssif_enhance.py) | The method — SSIF filter, decomposition, CLAHE, CLI |
| [`_eval_fullref.py`](_eval_fullref.py) | PSNR/SSIM evaluation against Part2 and VE-LOL-L |
| [`_eval_baseline.py`](_eval_baseline.py) | Controls: unenhanced input, and the pre-fix filter |
| [`_run_batch.py`](_run_batch.py) | Batch-enhance the LIME dataset |
| [`RESULTS.md`](RESULTS.md) | Reproduction notes vs. the paper's Table 2 |

## Datasets

`datasets/LIME/` is included. The two large full-reference datasets are **not
committed** (~5 GB) — download and unpack them into `datasets/`:

**https://drive.google.com/drive/folders/1POBQBbx7msRbjpXYr5Q1vk0YmwiZ78Rm?usp=sharing**

Expected layout:

```
datasets/
  Dataset_Part2/Dataset_Part2/   # 229 scene dirs (5-9 exposures each) + Label/
  VE-LOL-L/
    VE-LOL-L-Syn/                # 1000 synthetic pairs
    VE-LOL-L-Cap-Full/           # 500 real pairs
```

## Status

Two things to know before relying on this:

1. **A correctness bug in the SSIF filter was fixed** — `|phi_k|` in eq. (5)
   needed to be the variance-normalized covariance. Before the fix the filter
   never sharpened and collapsed to a box blur. The corrected reading is
   *inferred* from the paper's equations, not confirmed against the primary
   SSIF source (ref. [22], Deng et al. 2021).
2. **Table 2 does not reproduce.** The implementation beats a do-nothing
   baseline on every split, but the paper's reported PSNR/SSIM were not
   recovered. Two required parameters are never stated in the paper and the
   Part2 protocol is ambiguous. See [RESULTS.md](RESULTS.md).
