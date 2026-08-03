import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent))
from ssif_enhance import enhance

src_dir = Path("datasets/LIME")
out_dir = Path("datasets/LIME_enhanced")
out_dir.mkdir(parents=True, exist_ok=True)

for f in sorted(src_dir.glob("*.bmp"), key=lambda p: int(p.stem)):
    bgr = cv2.imread(str(f))
    if bgr is None:
        print(f"skip (unreadable): {f}")
        continue
    out = enhance(bgr, r=19, kappa=2.0, eps=0.001, levels=2, detail_weights=[1.5, 1.5], clip_limit=2.0)
    out_path = out_dir / f"{f.stem}.png"
    cv2.imwrite(str(out_path), out)
    print(f"{f.name}: mean {bgr.mean():.1f}->{out.mean():.1f}  std {bgr.std():.1f}->{out.std():.1f}  -> {out_path}")
