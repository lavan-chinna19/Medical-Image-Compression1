"""Compare test vs CXR_png images."""

from pathlib import Path
import cv2
import numpy as np

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")

test_dir = DATA_ROOT / "test"
cxr_dir = DATA_ROOT / "CXR_png"

for tf in sorted(test_dir.iterdir())[:5]:
    cf = cxr_dir / tf.name
    t_img = cv2.imread(str(tf), cv2.IMREAD_UNCHANGED)
    c_img = cv2.imread(str(cf), cv2.IMREAD_UNCHANGED)
    print(f"File: {tf.name}")
    print(f"  test shape: {t_img.shape}, dtype: {t_img.dtype}")
    print(f"  cxr  shape: {c_img.shape}, dtype: {c_img.dtype}")
    if t_img.shape == c_img.shape:
        diff = np.abs(t_img.astype(float) - c_img.astype(float))
        print(f"  Max diff: {np.max(diff)}, Mean diff: {np.mean(diff)}")
    else:
        print("  Shapes differ!")
