"""Deep dive inspection of masks and test folder."""

import os
import hashlib
from pathlib import Path
import cv2
import numpy as np

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def deep_inspect():
    cxr_dir = DATA_ROOT / "CXR_png"
    masks_dir = DATA_ROOT / "masks"
    test_dir = DATA_ROOT / "test"

    # Compare test images to CXR_png images
    test_files = sorted(test_dir.iterdir())
    identical_count = 0
    for tf in test_files:
        cxr_f = cxr_dir / tf.name
        if cxr_f.exists() and md5(tf) == md5(cxr_f):
            identical_count += 1
    print(f"Test images identical to CXR_png: {identical_count} / {len(test_files)}")

    # Unmasked images list
    cxr_files = sorted(cxr_dir.iterdir())
    unmasked = []
    for cf in cxr_files:
        if cf.name.startswith("MCUCXR"):
            mf = masks_dir / cf.name
        else:
            mf = masks_dir / f"{cf.stem}_mask.png"
        if not mf.exists():
            unmasked.append(cf.name)
    print(f"Total unmasked images: {len(unmasked)}")
    print(f"Unmasked image names:\n{unmasked}")

    # Inspect mask pixel values and connected components (left and right lungs)
    mcu_sample_mask = masks_dir / "MCUCXR_0001_0.png"
    chn_sample_mask = masks_dir / "CHNCXR_0001_0_mask.png"

    for name, p in [("MCU Sample Mask", mcu_sample_mask), ("CHN Sample Mask", chn_sample_mask)]:
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        print(f"\n{name} ({p.name}):")
        print(f"  Shape: {arr.shape}, Dtype: {arr.dtype}")
        print(f"  Unique values: {np.unique(arr)}")
        # Check connected components to see if both lungs are present
        _, labels, stats, _ = cv2.connectedComponentsWithStats((arr > 0).astype(np.uint8))
        print(f"  Number of connected components (including background): {len(stats)}")

if __name__ == "__main__":
    deep_inspect()
