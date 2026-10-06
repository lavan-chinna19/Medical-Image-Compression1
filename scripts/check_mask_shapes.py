"""Check if any mask shape differs from its corresponding image shape."""

from pathlib import Path
import cv2

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")
cxr_dir = DATA_ROOT / "CXR_png"
masks_dir = DATA_ROOT / "masks"

mismatches = []
total_checked = 0

for f in sorted(cxr_dir.glob("*.png")):
    stem = f.stem
    if f.name.startswith("MCUCXR"):
        mask_p = masks_dir / f.name
    else:
        mask_p = masks_dir / f"{stem}_mask.png"
    
    if mask_p.exists():
        total_checked += 1
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
        if img.shape[:2] != mask.shape[:2]:
            mismatches.append((stem, img.shape[:2], mask.shape[:2]))

print(f"Total masks checked: {total_checked}")
print(f"Mask shape mismatches count: {len(mismatches)}")
for m in mismatches:
    print(" ", m)
