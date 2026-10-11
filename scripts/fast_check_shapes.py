"""Fast header check of mask vs image shapes."""

from pathlib import Path
import struct

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")
cxr_dir = DATA_ROOT / "CXR_png"
masks_dir = DATA_ROOT / "masks"

def get_png_shape(path):
    with open(path, "rb") as f:
        sig = f.read(8)
        if sig != b'\x89PNG\r\n\x1a\n':
            return None
        f.read(4) # len
        f.read(4) # IHDR
        w, h = struct.unpack('>II', f.read(8))
        return (h, w)

mismatches = []
total = 0
for f in sorted(cxr_dir.glob("*.png")):
    stem = f.stem
    if f.name.startswith("MCUCXR"):
        mask_p = masks_dir / f.name
    else:
        mask_p = masks_dir / f"{stem}_mask.png"
    if mask_p.exists():
        total += 1
        img_shape = get_png_shape(f)
        mask_shape = get_png_shape(mask_p)
        if img_shape != mask_shape:
            mismatches.append((stem, img_shape, mask_shape))

print(f"Total checked: {total}")
print(f"Shape mismatches: {len(mismatches)}")
for m in mismatches:
    print(" ", m)
