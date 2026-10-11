"""Comprehensive dataset inspection script."""

import os
import struct
from pathlib import Path
from collections import Counter

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")

def get_png_info(filepath):
    # Read PNG header without PIL/cv2
    with open(filepath, 'rb') as f:
        sig = f.read(8)
        if sig != b'\x89PNG\r\n\x1a\n':
            return None
        ihdr_len = struct.unpack('>I', f.read(4))[0]
        ihdr_type = f.read(4)
        if ihdr_type != b'IHDR':
            return None
        w, h, bit_depth, color_type, comp, filter_m, interlace = struct.unpack('>IIBBBBB', f.read(13))
        # color_type: 0: Grayscale, 2: RGB, 3: Indexed, 4: Grayscale+alpha, 6: RGBA
        color_map = {0: 'Grayscale', 2: 'RGB', 3: 'Palette', 4: 'Gray+Alpha', 6: 'RGBA'}
        return {
            'width': w,
            'height': h,
            'bit_depth': bit_depth,
            'color_type': color_map.get(color_type, str(color_type)),
            'raw_color_type': color_type
        }

def inspect():
    print("=== STARTING DETAILED INSPECTION ===")
    cxr_dir = DATA_ROOT / "CXR_png"
    masks_dir = DATA_ROOT / "masks"
    clin_dir = DATA_ROOT / "ClinicalReadings"
    test_dir = DATA_ROOT / "test"

    cxr_files = sorted([f for f in cxr_dir.iterdir() if f.is_file()])
    mask_files = sorted([f for f in masks_dir.iterdir() if f.is_file()])
    clin_files = sorted([f for f in clin_dir.iterdir() if f.is_file()])
    test_files = sorted([f for f in test_dir.iterdir() if f.is_file()])

    mcu_cxr = [f for f in cxr_files if f.name.startswith("MCUCXR")]
    chn_cxr = [f for f in cxr_files if f.name.startswith("CHNCXR")]

    mcu_masks = [f for f in mask_files if f.name.startswith("MCUCXR")]
    chn_masks = [f for f in mask_files if f.name.startswith("CHNCXR")]

    print(f"CXR files: total={len(cxr_files)}, MCU={len(mcu_cxr)}, CHN={len(chn_cxr)}")
    print(f"Mask files: total={len(mask_files)}, MCU={len(mcu_masks)}, CHN={len(chn_masks)}")
    print(f"Test files: total={len(test_files)}")
    print(f"Clinical files: total={len(clin_files)}")

    print("\n--- SAMPLE MASK FILENAMES ---")
    print("MCU Masks samples:", [f.name for f in mcu_masks[:5]])
    print("CHN Masks samples:", [f.name for f in chn_masks[:5]])

    # Mask naming relationship
    # Let's see how mask filenames relate to CXR filenames
    # For MCU:
    # Does MCUCXR_XXXX_Y.png correspond to MCUCXR_XXXX_Y.png or MCUCXR_XXXX_Y_mask.png?
    print("\nMCU CXR samples:", [f.name for f in mcu_cxr[:5]])
    # For CHN:
    print("CHN CXR samples:", [f.name for f in chn_cxr[:5]])

    # Let's map images to masks
    # Check exact stem matches or suffix matches
    # Find images without mask
    cxr_names = {f.name for f in cxr_files}
    cxr_stems = {f.stem: f for f in cxr_files}

    # Check mask mapping logic
    mcu_matched = []
    mcu_unmatched_img = []
    for f in mcu_cxr:
        stem = f.stem
        # Possible mask names: stem + ".png", stem + "_mask.png", etc.
        cand1 = masks_dir / f"{stem}.png"
        cand2 = masks_dir / f"{stem}_mask.png"
        if cand1.exists():
            mcu_matched.append((f.name, cand1.name))
        elif cand2.exists():
            mcu_matched.append((f.name, cand2.name))
        else:
            mcu_unmatched_img.append(f.name)

    print(f"\nMCU images matched: {len(mcu_matched)}, unmatched images: {len(mcu_unmatched_img)}")
    if mcu_matched:
        print(f"MCU match example: image {mcu_matched[0][0]} -> mask {mcu_matched[0][1]}")

    chn_matched = []
    chn_unmatched_img = []
    for f in chn_cxr:
        stem = f.stem
        cand1 = masks_dir / f"{stem}.png"
        cand2 = masks_dir / f"{stem}_mask.png"
        if cand2.exists():
            chn_matched.append((f.name, cand2.name))
        elif cand1.exists():
            chn_matched.append((f.name, cand1.name))
        else:
            chn_unmatched_img.append(f.name)

    print(f"CHN images matched: {len(chn_matched)}, unmatched images: {len(chn_unmatched_img)}")
    if chn_matched:
        print(f"CHN match example: image {chn_matched[0][0]} -> mask {chn_matched[0][1]}")

    # Masks that have no image
    masks_without_image = []
    for m in mask_files:
        # strip _mask if present
        orig_stem = m.stem.replace("_mask", "")
        if f"{orig_stem}.png" not in cxr_names:
            masks_without_image.append(m.name)
    print(f"Masks without image in CXR_png: {len(masks_without_image)} -> {masks_without_image}")

    # Check test images vs unmatched CHN images
    test_names = [f.name for f in test_files]
    print(f"\nAre unmatched CHN images exactly test files? {sorted(chn_unmatched_img) == sorted(test_names)}")

    # Check labels (_0 and _1 suffix)
    # E.g. CHNCXR_0001_0.png -> 0 = normal, 1 = TB
    mcu_classes = Counter(f.stem.split('_')[-1] for f in mcu_cxr)
    chn_classes = Counter(f.stem.split('_')[-1] for f in chn_cxr)
    print(f"\nMCU class counts (_0=normal, _1=TB/abnormal): {dict(mcu_classes)}")
    print(f"CHN class counts (_0=normal, _1=TB/abnormal): {dict(chn_classes)}")

    # First 20 images of each source: size (H x W), mode, bit depth
    print("\n--- FIRST 20 MCU IMAGES ---")
    for f in mcu_cxr[:20]:
        info = get_png_info(f)
        print(f"{f.name}: {info['height']}x{info['width']}, mode={info['color_type']}, bit_depth={info['bit_depth']}")

    print("\n--- FIRST 20 CHN IMAGES ---")
    for f in chn_cxr[:20]:
        info = get_png_info(f)
        print(f"{f.name}: {info['height']}x{info['width']}, mode={info['color_type']}, bit_depth={info['bit_depth']}")

    # Clinical readings
    mcu_clin = [f for f in clin_files if f.name.startswith("MCUCXR")]
    chn_clin = [f for f in clin_files if f.name.startswith("CHNCXR")]
    print(f"\nClinical readings MCU: {len(mcu_clin)}, CHN: {len(chn_clin)}")
    print("\n--- 2 SAMPLE CLINICAL READINGS FOR MCU ---")
    for f in mcu_clin[:2]:
        print(f"=== {f.name} ===")
        print(f.read_text(encoding='utf-8', errors='replace'))
    print("--- 2 SAMPLE CLINICAL READINGS FOR CHN ---")
    for f in chn_clin[:2]:
        print(f"=== {f.name} ===")
        print(f.read_text(encoding='utf-8', errors='replace'))

    # Check mask values / channels / whether single combined or left/right
    # Let's inspect mask properties for MCU and CHN
    print("\n--- MASK PROPERTIES ---")
    mcu_mask_info = [get_png_info(f) for f in mcu_masks[:5]]
    print("MCU mask samples info:", mcu_mask_info)
    chn_mask_info = [get_png_info(f) for f in chn_masks[:5]]
    print("CHN mask samples info:", chn_mask_info)

    # Search for junk files
    junk_found = []
    for root, dirs, files in os.walk(DATA_ROOT):
        for f in files:
            p = Path(root) / f
            if f.lower() in ['thumbs.db', 'desktop.ini', '.ds_store'] or f.startswith('.'):
                junk_found.append(str(p.relative_to(DATA_ROOT)))
        for d in dirs:
            if d.startswith('.'):
                junk_found.append(str((Path(root) / d).relative_to(DATA_ROOT)))
    print(f"\nJunk files / hidden dirs found: {junk_found}")

if __name__ == "__main__":
    inspect()
