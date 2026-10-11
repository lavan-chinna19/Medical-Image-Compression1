"""Script to inspect the Kaggle Chest X-ray Masks and Labels dataset."""

import os
from pathlib import Path
from collections import defaultdict, Counter

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")

def run_inspection():
    print("=== INSPECTION OF DATASET AT:", DATA_ROOT)
    
    # 1. Top 3 levels folder tree with counts
    print("\n--- FOLDER TREE (TOP 3 LEVELS) ---")
    tree = {}
    for root, dirs, files in os.walk(DATA_ROOT):
        rel = Path(root).relative_to(DATA_ROOT)
        parts = rel.parts
        if len(parts) <= 3:
            ext_counts = Counter(Path(f).suffix.lower() for f in files)
            print(f"Path: {rel} | Total files: {len(files)} | Extensions: {dict(ext_counts)}")
            sample_files = files[:3]
            print(f"  Samples: {sample_files}")

    # 2. Inspect CXR_png, masks, ClinicalReadings, test
    # Let's inspect CXR_png
    cxr_dir = DATA_ROOT / "CXR_png"
    if cxr_dir.exists():
        cxr_files = [f for f in cxr_dir.iterdir() if f.is_file()]
        print(f"\nCXR_png total files: {len(cxr_files)}")
        montgomery_cxr = [f for f in cxr_files if f.name.startswith("MCUCXR")]
        shenzhen_cxr = [f for f in cxr_files if f.name.startswith("CHNCXR")]
        other_cxr = [f for f in cxr_files if not f.name.startswith("MCUCXR") and not f.name.startswith("CHNCXR")]
        print(f"Montgomery images (MCUCXR): {len(montgomery_cxr)}")
        print(f"Shenzhen images (CHNCXR): {len(shenzhen_cxr)}")
        print(f"Other images: {len(other_cxr)} -> {[f.name for f in other_cxr]}")
        
    # Masks
    masks_dir = DATA_ROOT / "masks"
    if masks_dir.exists():
        mask_files = [f for f in masks_dir.iterdir() if f.is_file()]
        print(f"\nmasks total files: {len(mask_files)}")
        montgomery_masks = [f for f in mask_files if f.name.startswith("MCUCXR")]
        shenzhen_masks = [f for f in mask_files if f.name.startswith("CHNCXR")]
        other_masks = [f for f in mask_files if not f.name.startswith("MCUCXR") and not f.name.startswith("CHNCXR")]
        print(f"Montgomery masks (MCUCXR): {len(montgomery_masks)}")
        print(f"Shenzhen masks (CHNCXR): {len(shenzhen_masks)}")
        print(f"Other masks: {len(other_masks)} -> {[f.name for f in other_masks]}")
        
        # Check subfolders inside masks if any
        subdirs = [d for d in masks_dir.iterdir() if d.is_dir()]
        print(f"Masks subdirectories: {[d.name for d in subdirs]}")
        for d in subdirs:
            sub_files = list(d.glob("*"))
            print(f"  {d.name}: {len(sub_files)} files. Samples: {[f.name for f in sub_files[:3]]}")

    # Let's check test folder
    test_dir = DATA_ROOT / "test"
    if test_dir.exists():
        test_files = list(test_dir.glob("**/*"))
        print(f"\ntest directory total items: {len(test_files)}")
        for f in test_files[:5]:
            print(f"  test item: {f.relative_to(test_dir)}")

    # Clinical readings
    clinical_dir = DATA_ROOT / "ClinicalReadings"
    if clinical_dir.exists():
        clin_files = [f for f in clinical_dir.iterdir() if f.is_file()]
        print(f"\nClinicalReadings total files: {len(clin_files)}")
        subdirs_clin = [d for d in clinical_dir.iterdir() if d.is_dir()]
        print(f"ClinicalReadings subdirectories: {[d.name for d in subdirs_clin]}")
        for d in subdirs_clin:
            sub_files = list(d.glob("*"))
            print(f"  {d.name}: {len(sub_files)} files. Samples: {[f.name for f in sub_files[:3]]}")

if __name__ == "__main__":
    run_inspection()
