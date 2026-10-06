"""Check patient_id uniqueness within each source."""

from pathlib import Path
from collections import Counter

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")
cxr_dir = DATA_ROOT / "CXR_png"

mcu_stems = [f.stem for f in cxr_dir.glob("MCUCXR_*.png")]
chn_stems = [f.stem for f in cxr_dir.glob("CHNCXR_*.png")]

# Option A: stem.rsplit("_", 1)[0]
mcu_ids_a = [s.rsplit("_", 1)[0] for s in mcu_stems]
chn_ids_a = [s.rsplit("_", 1)[0] for s in chn_stems]

# Option B: stem.split("_")[1]
mcu_ids_b = [s.split("_")[1] for s in mcu_stems]
chn_ids_b = [s.split("_")[1] for s in chn_stems]

print("Option A (source_id):")
print(f"  MCU total {len(mcu_ids_a)}, unique {len(set(mcu_ids_a))}")
print(f"  CHN total {len(chn_ids_a)}, unique {len(set(chn_ids_a))}")

print("Option B (number only):")
print(f"  MCU total {len(mcu_ids_b)}, unique {len(set(mcu_ids_b))}")
print(f"  CHN total {len(chn_ids_b)}, unique {len(set(chn_ids_b))}")
