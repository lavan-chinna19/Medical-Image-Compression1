"""Print all 29 non-standard CHN first lines."""

from pathlib import Path
import re

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")
clin_dir = DATA_ROOT / "ClinicalReadings"
chn_files = sorted(clin_dir.glob("CHNCXR_*.txt"))

for f in chn_files:
    lines = [l.strip() for l in f.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    if lines:
        first = lines[0]
        if not re.match(r'^(male|female)\s+\d+\s*(yrs|yr|y)?$', first.lower().strip()):
            print(f"{f.name}: {first!r}")
