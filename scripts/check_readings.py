"""Inspect all clinical reading files to understand variations."""

from pathlib import Path
import re

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")
clin_dir = DATA_ROOT / "ClinicalReadings"

files = sorted(clin_dir.glob("*.txt"))
print(f"Total clinical reading files: {len(files)}")

mcu_files = [f for f in files if f.name.startswith("MCUCXR")]
chn_files = [f for f in files if f.name.startswith("CHNCXR")]

print(f"MCU files: {len(mcu_files)}, CHN files: {len(chn_files)}")

# Check MCU readings lines
mcu_sexes = set()
mcu_ages = set()
for f in mcu_files:
    lines = [l.strip() for l in f.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    for l in lines:
        if "Sex:" in l:
            mcu_sexes.add(l)
        elif "Age:" in l:
            mcu_ages.add(l)

print("Sample MCU sexes:", list(mcu_sexes)[:10])
print("Sample MCU ages:", list(mcu_ages)[:10])

# Check CHN readings lines
chn_first_lines = []
for f in chn_files:
    lines = [l.strip() for l in f.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    if lines:
        chn_first_lines.append(lines[0])

print(f"Total CHN first lines: {len(chn_first_lines)}")
print("Sample CHN first lines:", chn_first_lines[:15])

# Find any CHN first lines that don't match typical sex/age pattern
unusual_chn = []
for f in chn_files:
    lines = [l.strip() for l in f.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    if not lines:
        unusual_chn.append((f.name, "empty"))
    else:
        text = lines[0].lower()
        if not ("male" in text or "female" in text or "m" in text or "f" in text):
            unusual_chn.append((f.name, lines))

print(f"Unusual CHN lines count: {len(unusual_chn)}")
for item in unusual_chn[:10]:
    print(" ", item)
