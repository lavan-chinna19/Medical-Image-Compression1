"""Inspect all ages and findings in readings."""

from pathlib import Path
import re

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")
clin_dir = DATA_ROOT / "ClinicalReadings"

all_ages_raw = []
all_sexes_raw = []
findings_samples = []

for f in sorted(clin_dir.glob("*.txt")):
    lines = [l.strip() for l in f.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    if f.name.startswith("MCUCXR"):
        sex = None
        age = None
        findings = []
        for l in lines:
            if "Sex:" in l:
                sex = l.split("Sex:", 1)[1].strip()
            elif "Age:" in l:
                age = l.split("Age:", 1)[1].strip()
            else:
                findings.append(l)
        all_ages_raw.append((f.name, age))
        all_sexes_raw.append((f.name, sex))
        findings_samples.append((f.name, " ".join(findings)))
    else:
        # CHN
        if lines:
            first = lines[0]
            rest = " ".join(lines[1:]) if len(lines) > 1 else ""
            all_ages_raw.append((f.name, first))
            findings_samples.append((f.name, rest))

# Let's inspect any unusual ages or patterns
mcu_age_patterns = set(a for name, a in all_ages_raw if name.startswith("MCUCXR"))
print("All MCU age patterns:", mcu_age_patterns)

chn_first_patterns = [a for name, a in all_ages_raw if name.startswith("CHNCXR")]
# Check if any contain days, months, etc.
has_month_or_day = [p for p in chn_first_patterns if any(w in p.lower() for w in ['m', 'd', 'month', 'day'])]
print("CHN with month/day/etc:", has_month_or_day[:10])

# Check all non-standard CHN first lines
non_std_chn = [p for p in chn_first_patterns if not re.match(r'^(male|female)\s+\d+\s*(yrs|yr|y)?$', p.lower().strip())]
print(f"Non-standard CHN first lines ({len(non_std_chn)}):", non_std_chn[:20])
