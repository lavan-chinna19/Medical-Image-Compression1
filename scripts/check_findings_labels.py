"""Check findings text vs label suffix."""

from pathlib import Path
import re

DATA_ROOT = Path(r"C:\project\medical\Lung Segmentation")
clin_dir = DATA_ROOT / "ClinicalReadings"

disagreements = []

for f in sorted(clin_dir.glob("*.txt")):
    stem = f.stem
    label = int(stem.rsplit("_", 1)[-1])
    lines = [l.strip() for l in f.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    if f.name.startswith("MCUCXR"):
        # Finding is everything after Sex and Age
        findings_lines = [l for l in lines if not l.startswith("Patient's Sex:") and not l.startswith("Patient's Age:")]
        findings = " ".join(findings_lines).strip()
    else:
        # Shenzhen: first line is demographics, rest is findings
        findings = " ".join(lines[1:]).strip() if len(lines) > 1 else ""
    
    # Check if findings indicate normal
    is_normal = (findings.lower() == "normal") or ("normal" in findings.lower() and not any(k in findings.lower() for k in ["tb", "tuberculosis", "infiltrat", "cavity", "effusion", "nodule", "opacity", "lesion", "abnormal", "thickening"]))
    # Or simply: does findings.lower().strip() == "normal"?
    # The prompt says: "(the reading's findings text says "normal" or something else)"
    says_normal = (findings.lower().strip() == "normal")
    
    # If label == 0 (normal) but findings does not say normal
    # Or label == 1 (TB) but findings says normal
    if (label == 0 and not says_normal) or (label == 1 and says_normal):
        disagreements.append((stem, label, says_normal, findings))

print(f"Total potential mismatches if strictly == 'normal': {len(disagreements)}")
for d in disagreements[:15]:
    print(d)
