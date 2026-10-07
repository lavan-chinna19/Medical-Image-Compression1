from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
stems = ["CHNCXR_0197_0", "CHNCXR_0202_0", "CHNCXR_0205_0", "CHNCXR_0215_0", "CHNCXR_0218_0"]
fig, axes = plt.subplots(2, len(stems), figsize=(22, 9))
for i, s in enumerate(stems):
    img = np.array(Image.open(ROOT / "work/processed/images" / f"{s}.png").convert("L"))
    m = np.array(Image.open(ROOT / "work/pred/mask" / f"{s}.png")) > 0
    unc = np.array(Image.open(ROOT / "work/pred/unc" / f"{s}.png").convert("L"))
    axes[0, i].imshow(img, cmap="gray")
    axes[0, i].contour(m.astype(float), levels=[0.5], colors="red", linewidths=1)
    axes[0, i].set_title(f"{s}  area={m.mean():.3f}", fontsize=9)
    axes[1, i].imshow(unc, cmap="magma")
    axes[1, i].set_title(f"uncertainty max={int(unc.max())}", fontsize=9)
    for a in (axes[0, i], axes[1, i]):
        a.axis("off")
fig.savefig(ROOT / "results" / "mask_review_outliers.png", dpi=110, bbox_inches="tight")
print("saved results/mask_review_outliers.png")
