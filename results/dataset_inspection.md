# Kaggle "Chest X-ray Masks and Labels" Dataset Inspection Report

**Dataset Path:** `C:\project\medical\Lung Segmentation`  
**Inspection Date:** 2026-10-06  
**Status:** Read-only, unedited, unmodified.

---

## 1. Folder Tree (Top 3 Levels) with File Counts & Extensions

```
Lung Segmentation/
├── [2 files]
│   ├── .docx: 1
│   └── .pdf: 1
├── .ipynb_checkpoints/ [1 file]
│   └── .ipynb: 1
├── ClinicalReadings/ [800 files]
│   └── .txt: 800
├── CXR_png/ [800 files]
│   └── .png: 800
├── masks/ [704 files]
│   └── .png: 704
└── test/ [96 files]
    └── .png: 96
```

### File Counts Summary per Folder

| Folder Relative Path | Total Files | Extension Breakdown |
|---|---|---|
| `.` (Root) | 2 | 1 `.docx`, 1 `.pdf` |
| `.ipynb_checkpoints` | 1 | 1 `.ipynb` |
| `ClinicalReadings` | 800 | 800 `.txt` |
| `CXR_png` | 800 | 800 `.png` |
| `masks` | 704 | 704 `.png` |
| `test` | 96 | 96 `.png` |
| **Total Files** | **2,403** | **1,600 `.png`, 800 `.txt`, 1 `.docx`, 1 `.pdf`, 1 `.ipynb`** |

### 3 Sample Filenames from Every Folder

- **Root (`Lung Segmentation/`):**
  1. `NLM-ChinaCXRSet-ReadMe.docx`
  2. `NLM-MontgomeryCXRSet-ReadMe.pdf`
- **`.ipynb_checkpoints/`:**
  1. `Montgomery-checkpoint.ipynb`
- **`ClinicalReadings/`:**
  1. `CHNCXR_0001_0.txt`
  2. `CHNCXR_0002_0.txt`
  3. `CHNCXR_0003_0.txt`
- **`CXR_png/`:**
  1. `CHNCXR_0001_0.png`
  2. `CHNCXR_0002_0.png`
  3. `CHNCXR_0003_0.png`
- **`masks/`:**
  1. `CHNCXR_0001_0_mask.png`
  2. `CHNCXR_0002_0_mask.png`
  3. `CHNCXR_0003_0_mask.png`
- **`test/`:**
  1. `CHNCXR_0025_0.png`
  2. `CHNCXR_0036_0.png`
  3. `CHNCXR_0037_0.png`

---

## 2. Chest X-Ray Images & Masks Count per Source

| Source Dataset | Image Prefix | Chest X-Ray Images (`CXR_png/`) | Masks Count (`masks/`) | Difference (Unmasked) |
|---|---|---|---|---|
| **Montgomery County (MCUCXR)** | `MCUCXR_` | 138 | 138 | 0 |
| **Shenzhen Hospital (CHNCXR)** | `CHNCXR_` | 662 | 566 | 96 |
| **Total** | | **800** | **704** | **96** |

---

## 3. Mask-to-Image Filename Mapping & Structure

### Montgomery County (`MCUCXR`)
- **Relationship:** Exact match. Mask filename is identical to image filename.  
  `CXR_png/MCUCXR_0001_0.png` $\longleftrightarrow$ `masks/MCUCXR_0001_0.png`
- **Single vs. Separate Masks:** In this Kaggle release, Montgomery masks are **single combined masks** (1 mask per image containing both left and right lung lobes). Unlike the original raw NIH Montgomery distribution which stored separate `leftMask` and `rightMask` folders, this dataset has already merged them into unified 2D binary masks (connected component count = 3: background + 2 lung lobes; values $\in \{0, 255\}$).

### Shenzhen Hospital (`CHNCXR`)
- **Relationship:** Suffix pattern. Mask filenames append `_mask.png` before the extension.  
  `CXR_png/CHNCXR_0001_0.png` $\longleftrightarrow$ `masks/CHNCXR_0001_0_mask.png`
- **Single vs. Separate Masks:** Shenzhen masks are **single combined masks** (1 mask per image containing both left and right lung fields; values $\in \{0, 255\}$).

---

## 4. Missing Masks and Orphaned Masks

### Masks without an image in `CXR_png/`
- **Count:** 0 (None). All 704 masks in `masks/` correspond to a valid image in `CXR_png/`.

### Images without a mask in `masks/`
- **Montgomery images without mask:** 0 (All 138 images have masks).
- **Shenzhen images without mask:** Exactly 96 images.
- **Complete List of Unmasked Images (96 files):**
  1. `CHNCXR_0025_0.png`
  2. `CHNCXR_0036_0.png`
  3. `CHNCXR_0037_0.png`
  4. `CHNCXR_0038_0.png`
  5. `CHNCXR_0039_0.png`
  6. `CHNCXR_0040_0.png`
  7. `CHNCXR_0065_0.png`
  8. `CHNCXR_0181_0.png`
  9. `CHNCXR_0182_0.png`
  10. `CHNCXR_0183_0.png`
  11. `CHNCXR_0184_0.png`
  12. `CHNCXR_0185_0.png`
  13. `CHNCXR_0186_0.png`
  14. `CHNCXR_0187_0.png`
  15. `CHNCXR_0188_0.png`
  16. `CHNCXR_0189_0.png`
  17. `CHNCXR_0190_0.png`
  18. `CHNCXR_0191_0.png`
  19. `CHNCXR_0192_0.png`
  20. `CHNCXR_0193_0.png`
  21. `CHNCXR_0194_0.png`
  22. `CHNCXR_0195_0.png`
  23. `CHNCXR_0196_0.png`
  24. `CHNCXR_0197_0.png`
  25. `CHNCXR_0198_0.png`
  26. `CHNCXR_0199_0.png`
  27. `CHNCXR_0200_0.png`
  28. `CHNCXR_0201_0.png`
  29. `CHNCXR_0202_0.png`
  30. `CHNCXR_0203_0.png`
  31. `CHNCXR_0204_0.png`
  32. `CHNCXR_0205_0.png`
  33. `CHNCXR_0206_0.png`
  34. `CHNCXR_0207_0.png`
  35. `CHNCXR_0208_0.png`
  36. `CHNCXR_0209_0.png`
  37. `CHNCXR_0210_0.png`
  38. `CHNCXR_0211_0.png`
  39. `CHNCXR_0212_0.png`
  40. `CHNCXR_0213_0.png`
  41. `CHNCXR_0214_0.png`
  42. `CHNCXR_0215_0.png`
  43. `CHNCXR_0216_0.png`
  44. `CHNCXR_0217_0.png`
  45. `CHNCXR_0218_0.png`
  46. `CHNCXR_0219_0.png`
  47. `CHNCXR_0220_0.png`
  48. `CHNCXR_0336_1.png`
  49. `CHNCXR_0341_1.png`
  50. `CHNCXR_0342_1.png`
  51. `CHNCXR_0343_1.png`
  52. `CHNCXR_0344_1.png`
  53. `CHNCXR_0345_1.png`
  54. `CHNCXR_0346_1.png`
  55. `CHNCXR_0347_1.png`
  56. `CHNCXR_0348_1.png`
  57. `CHNCXR_0349_1.png`
  58. `CHNCXR_0350_1.png`
  59. `CHNCXR_0351_1.png`
  60. `CHNCXR_0352_1.png`
  61. `CHNCXR_0353_1.png`
  62. `CHNCXR_0354_1.png`
  63. `CHNCXR_0355_1.png`
  64. `CHNCXR_0356_1.png`
  65. `CHNCXR_0357_1.png`
  66. `CHNCXR_0358_1.png`
  67. `CHNCXR_0359_1.png`
  68. `CHNCXR_0360_1.png`
  69. `CHNCXR_0481_1.png`
  70. `CHNCXR_0482_1.png`
  71. `CHNCXR_0483_1.png`
  72. `CHNCXR_0484_1.png`
  73. `CHNCXR_0485_1.png`
  74. `CHNCXR_0486_1.png`
  75. `CHNCXR_0487_1.png`
  76. `CHNCXR_0488_1.png`
  77. `CHNCXR_0489_1.png`
  78. `CHNCXR_0490_1.png`
  79. `CHNCXR_0491_1.png`
  80. `CHNCXR_0492_1.png`
  81. `CHNCXR_0493_1.png`
  82. `CHNCXR_0494_1.png`
  83. `CHNCXR_0495_1.png`
  84. `CHNCXR_0496_1.png`
  85. `CHNCXR_0497_1.png`
  86. `CHNCXR_0498_1.png`
  87. `CHNCXR_0499_1.png`
  88. `CHNCXR_0500_1.png`
  89. `CHNCXR_0502_1.png`
  90. `CHNCXR_0505_1.png`
  91. `CHNCXR_0560_1.png`
  92. `CHNCXR_0561_1.png`
  93. `CHNCXR_0562_1.png`
  94. `CHNCXR_0563_1.png`
  95. `CHNCXR_0564_1.png`
  96. `CHNCXR_0565_1.png`

*(Note: These 96 unmasked files correspond identically to the 96 files located inside the `test/` folder.)*

---

## 5. First 20 Images Characteristics per Source

### Montgomery County (`MCUCXR`)

| # | Filename | Size ($H \times W$) | Mode | Bit Depth |
|---|---|---|---|---|
| 1 | `MCUCXR_0001_0.png` | $4020 \times 4892$ | Grayscale | 8-bit |
| 2 | `MCUCXR_0002_0.png` | $4020 \times 4892$ | Grayscale | 8-bit |
| 3 | `MCUCXR_0003_0.png` | $4892 \times 4020$ | Grayscale | 8-bit |
| 4 | `MCUCXR_0004_0.png` | $4892 \times 4020$ | Grayscale | 8-bit |
| 5 | `MCUCXR_0005_0.png` | $4892 \times 4020$ | Grayscale | 8-bit |
| 6 | `MCUCXR_0006_0.png` | $4020 \times 4892$ | Grayscale | 8-bit |
| 7 | `MCUCXR_0008_0.png` | $4892 \times 4020$ | Grayscale | 8-bit |
| 8 | `MCUCXR_0011_0.png` | $4892 \times 4020$ | Grayscale | 8-bit |
| 9 | `MCUCXR_0013_0.png` | $4020 \times 4892$ | Grayscale | 8-bit |
| 10 | `MCUCXR_0015_0.png` | $4020 \times 4892$ | Grayscale | 8-bit |
| 11 | `MCUCXR_0016_0.png` | $4020 \times 4892$ | Grayscale | 8-bit |
| 12 | `MCUCXR_0017_0.png` | $4020 \times 4892$ | Grayscale | 8-bit |
| 13 | `MCUCXR_0019_0.png` | $4892 \times 4020$ | Grayscale | 8-bit |
| 14 | `MCUCXR_0020_0.png` | $4020 \times 4892$ | Grayscale | 8-bit |
| 15 | `MCUCXR_0021_0.png` | $4892 \times 4020$ | Grayscale | 8-bit |
| 16 | `MCUCXR_0022_0.png` | $4020 \times 4892$ | Grayscale | 8-bit |
| 17 | `MCUCXR_0023_0.png` | $4020 \times 4892$ | Grayscale | 8-bit |
| 18 | `MCUCXR_0024_0.png` | $4892 \times 4020$ | Grayscale | 8-bit |
| 19 | `MCUCXR_0026_0.png` | $4892 \times 4020$ | Grayscale | 8-bit |
| 20 | `MCUCXR_0027_0.png` | $4892 \times 4020$ | Grayscale | 8-bit |

### Shenzhen Hospital (`CHNCXR`)

| # | Filename | Size ($H \times W$) | Mode | Bit Depth |
|---|---|---|---|---|
| 1 | `CHNCXR_0001_0.png` | $2919 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 2 | `CHNCXR_0002_0.png` | $2951 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 3 | `CHNCXR_0003_0.png` | $2945 \times 2987$ | Palette (Indexed RGB) | 8-bit |
| 4 | `CHNCXR_0004_0.png` | $2933 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 5 | `CHNCXR_0005_0.png` | $2933 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 6 | `CHNCXR_0006_0.png` | $2948 \times 2775$ | Palette (Indexed RGB) | 8-bit |
| 7 | `CHNCXR_0007_0.png` | $2320 \times 2306$ | Palette (Indexed RGB) | 8-bit |
| 8 | `CHNCXR_0008_0.png` | $2937 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 9 | `CHNCXR_0009_0.png` | $2942 \times 2516$ | Palette (Indexed RGB) | 8-bit |
| 10 | `CHNCXR_0010_0.png` | $2947 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 11 | `CHNCXR_0011_0.png` | $2921 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 12 | `CHNCXR_0012_0.png` | $2959 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 13 | `CHNCXR_0013_0.png` | $2937 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 14 | `CHNCXR_0014_0.png` | $2919 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 15 | `CHNCXR_0015_0.png` | $2941 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 16 | `CHNCXR_0016_0.png` | $2943 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 17 | `CHNCXR_0017_0.png` | $2908 \times 2789$ | Palette (Indexed RGB) | 8-bit |
| 18 | `CHNCXR_0018_0.png` | $2907 \times 2991$ | Palette (Indexed RGB) | 8-bit |
| 19 | `CHNCXR_0019_0.png` | $2941 \times 3000$ | Palette (Indexed RGB) | 8-bit |
| 20 | `CHNCXR_0020_0.png` | $2555 \times 2760$ | Palette (Indexed RGB) | 8-bit |

---

## 6. Format of Clinical Reading Files

Clinical readings are stored as individual `.txt` files matching image stems in `ClinicalReadings/` (800 files total).

### Montgomery Samples (`MCUCXR`)

**Sample 1 (`MCUCXR_0001_0.txt`):**
```text
Patient's Sex: F 
Patient's Age: 027Y
normal
```

**Sample 2 (`MCUCXR_0002_0.txt`):**
```text
Patient's Sex: F 
Patient's Age: 040Y
normal
```

### Shenzhen Samples (`CHNCXR`)

**Sample 1 (`CHNCXR_0001_0.txt`):**
```text
male 45yrs
normal
```

**Sample 2 (`CHNCXR_0002_0.txt`):**
```text
male 63yrs
normal
```

---

## 7. Label Convention & Class Counts

Filenames follow the convention:
`<SOURCE>_<ID>_<LABEL>.png`
- `_0`: **Normal** (negative for tuberculosis / healthy)
- `_1`: **Abnormal / TB** (active tuberculosis manifestation)

### Class Breakdown

| Source | Normal (`_0`) | TB / Abnormal (`_1`) | Total |
|---|---|---|---|
| **Montgomery (MCUCXR)** | 80 | 58 | 138 |
| **Shenzhen (CHNCXR)** | 326 | 336 | 662 |
| **Combined** | **406** (50.75%) | **394** (49.25%) | **800** |

---

## 8. Non-Image Junk Files & Artifacts

- **`.ipynb_checkpoints/Montgomery-checkpoint.ipynb`**: Leftover Jupyter notebook checkpoint inside the dataset root directory.
- No `Thumbs.db`, `desktop.ini`, or `.DS_Store` files were found in the raw dataset directories.
- Two documentation files reside at the dataset root:
  - `NLM-ChinaCXRSet-ReadMe.docx` (52,996 bytes)
  - `NLM-MontgomeryCXRSet-ReadMe.pdf` (104,966 bytes)

---

## 9. Inconsistencies, Anomalies & Key Findings

1. **Test Split and Missing Masks:**
   - There are exactly 96 Shenzhen images in `CXR_png/` that have **no mask** in `masks/`.
   - The directory `test/` contains exactly these 96 images.
   - Verification reveals that the pixel arrays in `test/` and `CXR_png/` are identical (max diff = 0.0), but their file hashes differ due to different PNG compression encodings. These 96 images represent the withheld test evaluation set from the original challenge.
2. **Color Space & Storage Format Differences:**
   - **Montgomery images** are encoded as native 8-bit single-channel Grayscale PNGs (PNG color type 0).
   - **Shenzhen images** are encoded as 8-bit Palette/Indexed RGB PNGs (PNG color type 3, decoded into 3 channels).
   - *Implication for Pipeline:* All loaders must normalize images to single-channel 2D `uint8` arrays (as implemented in `medcomp.io_utils.load_image`).
3. **Mask Filename Disparity:**
   - Montgomery masks share the exact filename with their image (`MCUCXR_0001_0.png`).
   - Shenzhen masks insert `_mask` before the extension (`CHNCXR_0001_0_mask.png`).
4. **Resolution & Orientation Variation:**
   - Montgomery images have very high resolution (~$4000 \times 4892$ or $4892 \times 4020$), alternating between portrait and landscape orientations.
   - Shenzhen images have square-ish resolutions (~$3000 \times 3000$ or ~$2900 \times 3000$).
   - Both require adaptive block-padding (e.g. `pad_to_multiple`) for block-based DCT and multiresolution DWT pipelines.
5. **Pre-Merged Montgomery Masks:**
   - The original NIH Montgomery release provides two separate files per subject (`leftMask/` and `rightMask/`). In this Kaggle release, they are already unified into a single combined mask.
