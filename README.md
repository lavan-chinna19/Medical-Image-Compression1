# Region-Aware Medical Image Compression with Diagnostic Fidelity Evaluation

A comprehensive benchmark and modular Python framework for chest radiograph compression.
This repository implements four custom codecs from scratch—a block DCT codec with canonical Huffman coding,
a tuned 4-level biorthogonal DWT codec, a bit-exact lossless Region-of-Interest (ROI) coder,
and a near-lossless graded multi-zone coder with deterministic bounded error ($L_\infty \le \delta$).
Downstream clinical impact is evaluated on 800 chest X-rays (Shenzhen and Montgomery cohorts)
using an automated U-Net lung segmenter (Dice = 0.9632) and two independent deep neural networks:
a ResNet-18 TB screening model (AUC = 0.9413) and an independent EfficientNet-B0 evaluation judge (AUC = 0.9528).
Whole-image lossy JPEG 2000 maintains classifier AUC within 0.01 down to 0.27 bpp,
proving ~5× more rate-efficient than bit-exact ROI coding (1.33 bpp) for automated CAD pipelines.
Closed-loop bitrate controllers steered by a single classifier fail to generalize to an independent judge,
confirming that single-model adaptive compression is unsafe without multi-model consensus.

---

## Headline Results

Every result is evaluated on out-of-fold patient-stratified test cohorts with 95% bootstrap confidence intervals ($B=1000$):

| Codec / Model Architecture | Operational Setting | Bitrate (bpp) | PSNR (dB) | Full SSIM | ROI SSIM | Classifier AUC [95% CI] | Flip Rate |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Uncompressed Baseline** | 8-bit Raw PACS | 8.0000 | $\infty$ | 1.0000 | 1.0000 | 0.9413 [0.9249, 0.9547] | 0.00% |
| **Standard Lossless PNG** | Level 6 (N=800) | 3.4296 | $\infty$ | 1.0000 | 1.0000 | 0.9413 [0.9249, 0.9547] | 0.00% |
| **Custom 8×8 DCT Codec** | Quality 50 (N=800) | 0.4451 | 40.04 | 0.9525 | 0.9789 | 0.9404 [0.9238, 0.9538] | 3.75% |
| **Tuned DWT Codec** | `bior2.2` Q50 (N=800) | 0.4360 | 40.28 | 0.9610 | 0.9815 | 0.9410 [0.9245, 0.9544] | 4.88% |
| **Standard JPEG 2000** | CR 20 (N=800) | 0.3980 | 44.82 | 0.9858 | 0.9928 | 0.9410 [0.9246, 0.9544] | 3.25% |
| **Lossless ROI Hybrid** | Lungs Exact, DCT Q50 | 1.3267 | 45.18 | 0.9745 | **1.0000** | 0.9402 [0.9236, 0.9536] | 4.12% |
| **Graded Multi-Zone** | $\delta=1$, DCT Q30 | 0.9540 | 46.12 | 0.9780 | 0.9982 | 0.9388 [0.9218, 0.9526] | 5.00% |
| **U-Net Lung Segmentation**| 5-Fold OOF (N=704) | — | — | — | — | **Dice: 0.9632** [0.9602, 0.9660] | **IoU: 0.9302** |
| **EfficientNet-B0 Judge** | Independent OOF | — | — | — | — | **AUC: 0.9528** [0.9378, 0.9664] | **Acc: 89.00%** |

*Note: For detailed results, rate-distortion curves, and full claim verifications, refer to [report_assets/claims_register.md](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/report_assets/claims_register.md) and [report/report.md](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/report/report.md).*

---

## Repository Layout

```
Medical-Image-Compression/
├── app/
│   └── streamlit_app.py        # Interactive Streamlit clinical compression demo
├── data/
│   ├── manifest.csv            # 800-image dataset manifest (sources, labels, masks)
│   └── splits/                 # Stratified 5-fold cross-validation split definitions
├── docs/
│   └── architecture.md         # Architecture specification, bitstream headers & metrics
├── report/
│   ├── report.md               # 25-page comprehensive project research report
│   ├── report.docx             # Publication-styled Word document with embedded figures
│   ├── report.pdf              # Generated print PDF document
│   └── viva_notes.md           # Oral examination study sheet & 15 key numbers
├── report_assets/
│   ├── claims_register.md      # Claims register with 18 verified claims & CIs
│   ├── consistency_audit.md    # Repository consistency audit on numbers and cohorts
│   ├── figures/                # 10 publication figures (300 DPI PNG + vector PDF)
│   └── tables/                 # Clean CSV and Markdown result tables with 95% CIs
├── scripts/
│   ├── build_manifest.py       # Discover raw dataset and build manifest.csv
│   ├── build_report_docs.py    # Compile report.md into report.docx and report.pdf
│   ├── check_report_numbers.py # Automated regex validator for report claims (92.5% pass)
│   ├── make_report_assets.py   # Re-generate all tables, figures, and registers
│   ├── run_pipeline.ps1        # PowerShell pipeline orchestration with skip logic
│   ├── run_pipeline.md         # Pipeline stage guide and expected outputs
│   └── smoke_test.py           # 30-second 10-point test of all codecs and models
├── src/
│   └── medcomp/                # Core Python package (21 modules)
├── tests/                      # Unit & integration test suite (187 test cases)
├── LICENSE                     # MIT Open Source License
├── pytest.ini                  # Pytest runner configuration
├── requirements.txt            # Core dependencies
└── requirements.lock.txt       # Exact frozen dependencies (Python 3.13)
```

---

## Environment Setup

### Prerequisites
- **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, and 3.13)
- Windows PowerShell, Linux Bash, or macOS zsh

### 1. Virtual Environment Setup
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.venv\Scripts\Activate.ps1

# Activate on Linux / macOS:
source .venv/bin/activate
```

### 2. Dependency Installation
Install core dependencies:
```powershell
pip install -r requirements.txt
```
Or install exact locked versions:
```powershell
pip install -r requirements.lock.txt
```

### 3. PyTorch Installation (CUDA GPU vs. CPU)
- **NVIDIA GPU Acceleration (Recommended for Training & Web App):**
  Install PyTorch with CUDA 12.6 wheels:
  ```powershell
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  ```
- **CPU-Only Machines:**
  The standard PyTorch wheels installed via `requirements.txt` run automatically on CPU. Note that training scripts run faster on CUDA GPUs (e.g., NVIDIA RTX 2050 or higher), but all codec transforms, metrics, and smoke tests execute natively on CPU in seconds.

---

## Obtaining the Dataset

This project uses posteroanterior chest radiographs and manual lung segmentation masks from two public research repositories:
1. **Montgomery County X-ray Set (MCUCXR):** 138 radiographs (80 normal, 58 TB) collected by the Montgomery County Department of Health and Human Services, Maryland, USA. 100% of images include manual lung masks.
2. **Shenzhen Hospital X-ray Set (CHNCXR):** 662 radiographs (326 normal, 336 TB) collected by Shenzhen No. 3 People's Hospital, Guangdong, China. 566 images include manual masks; 96 are unmasked.

### Download Instructions
1. Download the unified archive **"Chest X-ray Masks and Labels"** from Kaggle:
   [https://www.kaggle.com/datasets/nikhilpandey333/chest-xray-masks-and-labels](https://www.kaggle.com/datasets/nikhilpandey333/chest-xray-masks-and-labels)
   *(Or download the original NLM Montgomery and Shenzhen archives from the U.S. National Library of Medicine).*
2. Extract the archive into the repository root so that the following directory structure exists:
   ```
   Medical-Image-Compression/
   └── Lung Segmentation/
       ├── Montgomery/
       │   └── ManualMask/
       ├── Shenzhen/
       │   └── mask/
       ├── CXR_png/
       └── ClinicalReadings/
   ```
> [!IMPORTANT]
> Raw dataset files must **never be committed to Git** (enforced by `.gitignore`). The manifest at [`data/manifest.csv`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/data/manifest.csv) tracks cohort assignments, dimensions, and labels without storing image pixels.

---

## Full Pipeline Execution

Run the complete pipeline from data preprocessing to final report generation via PowerShell:
```powershell
.\scripts\run_pipeline.ps1
```
*(The script automatically skips stages whose outputs exist on disk. Use `-Force` to re-execute everything).*

### Step-by-Step Stage Commands

```powershell
# 1. Dataset Preprocessing & Stratified Folds
python scripts/build_manifest.py         # Output: data/manifest.csv
python scripts/preprocess.py             # Output: work/processed/images/, work/processed/masks/
python scripts/make_folds.py             # Output: data/splits/folds.json

# 2. Standard Baseline Evaluation (PNG, JPEG, JPEG 2000)
python scripts/run_baselines.py          # Output: results/baselines.csv
python scripts/summarize_baselines.py    # Output: results/baseline_summary.csv

# 3. Custom DCT Codec Ladder
python scripts/run_dct.py                # Output: results/dct_results.csv
python scripts/summarize_dct.py          # Output: results/dct_summary.csv

# 4. Tuned DWT Codec Ladder & Wavelet Grid Search
python scripts/run_dwt.py                # Output: results/dwt_results.csv
python scripts/summarize_dwt.py          # Output: results/dwt_summary.csv
python scripts/tune_dwt.py               # Output: results/dwt_tuning_summary.csv

# 5. Lossless Lung ROI Coding (MED Predictor)
python scripts/run_lossless.py           # Output: results/lossless_results.csv, results/lossless_summary.csv

# 6. Hybrid ROI Multiplexed Codecs
python scripts/run_roi_codec.py          # Output: results/roi_codec_results.csv, results/roi_codec_summary.csv
python scripts/run_roi_codec_predicted.py # Output: results/roi_codec_predicted_summary.csv

# 7. U-Net Segmentation Training & Cross-Validation
python scripts/train_unet.py             # Output: work/segmentation/ (model checkpoints)
python scripts/evaluate_segmentation.py  # Output: results/segmentation_results.csv, results/segmentation_summary.csv

# 8. TB Classifiers (ResNet-18 Steering Model & EfficientNet-B0 Judge)
python scripts/train_classifier.py       # Output: work/classifier/ (5 fold weights)
python scripts/train_judge.py            # Output: work/judge/ (5 fold weights)
python scripts/evaluate_classifier.py    # Output: results/classifier_metrics.csv, results/judge_oof.csv

# 9. Downstream Rate-Distortion Diagnostic Benchmark
python scripts/run_benchmark.py          # Output: results/benchmark_per_image.csv
python scripts/evaluate_benchmark.py     # Output: results/benchmark_summary.csv, results/benchmark_operating_points.csv

# 10. Graded Multi-Zone Pilot Experiments
python scripts/run_graded_pilot.py       # Output: results/graded_pilot_results.csv
python scripts/summarize_graded_pilot.py # Output: results/graded_pilot_summary.csv

# 11. Closed-Loop Bitrate Controller Evaluation
python scripts/evaluate_controller.py    # Output: results/controller_summary.csv
python scripts/evaluate_controller_variants.py # Output: results/controller_vs_fixed.csv

# 12. Report Assets & Document Compilation
python scripts/make_report_assets.py     # Output: report_assets/tables/, report_assets/figures/
python scripts/build_report_docs.py      # Output: report/report.docx, report/report.pdf
python scripts/check_report_numbers.py   # Output: Numeric audit pass verification
```

---

## Testing & Quality Assurance

### 1. Full PyTest Suite
Run the 187-case unit and integration test suite:
```powershell
pytest
```
Expected output:
```text
================ 170 passed, 17 skipped, 36 warnings in 26.00s ================
```

### 2. End-to-End Smoke Test
Run the standalone 10-point end-to-end smoke test on 5 stratified images (runs in ~20 seconds without requiring raw data):
```powershell
python scripts/smoke_test.py
```
Expected output:
```text
[PASS] PNG Baseline (Encode/Decode & Bit-Exact Lossless)
[PASS] JPEG Baseline (Encode/Decode from bytes)
[PASS] JPEG 2000 Baseline (OpenJPEG DWT from bytes)
[PASS] Custom DCT Codec (8x8 Transform + Canonical Huffman)
[PASS] Tuned DWT Codec (bior2.2 4-Level + Dead-Zone Quantization)
[PASS] Lossless ROI Codec (Bit-Exact MED Prediction on Lung Mask)
[PASS] ROI Hybrid Codec (Multiplexed Bit-Exact ROI + Lossy Background)
[PASS] Graded Multi-Zone Codec (Strict Error Bounds |e| <= delta)
[PASS] Bitstream Container Size Integrity (Reported Size == len(bytes))
[PASS] Classifier Inference (TB Probabilities Strictly in [0, 1])
TOTAL: 10/10 checks passed in 21.27 seconds.
```

---

## Launching the Interactive Web Demo

Launch the interactive clinical compression web application:
```powershell
streamlit run app/streamlit_app.py
```

The web interface features:
- Upload any DICOM or 8-bit/16-bit radiograph.
- Interactive mode selection: Full Lossy, Bit-Exact Lossless ROI, Graded Multi-Zone, or Adaptive Controller.
- Real-time U-Net lung segmentation with epistemic MC dropout uncertainty heatmaps.
- Side-by-side original vs. reconstructed visual comparisons with error residual magnification ($10\times$).
- Real-time diagnostic impact evaluation comparing ResNet-18 vs. EfficientNet-B0 probability shifts and decision flips.

---

## Citations

If you use this benchmark or codebase, please cite the underlying datasets and frameworks:

```bibtex
@article{jaeger2014two,
  title={Two public chest X-ray datasets for computer-aided screening of pulmonary diseases},
  author={Jaeger, Stefan and Candemir, Sema and Antani, Sameer and W{\'a}ng, Y{\`\i}-Xi{\'a}ng J and Lu, Pu-Xuan and Thoma, George},
  journal={Quantitative Imaging in Medicine and Surgery},
  volume={4},
  number={6},
  pages={475},
  year={2014}
}

@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  pages={234--241},
  year={2015}
}

@inproceedings{he2016deep,
  title={Deep residual learning for image recognition},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle={CVPR},
  pages={770--778},
  year={2016}
}
```

---

## Research Prototype Disclaimer

> [!CAUTION]
> **RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL USE**  
> This software is an academic research prototype developed strictly for evaluating data compression theory, rate-distortion tradeoffs, and downstream computer-aided diagnostic sensitivity. It has **not** been cleared, certified, or approved by the U.S. Food and Drug Administration (FDA), the European Medicines Agency (EMA), or any other medical regulatory authority. Do **not** use this software for primary clinical interpretation, real-time patient diagnosis, treatment planning, or clinical PACS archival.

---

## License & Data Terms

- **Source Code:** Released under the [MIT License](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/LICENSE).
- **Dataset Terms:** The chest radiographs and lung masks from Montgomery County and Shenzhen Hospital are subject to their respective original data licenses and terms of use from the U.S. National Library of Medicine, NIH, and the Shenzhen No. 3 People's Hospital.
