# Pipeline Execution Guide (`scripts/run_pipeline.ps1`)

This guide details how to reproduce, execute, and verify every experimental stage of the medical image compression project using the PowerShell orchestration script.

---

## 1. Quick Start

Run all stages (automatically skipping stages whose output artifacts already exist):
```powershell
.\scripts\run_pipeline.ps1
```

Force re-execution of all stages from scratch:
```powershell
.\scripts\run_pipeline.ps1 -Force
```

Run a specific subset of stages:
```powershell
.\scripts\run_pipeline.ps1 -Stage assets
.\scripts\run_pipeline.ps1 -Stage baselines, dct, dwt
```

---

## 2. Pipeline Stages & Output Artifacts

| Stage | Command Executed | Primary Output Files | Description |
|:---|:---|:---|:---|
| `data` | `build_manifest.py`, `preprocess.py`, `make_folds.py` | `data/manifest.csv`<br>`data/splits/folds.json` | Parses raw images, normalizes to 512×512, builds stratified 5-fold cross-validation. |
| `baselines` | `run_baselines.py`, `summarize_baselines.py` | `results/baselines.csv`<br>`results/baseline_summary.csv` | Standard codecs: PNG, baseline JPEG (Q10–90), and JPEG 2000 (CR 5–80). |
| `dct` | `run_dct.py`, `summarize_dct.py` | `results/dct_results.csv`<br>`results/dct_summary.csv` | Custom 8×8 DCT with JPEG quantization scaling and canonical Huffman coding. |
| `dwt` | `run_dwt.py`, `summarize_dwt.py`, `tune_dwt.py` | `results/dwt_results.csv`<br>`results/dwt_tuning_summary.csv` | 4-level `bior2.2` DWT with dead-zone quantization and wavelet tuning comparison. |
| `lossless` | `run_lossless.py` | `results/lossless_results.csv`<br>`results/lossless_summary.csv` | Bit-exact MED predictor lossless lung coding on the 704 masked images. |
| `roi` | `run_roi_codec.py`, `run_roi_codec_predicted.py` | `results/roi_codec_results.csv`<br>`results/roi_codec_summary.csv` | Multiplexed ROI container with ground-truth and U-Net predicted masks. |
| `segmentation` | `train_unet.py`, `evaluate_segmentation.py` | `results/segmentation_results.csv`<br>`results/segmentation_summary.csv` | 5-fold cross-validation U-Net segmentation with Monte Carlo dropout. |
| `classifier` | `train_classifier.py`, `train_judge.py`, `evaluate_classifier.py` | `results/classifier_metrics.csv`<br>`results/judge_oof.csv` | ResNet-18 steering model and EfficientNet-B0 evaluation judge training. |
| `benchmark` | `run_benchmark.py`, `evaluate_benchmark.py` | `results/benchmark_summary.csv`<br>`results/benchmark_operating_points.csv` | Rate-distortion diagnostic evaluation measuring AUC degradation and flip rates. |
| `graded` | `run_graded_pilot.py`, `summarize_graded_pilot.py` | `results/graded_pilot_results.csv`<br>`results/graded_pilot_summary.csv` | Near-lossless graded multi-zone coding ($\delta \in \{0, 1, 2, 3\}$). |
| `controller` | `evaluate_controller.py`, `evaluate_controller_variants.py` | `results/controller_summary.csv`<br>`results/controller_vs_fixed.csv` | Closed-loop adaptive bitrate controllers (V0–V4) evaluated against the judge. |
| `assets` | `make_report_assets.py`, `build_report_docs.py`, `check_report_numbers.py` | `report_assets/claims_register.md`<br>`report/report.md`<br>`report/report.docx`<br>`report/report.pdf` | Audited Markdown tables, 300 DPI figures, comprehensive report, and DOCX/PDF builds. |

---

## 3. Execution Logging

All stage outputs and errors are automatically logged to the `logs/` directory:
- `logs/<stage>.log`: Standard output stream with timestamps.
- `logs/<stage>_err.log`: Standard error stream.
