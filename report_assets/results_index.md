# MedComp Results Index: Tables & Figures Mapping

This index provides a comprehensive cross-reference mapping each generated table and figure in `report_assets/` to its primary result source file and the pipeline script that generated it.

---

## 1. Clean Report Tables (`report_assets/tables/`)

| Table Name | Output Formats | Primary Source File in `results/` or `data/` | Original Producer Script | Key Metrics / Contents |
|:---|:---|:---|:---|:---|
| **dataset_overview** | `.csv`, `.md` | `data/manifest.csv` | `scripts/build_manifest.py` | Image counts, normal/TB prevalence, mask availability |
| **baseline_summary** | `.csv`, `.md` | `results/baseline_summary.csv` | `scripts/summarize_baselines.py` | Baseline PNG, JPEG, JPEG 2000 rate-distortion metrics |
| **dct_summary** | `.csv`, `.md` | `results/dct_summary.csv` | `scripts/summarize_dct.py` | Custom 8x8 DCT codec rate, distortion, and Huffman code lengths |
| **dwt_summary** | `.csv`, `.md` | `results/dwt_summary.csv` | `scripts/summarize_dwt.py` | Tuned bior2.2 L4 DWT codec rate, distortion, and timing |
| **dwt_tuning_summary** | `.csv`, `.md` | `results/dwt_tuning_summary.csv` | `scripts/tune_dwt.py` | BD-PSNR gains across wavelets, deadzone, weights, context |
| **dwt_ablation_summary**| `.csv`, `.md` | `results/dwt_ablation_summary.csv` | `scripts/run_dwt_ablation.py` | 24-configuration wavelet, level, and quantization table ablation |
| **dwt_ablation** | `.csv`, `.md` | `results/dwt_ablation.csv` | `scripts/run_dwt_ablation.py` | Per-configuration aggregate metrics |
| **lossless_summary** | `.csv`, `.md` | `results/lossless_summary.csv` | `scripts/run_lossless.py` | Context-adaptive lossless ROI coding and mask overhead |
| **roi_codec_summary** | `.csv`, `.md` | `results/roi_codec_summary.csv` | `scripts/run_roi_codec.py` | Hybrid lossless ROI + lossy background performance vs PNG |
| **roi_fill_ablation** | `.csv`, `.md` | `results/roi_fill_ablation.csv` | `scripts/run_roi_fill_ablation.py`| Boundary dilation and fill ablation byte comparison |
| **segmentation_summary**| `.csv`, `.md` | `results/segmentation_summary.csv` | `scripts/evaluate_segmentation.py`| 5-Fold out-of-fold U-Net Dice, IoU, recall, precision |
| **segmentation_margins**| `.csv`, `.md` | `results/segmentation_margins.csv` | `scripts/evaluate_segmentation.py`| Morphological dilation and uncertainty threshold margins |
| **roi_codec_predicted** | `.csv`, `.md` | `results/roi_codec_predicted_summary.csv`| `scripts/run_roi_codec_predicted.py`| Predicted mask hybrid codec leakage and true lung PSNR |
| **classifier_metrics** | `.csv`, `.md` | `results/classifier_metrics.csv` | `scripts/evaluate_classifier.py` | 5-Fold ResNet18 OOF accuracy, recall, specificity, AUC |
| **classifier_crossscanner**| `.csv`, `.md`| `results/classifier_crossscanner.csv`| `scripts/evaluate_classifier.py` | Cross-domain transfer (Shenzhen <-> Montgomery) |
| **judge_metrics** | `.csv`, `.md` | `results/judge_metrics.csv` | `scripts/evaluate_classifier.py` | EfficientNet-B0 independent judge performance |
| **benchmark_summary** | `.csv`, `.md` | `results/benchmark_summary.csv` | `scripts/evaluate_benchmark.py` | Downstream task-based compression benchmark across 5 codecs |
| **benchmark_per_source**| `.csv`, `.md` | `results/benchmark_summary_per_source.csv`| `scripts/evaluate_benchmark.py`| Benchmark metrics partitioned by clinical site |
| **benchmark_operating_points**| `.csv`, `.md`| `results/benchmark_operating_points.csv`| `scripts/evaluate_benchmark.py`| Bitrate thresholds for <=5% flip, <=2% flip, <=0.01 AUC loss |
| **region_ablation_summary**| `.csv`, `.md`| `results/region_ablation_summary.csv`| `scripts/evaluate_region_ablation.py`| Lungs-lossy vs Background-lossy vs All-lossy ablation |
| **region_ablation_roi_comparison**| `.csv`, `.md`| `results/region_ablation_roi_comparison.csv`| `scripts/evaluate_region_ablation.py`| Paired differences between background and ROI ablation |
| **graded_pilot_summary**| `.csv`, `.md` | `results/graded_pilot_summary.csv` | `scripts/summarize_graded_pilot.py`| Graded near-lossless pilot (120 images) across delta cores |
| **graded_pilot_guarantee_cost**| `.csv`, `.md`| `results/graded_pilot_guarantee_cost.csv`| `scripts/summarize_graded_pilot.py`| Bitrate required for 99th percentile error guarantees |
| **controller_summary** | `.csv`, `.md` | `results/controller_summary.csv` | `scripts/evaluate_controller.py` | Closed-loop rate controller steering vs judge evaluations |
| **controller_vs_fixed**| `.csv`, `.md` | `results/controller_vs_fixed.csv` | `scripts/evaluate_controller.py` | Controller byte savings vs fixed operating curves |
| **controller_variants_summary**| `.csv`, `.md`| `results/controller_variants_vs_fixed.csv` & `results/controller_variants_summary.txt`| `scripts/evaluate_controller_variants.py`| Diagnostic variants V0-V4 and oracle bounds |

---

## 2. Publication-Quality Figures (`report_assets/figures/`)

All figures are rendered at 300 dpi (PNG) and as vector graphics (PDF) using a unified, codec-consistent palette:
- **JPEG:** `#1f77b4` (Blue)
- **JPEG 2000:** `#ff7f0e` (Orange)
- **Custom DCT:** `#d62728` (Red)
- **Tuned DWT:** `#2ca02c` (Green)
- **ROI Hybrid:** `#9467bd` (Purple)

| Figure Filename | Formats | Primary Source Files | Contents / Visualization |
|:---|:---|:---|:---|
| **fig_rd_psnr** | `.png`, `.pdf` | `results/baseline_summary.csv`, `results/dct_summary.csv`, `results/dwt_summary.csv` | Rate-Distortion: PSNR [dB] vs Bitrate [bpp] across 4 codecs |
| **fig_rd_ssim** | `.png`, `.pdf` | `results/baseline_summary.csv`, `results/dct_summary.csv`, `results/dwt_summary.csv` | Rate-Distortion: Full-image SSIM vs Bitrate [bpp] |
| **fig_rd_roi_ssim** | `.png`, `.pdf` | `results/baseline_summary.csv`, `results/dct_summary.csv`, `results/dwt_summary.csv`, `results/roi_codec_summary.csv` | Lung ROI Quality: ROI SSIM vs Bitrate including exact ROI hybrid |
| **fig_benchmark_auc_vs_bpp** | `.png`, `.pdf` | `results/benchmark_summary.csv` | Task Utility: Downstream TB Classifier AUC vs Bitrate with 95% CIs |
| **fig_benchmark_flips_vs_bpp** | `.png`, `.pdf` | `results/benchmark_summary.csv` | Task Reliability: Prediction Flip Rate [%] vs Bitrate with target lines |
| **fig_segmentation_margins_dice_recall**| `.png`, `.pdf`| `results/segmentation_margins.csv`, `results/roi_codec_predicted_summary.csv` | Lung Safety: Margin Dilation [px] vs Coverage [%] & Leakage [%] |
| **fig_region_ablation_quality**| `.png`, `.pdf` | `results/region_ablation_summary.csv` | Regional Sensitivity: Flip Rate [%] for Lungs vs Background vs All |
| **fig_graded_pilot_tradeoff** | `.png`, `.pdf` | `results/graded_pilot_summary.csv` | Graded Pilot: Core Error Bound $\delta$ vs Probability Drift |
| **fig_controller_vs_fixed** | `.png`, `.pdf` | `results/controller_vs_fixed.csv` | Rate Controller: Judge Flip Rate vs Bitrate across families |
| **fig_montgomery_ssim_dip** | `.png`, `.pdf` | `results/dct_summary.csv` | The Montgomery Artifact: Collimator Flat-Black Noise Dip at Q=70 |

---

## 3. Audits and Claims Registers

- **Consistency Audit:** `report_assets/consistency_audit.md`
- **Claims Register:** `report_assets/claims_register.csv` and `report_assets/claims_register.md`
