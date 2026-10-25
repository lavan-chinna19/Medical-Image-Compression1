"""Streamlit Multipage: Comparative Analysis of Medical Image Compression Techniques.

Compares standard baseline techniques (DCT, DWT, JPEG, JPEG2000, Hybrid DCT-DWT)
against the proposed Region-Aware Medical Image Compression pipeline across 9 metrics:
1. Accuracy (%) -> "N/A" (Image reconstruction is a pixel regression task without binary class boundaries)
2. Precision (%) -> "N/A"
3. Recall (%) -> "N/A"
4. R² Score -> Coefficient of determination between original and reconstructed pixel intensities
5. PSNR (dB) -> Peak Signal-to-Noise Ratio (dB)
6. SSIM -> Structural Similarity Index
7. MSE -> Mean Squared Error
8. Compression Ratio -> Original raw bytes / Compressed bitstream bytes
9. Reconstruction Time (s) -> Exact measured decompression execution time in seconds
"""

from __future__ import annotations

from datetime import datetime
import io
from pathlib import Path
import sys
import time
from typing import Any, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medcomp.app_logic import (
    is_dataset_sample,
    load_and_preprocess_uploaded_file,
    load_models,
    segment,
)
from medcomp.comparison import (
    TECHNIQUE_FULL_NAMES,
    TECHNIQUE_NAMES,
    TECHNIQUE_ORDER,
    TechniqueResult,
    evaluate_technique,
    get_initial_comparison_df,
    run_all_techniques,
)
from medcomp.config import DATA_DIR, PROCESSED_IMAGES_DIR, WORK_DIR
from medcomp.data_utils import resize_long_side

# Page configuration
st.set_page_config(
    page_title="Technique Comparison | MedComp",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling matching existing MedComp theme
st.markdown(
    """
    <style>
    .banner-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 12px 18px;
        border-radius: 6px;
        border-left: 6px solid #ffeeba;
        font-weight: 500;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.55rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #495057;
        margin-top: 4px;
    }
    .proposed-row {
        background-color: #e8f4fd !important;
        font-weight: bold;
    }
    .footer-text {
        font-size: 0.85rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        padding-top: 15px;
        margin-top: 35px;
    }
    .custom-table-container {
        overflow-x: auto;
        margin-top: 10px;
        margin-bottom: 20px;
        border-radius: 8px;
        border: 1px solid #dee2e6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main() -> None:
    # -------------------------------------------------------------------------
    # Top Banner Disclaimer
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="banner-warning">
            ⚠️ <strong>RESEARCH PROTOTYPE ONLY &mdash; NOT A CLINICAL TOOL</strong><br>
            This comparative analysis workbench evaluates mathematical rate-distortion performance across standard
            and region-aware image compression techniques. High PSNR or SSIM alone does not prove clinical equivalence;
            all diagnostic decisions require certified PACS and radiologist review.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("Comparative Analysis of Medical Image Compression Techniques")
    st.markdown(
        """
        Evaluate and benchmark standard compression baselines (**DCT**, **DWT**, **JPEG**, **JPEG2000**, **Hybrid DCT–DWT**)
        against our **Proposed Region-Aware Near-Lossless Technique** on identical medical radiographs.
        All nine evaluation metrics are measured rigorously from in-memory bitstream representations.
        """
    )

    # -------------------------------------------------------------------------
    # Sidebar: Image Input & Parameters
    # -------------------------------------------------------------------------
    st.sidebar.header("📁 Image Input")
    input_source = st.sidebar.radio(
        "Select Image Source:",
        ["Pick Dataset Sample", "Upload Image (PNG/JPG/DICOM)"],
        index=0,
        key="comp_input_source",
    )

    img_uint8: Optional[np.ndarray] = None
    selected_filename = ""
    is_dataset = False

    if input_source == "Pick Dataset Sample":
        if PROCESSED_IMAGES_DIR.is_dir():
            sample_files = sorted([f.name for f in PROCESSED_IMAGES_DIR.glob("*.png")])
        else:
            sample_files = []

        if sample_files:
            default_idx = 0
            for idx, name in enumerate(sample_files):
                if "MCUCXR_0001_0" in name:
                    default_idx = idx
                    break
            selected_filename = st.sidebar.selectbox("Choose Sample:", sample_files, index=default_idx, key="comp_sample_select")
            img_path = PROCESSED_IMAGES_DIR / selected_filename
            img_raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_raw is not None:
                img_uint8 = resize_long_side(img_raw, target=512)
            is_dataset = True
        else:
            st.sidebar.error("No dataset samples found in work/processed/images.")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Chest X-ray:",
            type=["png", "jpg", "jpeg", "dcm", "dicom"],
            key="comp_file_uploader",
        )
        if uploaded_file is not None:
            selected_filename = uploaded_file.name
            file_bytes = uploaded_file.read()
            try:
                img_uint8 = load_and_preprocess_uploaded_file(file_bytes, selected_filename)
                is_dataset = is_dataset_sample(selected_filename)
            except Exception as e:
                st.sidebar.error(f"Error loading image: {e}")

    # Metadata lookup
    dataset_manifest_row = None
    manifest_csv = DATA_DIR / "manifest.csv"
    if manifest_csv.is_file():
        try:
            m_df = pd.read_csv(manifest_csv)
            stem_query = Path(selected_filename).stem
            matched = m_df[m_df["stem"] == stem_query]
            if not matched.empty:
                dataset_manifest_row = matched.iloc[0]
                is_dataset = True
        except Exception:
            pass

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Compression Settings")
    q_baseline = st.sidebar.slider("Baseline Quality Factor (DCT, DWT, JPEG, Hybrid)", 1, 100, 50, key="comp_q_baseline")
    r_jp2k = st.sidebar.slider("JPEG 2000 Target Ratio", 2.0, 80.0, 10.0, step=1.0, key="comp_r_jp2k")

    st.sidebar.markdown("##### Proposed Technique (Graded Near-Lossless)")
    delta_core = st.sidebar.slider("Delta Core (Max Abs Error on Lungs)", 0, 10, 2, help="0 = Bit-exact lossless lung core", key="comp_d_core")
    delta_band = st.sidebar.slider("Delta Band (Max Abs Error on Border)", 0, 15, 4, key="comp_d_band")
    band_px = st.sidebar.slider("Band Dilation Radius (pixels)", 1, 20, 8, key="comp_band_px")
    bg_q = st.sidebar.slider("Background Quality Factor", 10, 95, 50, key="comp_bg_q")

    run_clicked = st.sidebar.button("🚀 Run Evaluation", type="primary", use_container_width=True)

    if img_uint8 is None:
        st.info("👈 Please select or upload a chest X-ray image in the sidebar to begin.")
        _render_metric_explanations()
        _render_footer()
        return

    orig_h, orig_w = img_uint8.shape
    raw_size_bytes = orig_h * orig_w
    raw_size_kb = raw_size_bytes / 1024.0

    # Display Image Overview Bar
    cohort_str = str(dataset_manifest_row["source"]).capitalize() if dataset_manifest_row is not None else "Uploaded"
    diag_str = "TB Positive" if (dataset_manifest_row is not None and dataset_manifest_row["label"] == 1) else ("Normal" if dataset_manifest_row is not None else "Unlabeled")

    st.markdown(
        f"""
        <div style="background-color: #f1f3f5; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px 16px; margin-bottom: 20px;">
            <strong>Selected Image:</strong> <code>{selected_filename}</code> &nbsp;|&nbsp;
            <strong>Resolution:</strong> {orig_w} &times; {orig_h} px &nbsp;|&nbsp;
            <strong>Raw Size:</strong> {raw_size_kb:.1f} KB ({raw_size_bytes:,} bytes) &nbsp;|&nbsp;
            <strong>Cohort:</strong> {cohort_str} &nbsp;|&nbsp;
            <strong>Ground Truth:</strong> {diag_str}
        </div>
        """,
        unsafe_allow_html=True,
    )

    comp_params = {
        "quality": q_baseline,
        "ratio": r_jp2k,
        "delta_core": delta_core,
        "delta_band": delta_band,
        "band_px": band_px,
        "bg_quality": bg_q,
    }

    cache_key = f"{selected_filename}_{q_baseline}_{r_jp2k}_{delta_core}_{delta_band}_{band_px}_{bg_q}"

    # Auto-execute evaluation if results are not yet present or if image/settings changed or if Run Evaluation was clicked
    should_run = (
        run_clicked
        or (st.session_state.get("comparison_results") is None)
        or (st.session_state.get("comparison_cache_key") != cache_key)
    )

    if should_run:
        with st.spinner(f"Evaluating all 6 compression techniques on {selected_filename}..."):
            # 1. Obtain lung core mask for proposed technique
            mask_bool = None
            pred_mask_path = WORK_DIR / "pred" / "mask" / f"{Path(selected_filename).stem}.png"
            if pred_mask_path.is_file():
                mask_raw = cv2.imread(str(pred_mask_path), cv2.IMREAD_GRAYSCALE)
                if mask_raw is not None:
                    mask_bool = (cv2.resize(mask_raw, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 127)

            if mask_bool is None:
                # Fast U-Net segmentation inference
                try:
                    models = load_models()
                    mask_bool = segment(img_uint8, models=models, fast=True)[0]
                except Exception:
                    # Central prior fallback
                    mask_bool = np.zeros_like(img_uint8, dtype=bool)
                    mask_bool[int(0.15 * orig_h) : int(0.85 * orig_h), int(0.12 * orig_w) : int(0.88 * orig_w)] = True

            results = run_all_techniques(
                img_uint8=img_uint8,
                mask_bool=mask_bool,
                params=comp_params,
            )

            st.session_state["comparison_results"] = results
            st.session_state["comparison_cache_key"] = cache_key
            st.session_state["comparison_image_name"] = selected_filename
            st.toast(f"Evaluated 6 techniques on {selected_filename}!", icon="✅")

    results_list: Optional[list[TechniqueResult]] = st.session_state.get("comparison_results")

    # -------------------------------------------------------------------------
    # Metric Summary Cards (Dynamic)
    # -------------------------------------------------------------------------
    c_psnr, c_ssim, c_mse, c_cr, c_time = st.columns(5)

    if results_list is not None and any(r.status == "SUCCESS" for r in results_list):
        valid = [r for r in results_list if r.status == "SUCCESS"]
        best_psnr = max(valid, key=lambda r: (r.psnr if r.psnr is not None and not np.isinf(r.psnr) else -1.0))
        best_ssim = max(valid, key=lambda r: (r.ssim if r.ssim is not None else -1.0))
        lowest_mse = min(valid, key=lambda r: (r.mse if r.mse is not None else 1e9))
        highest_cr = max(valid, key=lambda r: (r.compression_ratio if r.compression_ratio is not None else -1.0))
        fastest_time = min(valid, key=lambda r: (r.reconstruction_time if r.reconstruction_time is not None else 1e9))

        with c_psnr:
            psnr_display = "∞ (Identical)" if np.isinf(best_psnr.psnr) else f"{best_psnr.psnr:.2f} dB"
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">Best PSNR</div>
                    <div class="metric-val">{psnr_display}</div>
                    <div class="metric-sub">🏆 {best_psnr.technique_name}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c_ssim:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">Best SSIM</div>
                    <div class="metric-val">{best_ssim.ssim:.4f}</div>
                    <div class="metric-sub">🏆 {best_ssim.technique_name}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c_mse:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">Lowest MSE</div>
                    <div class="metric-val">{lowest_mse.mse:.2f}</div>
                    <div class="metric-sub">🏆 {lowest_mse.technique_name}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c_cr:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">Highest Ratio</div>
                    <div class="metric-val">{highest_cr.compression_ratio:.2f}:1</div>
                    <div class="metric-sub">🏆 {highest_cr.technique_name}</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with c_time:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-title">Fastest Decode</div>
                    <div class="metric-val">{fastest_time.reconstruction_time:.4f}s</div>
                    <div class="metric-sub">🏆 {fastest_time.technique_name}</div>
                </div>""",
                unsafe_allow_html=True,
            )
    else:
        for col, title in [
            (c_psnr, "Best PSNR"),
            (c_ssim, "Best SSIM"),
            (c_mse, "Lowest MSE"),
            (c_cr, "Highest Ratio"),
            (c_time, "Fastest Decode"),
        ]:
            with col:
                st.markdown(
                    f"""<div class="metric-card">
                        <div class="metric-title">{title}</div>
                        <div class="metric-val">—</div>
                        <div class="metric-sub">Pending Execution</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------------------------------
    # Controls Toolbar: Sorting, Filtering, and Export
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Technique Evaluation Results")

    col_sort, col_filter, col_export = st.columns([1.5, 2.0, 1.2])

    with col_sort:
        sort_option = st.selectbox(
            "Sort Table By:",
            [
                "Default (S. No.)",
                "Technique Name (A-Z)",
                "PSNR (Highest First)",
                "SSIM (Highest First)",
                "MSE (Lowest First)",
                "Compression Ratio (Highest First)",
                "Reconstruction Time (Fastest First)",
                "R² Score (Highest First)",
            ],
            key="comp_sort_by",
        )

    with col_filter:
        available_names = [TECHNIQUE_NAMES[k] for k in TECHNIQUE_ORDER]
        selected_techniques = st.multiselect(
            "Filter Techniques:",
            options=available_names,
            default=available_names,
            key="comp_filter_tech",
        )

    # Build Display Dataframe
    if results_list is None:
        df_display = get_initial_comparison_df()
        df_export = df_display.copy()
    else:
        # Filter
        filtered_results = [r for r in results_list if r.technique_name in selected_techniques]

        # Numeric Sorting
        if sort_option == "Technique Name (A-Z)":
            filtered_results.sort(key=lambda r: r.technique_name)
        elif sort_option == "PSNR (Highest First)":
            filtered_results.sort(key=lambda r: (float("inf") if (r.psnr is not None and np.isinf(r.psnr)) else (r.psnr or -1.0)), reverse=True)
        elif sort_option == "SSIM (Highest First)":
            filtered_results.sort(key=lambda r: (r.ssim if r.ssim is not None else -1.0), reverse=True)
        elif sort_option == "MSE (Lowest First)":
            filtered_results.sort(key=lambda r: (r.mse if r.mse is not None else 1e9))
        elif sort_option == "Compression Ratio (Highest First)":
            filtered_results.sort(key=lambda r: (r.compression_ratio if r.compression_ratio is not None else -1.0), reverse=True)
        elif sort_option == "Reconstruction Time (Fastest First)":
            filtered_results.sort(key=lambda r: (r.reconstruction_time if r.reconstruction_time is not None else 1e9))
        elif sort_option == "R² Score (Highest First)":
            filtered_results.sort(key=lambda r: (r.r2_score if r.r2_score is not None else -100.0), reverse=True)
        else:
            filtered_results.sort(key=lambda r: r.s_no)

        display_rows = [r.to_display_dict() for r in filtered_results]
        df_display = pd.DataFrame(display_rows)
        # Drop internal status column from main 11-column table view
        if "Status" in df_display.columns:
            df_display = df_display.drop(columns=["Status"])

        # Export rows
        export_rows = []
        for r in filtered_results:
            d = r.to_numeric_dict()
            d["image_file"] = selected_filename
            d["timestamp"] = datetime.now().isoformat()
            export_rows.append(d)
        df_export = pd.DataFrame(export_rows)

    with col_export:
        st.write("")  # spacing
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Export Results (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"technique_comparison_{Path(selected_filename).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download the comparative analysis results table as a standard CSV file.",
        )

    # -------------------------------------------------------------------------
    # Render Comparison Table with Highlighting
    # -------------------------------------------------------------------------
    def highlight_proposed(row: pd.Series) -> list[str]:
        is_prop = "Proposed" in str(row.get("Compression Technique", ""))
        return ["background-color: #e8f4fd; font-weight: bold;" if is_prop else "" for _ in row]

    st.markdown('<div class="custom-table-container">', unsafe_allow_html=True)
    styled_table = df_display.style.apply(highlight_proposed, axis=1)
    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.caption(
        "💡 **Evaluation Note:** Accuracy, Precision, and Recall display **N/A** because single-image compression is a continuous pixel reconstruction "
        "(fidelity regression) task without binary diagnostic classification boundaries. Paired clinical diagnostic accuracy and classification flip rates "
        "across patient cohorts are rigorously reported on the **Model Performance** and **Mini-Benchmark** pages. "
        "The **Proposed Technique** row is highlighted in light blue."
    )

    # -------------------------------------------------------------------------
    # Visual Image Reconstructions Gallery
    # -------------------------------------------------------------------------
    if results_list is not None and any(r.recon_img is not None for r in results_list):
        st.markdown("---")
        st.markdown("### 🖼️ Reconstructed Image Gallery (Side-by-Side Quality Inspection)")

        tabs = st.tabs(["Original Reference"] + [r.technique_name for r in results_list if r.recon_img is not None])

        with tabs[0]:
            c_orig_img, c_orig_info = st.columns([2, 1])
            with c_orig_img:
                st.image(img_uint8, caption=f"Original Uncompressed: {selected_filename} ({orig_w}x{orig_h})", use_container_width=True)
            with c_orig_info:
                st.markdown("#### Original Specifications")
                st.write(f"- **Format:** 8-bit Grayscale")
                st.write(f"- **Dimensions:** {orig_w} &times; {orig_h} pixels")
                st.write(f"- **Raw Array Size:** {raw_size_bytes:,} bytes ({raw_size_kb:.1f} KB)")
                st.write(f"- **Dynamic Range:** [0, 255]")

        for idx, r in enumerate([r for r in results_list if r.recon_img is not None]):
            with tabs[idx + 1]:
                c_img, c_diff = st.columns([1, 1])
                with c_img:
                    st.image(
                        r.recon_img,
                        caption=f"{r.technique_full_name} | PSNR: {r.psnr:.2f} dB, SSIM: {r.ssim:.4f}",
                        use_container_width=True,
                    )
                with c_diff:
                    # Difference error map (amplified 5x for visual clarity)
                    diff = np.abs(img_uint8.astype(float) - r.recon_img.astype(float))
                    diff_vis = np.clip(diff * 5.0, 0, 255).astype(np.uint8)
                    st.image(
                        diff_vis,
                        caption=f"Absolute Error Map (Amplified 5&times;) | MSE: {r.mse:.2f}, Max Error: {float(np.max(diff)):.1f}",
                        use_container_width=True,
                    )

    # -------------------------------------------------------------------------
    # Metric Explanations Section
    # -------------------------------------------------------------------------
    _render_metric_explanations()

    # Footer
    _render_footer()


def _render_metric_explanations() -> None:
    """Render the expandable 'Understanding the Evaluation Metrics' technical documentation."""
    st.markdown("---")
    with st.expander("📖 Understanding the Evaluation Metrics", expanded=False):
        st.markdown(
            """
            This section defines the nine standard evaluation metrics used to assess medical image compression performance:

            #### 1. Accuracy (%)
            - **Definition:** The proportion of correct predictions across all evaluation samples:
              $$\\text{Accuracy} = \\frac{\\text{TP} + \\text{TN}}{\\text{TP} + \\text{TN} + \\text{FP} + \\text{FN}}$$
            - **Application in Medical Compression:** Displayed as **N/A** for single-image reconstruction because pixel compression is a continuous intensity restoration task rather than discrete classification. Diagnostic accuracy is evaluated across patient cohorts on the **Model Performance** page.

            #### 2. Precision (%)
            - **Definition:** The fraction of predicted positive cases that are genuine ground-truth positives:
              $$\\text{Precision} = \\frac{\\text{TP}}{\\text{TP} + \\text{FP}}$$
            - **Application:** Displayed as **N/A** unless an explicit binary anomaly detector or corruption classifier is evaluated.

            #### 3. Recall (%)
            - **Definition:** The sensitivity or true positive rate, representing the fraction of actual positives identified:
              $$\\text{Recall} = \\frac{\\text{TP}}{\\text{TP} + \\text{FN}}$$
            - **Application:** Displayed as **N/A** for pixel reconstruction; evaluated clinically in downstream disease screening.

            #### 4. R² Score (Coefficient of Determination)
            - **Definition:** Measures the proportion of variance in the original image pixel intensities explained by the reconstructed image:
              $$R^2 = 1 - \\frac{\\sum (y_i - \\hat{y}_i)^2}{\\sum (y_i - \\bar{y})^2} = 1 - \\frac{\\text{SS}_{\\text{res}}}{\\text{SS}_{\\text{tot}}}$$
            - **Interpretation:** $R^2 = 1.0$ indicates perfect bit-exact reconstruction. Scores close to 1.0 confirm that anatomical structures and texture variations are faithfully preserved.

            #### 5. PSNR (Peak Signal-to-Noise Ratio in dB)
            - **Definition:** Logarithmic measure of peak reconstruction fidelity relative to mean squared distortion:
              $$\\text{PSNR} = 10 \\cdot \\log_{10}\\left(\\frac{\\text{MAX}_I^2}{\\text{MSE}}\\right)$$
              where $\\text{MAX}_I = 255.0$ for 8-bit medical radiographs.
            - **Interpretation:** Higher values represent higher mathematical fidelity. $\\text{PSNR} = \\infty$ indicates identical images ($\text{MSE} = 0$). Typical diagnostic-quality radiographs maintain $\\text{PSNR} > 40\\text{ dB}$.

            #### 6. SSIM (Structural Similarity Index)
            - **Definition:** Multi-component perceptual metric assessing luminance, contrast, and structural correlation:
              $$\\text{SSIM}(x, y) = \\frac{(2\\mu_x\\mu_y + C_1)(2\\sigma_{xy} + C_2)}{(\\mu_x^2 + \\mu_y^2 + C_1)(\\sigma_x^2 + \\sigma_y^2 + C_2)}$$
            - **Interpretation:** Ranges in $[-1, 1]$, where $1.0$ denotes perfect structural congruence. SSIM correlates more strongly with human radiologist perception than MSE.

            #### 7. MSE (Mean Squared Error)
            - **Definition:** Average squared difference between corresponding original and reconstructed pixels:
              $$\\text{MSE} = \\frac{1}{H \\cdot W} \\sum_{i=1}^H \\sum_{j=1}^W \\left(I(i, j) - \\hat{I}(i, j)\\right)^2$$
            - **Interpretation:** Lower MSE indicates smaller reconstruction error. $\\text{MSE} = 0.0$ for lossless reconstruction.

            #### 8. Compression Ratio (CR)
            - **Definition:** Ratio of uncompressed raw image size to compressed bitstream size:
              $$\\text{Compression Ratio} = \\frac{\\text{Original Uncompressed Size (bytes)}}{\\text{Compressed File Size (bytes)}}$$
            - **Interpretation:** Expressed as a factor (e.g. $12.50:1$ or $12.50\\times$). Higher compression ratio reduces PACS storage and transmission bandwidth, but must be balanced against anatomical preservation.

            #### 9. Reconstruction Time (s)
            - **Definition:** Exact elapsed time required to decompress the serialized bitstream and reconstruct the spatial image array:
              $$\\text{Reconstruction Time} = t_{\\text{decode\\_end}} - t_{\\text{decode\\_start}}$$
            - **Interpretation:** Lower reconstruction time enables low-latency clinical review and responsive interactive display.

            ---

            #### Clinical Relevance & Compression Trade-offs
            **PSNR, SSIM, MSE, Compression Ratio, and Reconstruction Time** are the primary benchmarks for evaluating medical image compression.
            Higher PSNR and SSIM combined with lower MSE denote superior fidelity. However, in medical imaging, high whole-image PSNR can conceal localized micro-calcifications or subtle lesion blur.
            Our **Proposed Technique** addresses this fundamental trade-off by dedicating deterministic near-lossless guarantees ($L_\\infty \\le \\delta$)
            to the segmented lung fields while aggressively compressing non-diagnostic background areas.
            """
        )


def _render_footer() -> None:
    """Render footer disclaimer and hardware metadata."""
    st.markdown(
        """
        <div class="footer-text">
            <strong>MedComp: Medical Image Compression & Clinical Audit Benchmark</strong><br>
            Hardware: NVIDIA RTX 2050 GPU (CUDA accelerated) &bull; PyTorch 2.x &bull; Streamlit<br>
            <em>Disclaimer: Strictly for investigational engineering and academic research. Not FDA/CE cleared for primary clinical diagnosis.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
