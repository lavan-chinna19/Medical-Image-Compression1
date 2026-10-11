"""Streamlit Demonstration Application for Medical Image Compression (medcomp).

DISCLAIMER:
This application is an investigational research prototype, NOT a clinical tool.
It is strictly not intended for primary clinical diagnosis or medical decision making.

Hardware: RTX 2050 GPU with CUDA acceleration.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medcomp.app_logic import (
    LoadedModels,
    classify,
    compress,
    diagnose,
    get_model_benchmark_metrics,
    is_dataset_sample,
    load_and_preprocess_uploaded_file,
    load_models,
    segment,
)
from medcomp.config import PROCESSED_IMAGES_DIR

# Page configuration
st.set_page_config(
    page_title="MedComp: Medical Image Compression Prototype",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS styling
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
        border: 1px solid #e9ecef;
        padding: 14px;
        border-radius: 8px;
        text-align: center;
    }
    .footer-text {
        font-size: 0.85rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        padding-top: 15px;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_models() -> LoadedModels:
    """Load and cache all 15 neural network models on GPU."""
    return load_models()


def main() -> None:
    # -------------------------------------------------------------------------
    # Top Banner Disclaimer
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="banner-warning">
            ⚠️ <strong>RESEARCH PROTOTYPE ONLY &mdash; NOT A CLINICAL TOOL</strong><br>
            This system is an investigational research prototype developed for medical image compression analysis.
            It is strictly not certified for primary clinical diagnosis, screening, or medical decision making.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("🫁 MedComp: Rate-Constrained Chest X-Ray Compression")
    st.caption("GPU-Accelerated Rate-Distortion & Diagnostic Fidelity Analysis on RTX 2050")

    # Load models
    with st.spinner("Initializing neural networks on RTX 2050 GPU..."):
        try:
            models = get_cached_models()
        except Exception as e:
            st.error(f"Failed to initialize GPU models: {e}")
            st.stop()

    # -------------------------------------------------------------------------
    # Sidebar: Input Selection and Compression Parameters
    # -------------------------------------------------------------------------
    st.sidebar.header("📁 Image Input")

    input_source = st.sidebar.radio(
        "Select Image Source:",
        ["Pick Dataset Sample", "Upload Image (PNG/JPG/DICOM)"],
        index=0,
    )

    img_uint8: np.ndarray | None = None
    selected_filename = ""
    is_dataset = False

    if input_source == "Pick Dataset Sample":
        if PROCESSED_IMAGES_DIR.is_dir():
            sample_files = sorted([f.name for f in PROCESSED_IMAGES_DIR.glob("*.png")])
        else:
            sample_files = []

        if sample_files:
            # Default to a representative sample like CHNCXR_0001_0 or MCUCXR_0001_0
            default_idx = 0
            for idx, name in enumerate(sample_files):
                if "MCUCXR_0001_0" in name:
                    default_idx = idx
                    break
            selected_filename = st.sidebar.selectbox("Choose Sample:", sample_files, index=default_idx)
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
        )
        if uploaded_file is not None:
            selected_filename = uploaded_file.name
            file_bytes = uploaded_file.read()
            try:
                img_uint8 = load_and_preprocess_uploaded_file(file_bytes, selected_filename)
                is_dataset = is_dataset_sample(selected_filename)
            except Exception as e:
                st.sidebar.error(f"Error loading image: {e}")

    # Dataset warning
    if is_dataset:
        st.sidebar.warning(
            "⚠️ This image was used to train the models; results are for demonstration, not evaluation."
        )

    if img_uint8 is None:
        st.info("👈 Please select or upload a chest X-ray image in the sidebar to begin.")
        _render_footer()
        return

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Compression Settings")

    mode_options = {
        "graded": "Graded Near-Lossless (ROI + Band + DWT)",
        "lossless_dilated": "Lossless Dilated (Lossless Lung + DWT)",
        "dwt": "Tuned Custom DWT (bior2.2, 4 Levels)",
        "dct": "Custom Block DCT (8x8 JPEG Matrix)",
        "jpeg": "Standard JPEG (DCT)",
        "jpeg2000": "Standard JPEG 2000 (OpenJPEG DWT)",
        "png": "Lossless PNG (DEFLATE)",
    }

    mode_keys = list(mode_options.keys())
    selected_mode = st.sidebar.selectbox(
        "Compression Mode:",
        mode_keys,
        format_func=lambda k: mode_options[k],
        index=0,
    )

    params: dict[str, Any] = {}

    if selected_mode == "graded":
        st.sidebar.caption("Configures 3-zone error guarantees: Core (lung), Band (dilated border), Background.")
        params["delta_core"] = st.sidebar.slider("Delta Core (Max Lung Abs Error)", 0, 10, 2, help="0 = Bit-exact lossless")
        params["delta_band"] = st.sidebar.slider("Delta Band (Max Border Abs Error)", 0, 15, 4)
        params["band_px"] = st.sidebar.slider("Band Dilation Radius (pixels)", 1, 20, 8)
        params["bg_quality"] = st.sidebar.slider("Background DWT Quality", 10, 95, 50)

    elif selected_mode == "lossless_dilated":
        st.sidebar.caption("Guarantees 100% bit-exact reconstruction over dilated lung field.")
        params["band_px"] = st.sidebar.slider("Lung Dilation Radius (pixels)", 1, 20, 8)
        params["bg_quality"] = st.sidebar.slider("Background DWT Quality", 10, 95, 50)

    elif selected_mode in ("dwt", "dct", "jpeg"):
        params["quality"] = st.sidebar.slider(f"{selected_mode.upper()} Quality Factor", 1, 100, 50)

    elif selected_mode == "jpeg2000":
        params["ratio"] = st.sidebar.slider("JPEG 2000 Compression Ratio", 2.0, 80.0, 10.0, step=1.0)

    fast_seg = st.sidebar.checkbox(
        "Fast segmentation preview (1 model, 1 pass)",
        value=False,
        help="Skips 8-pass Monte Carlo dropout ensemble for rapid interactive preview.",
    )

    # -------------------------------------------------------------------------
    # Pipeline Execution: Segmentation, Compression, Diagnosis
    # -------------------------------------------------------------------------
    try:
        mask_bool, prob_map, unc_map = segment(img_uint8, models=models, fast=fast_seg)
    except torch.cuda.OutOfMemoryError:
        st.warning("⚠️ GPU memory limit reached during ensemble. Automatically retrying in fast mode...")
        torch.cuda.empty_cache()
        mask_bool, prob_map, unc_map = segment(img_uint8, models=models, fast=True)
    except Exception as e:
        st.error(f"Segmentation failed: {e}")
        return

    try:
        comp_res = compress(img_uint8, mask=mask_bool, mode=selected_mode, params=params)
    except Exception as e:
        st.error(f"Compression failed: {e}")
        return

    try:
        diag_res = diagnose(img_uint8, comp_res["recon"], models=models)
    except Exception as e:
        st.error(f"Diagnosis inference failed: {e}")
        return

    # -------------------------------------------------------------------------
    # Main Area: 4 Analysis Tabs
    # -------------------------------------------------------------------------
    tab_img, tab_size, tab_lungs, tab_diag = st.tabs([
        "🖼️ Image Inspection",
        "📊 Compression & Size",
        "🫁 Lung Diagnostics",
        "🩺 Clinical Flip Diagnosis",
    ])

    recon_img = comp_res["recon"]

    # --- TAB 1: IMAGE ---
    with tab_img:
        st.subheader("Side-by-Side Visual Inspection")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img_uint8, caption=f"Original ({img_uint8.shape[1]}x{img_uint8.shape[0]})", use_container_width=True)

        with col2:
            st.image(recon_img, caption=f"Decoded Reconstruction ({selected_mode})", use_container_width=True)

        with col3:
            diff = np.abs(img_uint8.astype(float) - recon_img.astype(float))
            diff_amp = np.clip(diff * 10.0, 0, 255).astype(np.uint8)
            st.image(diff_amp, caption="Absolute Difference (Amplified ×10)", use_container_width=True)

        st.caption("The difference image visualizes compression artifacts amplified by 10x. Dark regions indicate zero or minimal error.")

    # --- TAB 2: SIZE & BITRATE ---
    with tab_size:
        st.subheader("Compression Metrics & Bitstream Analysis")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Compressed Size", f"{comp_res['num_bytes']:,} bytes", f"{comp_res['num_bytes']/1024:.1f} KB")
        m2.metric("Bitrate", f"{comp_res['bpp']:.3f} bpp")
        m3.metric("Ratio vs Raw 8-bit", f"{comp_res['ratio_vs_raw']:.2f}×")
        m4.metric("Ratio vs Lossless PNG", f"{comp_res['ratio_vs_png']:.2f}×")

        # Stacked bar chart for zone-based modes
        parts = comp_res["parts"]
        if selected_mode in ("graded", "lossless_dilated") and "header" in parts:
            st.markdown("#### Zone Byte Consumption Breakdown")
            fig, ax = plt.subplots(figsize=(9, 2.2))
            labels = ["Header", "Mask", "Core (Lung)", "Band (Border)", "Background"]
            keys = ["header", "mask", "core", "band", "background"]
            colors = ["#6c757d", "#17a2b8", "#28a745", "#ffc107", "#007bff"]
            values = [parts.get(k, 0) for k in keys]
            total_b = max(1, sum(values))

            left = 0
            for label, val, col in zip(labels, values, colors):
                pct = (val / total_b) * 100.0
                ax.barh(0, val, left=left, color=col, edgecolor="white", label=f"{label}: {val:,} B ({pct:.1f}%)")
                left += val

            ax.set_yticks([])
            ax.set_xlabel("Bytes")
            ax.set_xlim(0, total_b)
            ax.legend(bbox_to_anchor=(0.5, 1.35), loc="upper center", ncol=5, frameon=False, fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        # Comparison table across all modes
        st.markdown("#### Mode Comparison at Current Quality Settings")
        with st.spinner("Benchmarking all modes on current image..."):
            all_modes_data = []
            for test_mode in ["png", "jpeg", "jpeg2000", "dct", "dwt", "lossless_dilated", "graded"]:
                try:
                    c_out = compress(img_uint8, mask=mask_bool, mode=test_mode, params=params)
                    all_modes_data.append({
                        "Mode": mode_options.get(test_mode, test_mode),
                        "Bytes": f"{c_out['num_bytes']:,}",
                        "bpp": f"{c_out['bpp']:.3f}",
                        "Ratio vs Raw": f"{c_out['ratio_vs_raw']:.2f}×",
                        "Ratio vs PNG": f"{c_out['ratio_vs_png']:.2f}×",
                        "Max Lung Error": f"{c_out.get('max_err_core', 0.0):.0f} px",
                        "Exact Lung %": f"{c_out.get('exact_fraction', 0.0)*100.0:.1f}%",
                    })
                except Exception:
                    pass

            if all_modes_data:
                st.dataframe(pd.DataFrame(all_modes_data), use_container_width=True, hide_index=True)

    # --- TAB 3: LUNGS ---
    with tab_lungs:
        st.subheader("Anatomical Lung ROI & Uncertainty Analysis")
        lcol1, lcol2 = st.columns(2)

        with lcol1:
            # Mask outline overlay
            overlay = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
            contours, _ = cv2.findContours(mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
            st.image(overlay, caption="Predicted Lung Boundary Overlay (U-Net)", use_container_width=True)

        with lcol2:
            # Colormap of uncertainty
            fig_u, ax_u = plt.subplots(figsize=(5, 5))
            im_u = ax_u.imshow(unc_map, cmap="magma")
            ax_u.axis("off")
            fig_u.colorbar(im_u, ax=ax_u, fraction=0.046, pad=0.04, label="MC Dropout Std Dev")
            ax_u.set_title("Segmentation Epistemic Uncertainty", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig_u)
            plt.close()

        st.markdown("#### Intra-Lung Reconstruction Fidelity")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Max Core Error", f"{comp_res.get('max_err_core', 0.0):.1f} / 255")
        s2.metric("99th %-tile Core Error", f"{comp_res.get('p99_err_core', 0.0):.1f} / 255")
        s3.metric("Max Band Error", f"{comp_res.get('max_err_band', 0.0):.1f} / 255")
        s4.metric("Exact Lung Pixels", f"{comp_res.get('exact_fraction', 0.0)*100.0:.2f}%")

    # --- TAB 4: DIAGNOSIS ---
    with tab_diag:
        st.subheader("Clinical TB Prediction & Flip Detection")
        st.caption("Measures whether lossy compression triggers classification flips across the 0.5 decision threshold.")

        dcol1, dcol2 = st.columns(2)

        with dcol1:
            st.markdown("##### ResNet18 (Steering Classifier)")
            p_ro = diag_res["resnet_prob_orig"]
            p_rd = diag_res["resnet_prob_dec"]
            diff_r = p_rd - p_ro
            st.write(f"**Original Image:** `{p_ro:.4f}`")
            st.write(f"**Reconstructed Image:** `{p_rd:.4f}` (Drift: `{diff_r:+.4f}`)")
            if diag_res["flip_resnet"]:
                st.error("⚠️ **CLASSIFIER DECISION FLIPPED!** Binary prediction changed across 0.5.")
            else:
                st.success("✅ **Binary Decision Preserved** (Consistent with original).")

        with dcol2:
            st.markdown("##### EfficientNet-B0 (Independent Judge)")
            p_jo = diag_res["judge_prob_orig"]
            p_jd = diag_res["judge_prob_dec"]
            diff_j = p_jd - p_jo
            st.write(f"**Original Image:** `{p_jo:.4f}`")
            st.write(f"**Reconstructed Image:** `{p_jd:.4f}` (Drift: `{diff_j:+.4f}`)")
            if diag_res["flip_judge"]:
                st.error("⚠️ **JUDGE DECISION FLIPPED!** Binary prediction changed across 0.5.")
            else:
                st.success("✅ **Judge Decision Preserved** (Consistent with original).")

        st.markdown(
            """
            > **Plain-Language Clarification:** The probabilities shown above represent automated diagnostic model
            > outputs evaluated for research sensitivity analysis. They do **not** constitute a validated medical diagnosis
            > or clinical statement of disease presence.
            """
        )

    # -------------------------------------------------------------------------
    # Download Section
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Export Bitstream & Artifacts")
    bcol1, bcol2, bcol3 = st.columns(3)

    # 1. Compressed Bitstream (.bin)
    with bcol1:
        st.download_button(
            label="💾 Download Bitstream (.bin)",
            data=comp_res["bytes"],
            file_name=f"{Path(selected_filename).stem}_{selected_mode}.bin",
            mime="application/octet-stream",
            use_container_width=True,
        )

    # 2. Reconstructed Image (.png)
    with bcol2:
        recon_buf = io.BytesIO()
        Image.fromarray(recon_img).save(recon_buf, format="PNG")
        st.download_button(
            label="🖼️ Download Reconstruction (.png)",
            data=recon_buf.getvalue(),
            file_name=f"{Path(selected_filename).stem}_{selected_mode}_recon.png",
            mime="image/png",
            use_container_width=True,
        )

    # 3. JSON Summary Report
    with bcol3:
        report_data = {
            "disclaimer": "RESEARCH PROTOTYPE ONLY - NOT A CLINICAL TOOL",
            "filename": selected_filename,
            "is_dataset_training_sample": is_dataset,
            "image_dimensions": {"height": img_uint8.shape[0], "width": img_uint8.shape[1]},
            "compression_mode": selected_mode,
            "parameters": params,
            "metrics": {
                "compressed_bytes": comp_res["num_bytes"],
                "bpp": comp_res["bpp"],
                "ratio_vs_raw": comp_res["ratio_vs_raw"],
                "ratio_vs_png": comp_res["ratio_vs_png"],
                "exact_lung_fraction": comp_res.get("exact_fraction", 0.0),
                "max_err_core": comp_res.get("max_err_core", 0.0),
                "p99_err_core": comp_res.get("p99_err_core", 0.0),
                "max_err_band": comp_res.get("max_err_band", 0.0),
                "parts_bytes": comp_res["parts"],
            },
            "diagnosis": diag_res,
        }
        st.download_button(
            label="📋 Download JSON Report (.json)",
            data=json.dumps(report_data, indent=2),
            file_name=f"{Path(selected_filename).stem}_{selected_mode}_report.json",
            mime="application/json",
            use_container_width=True,
        )

    # -------------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------------
    _render_footer()


def _render_footer() -> None:
    benchmarks = get_model_benchmark_metrics()
    st.markdown(
        f"""
        <div class="footer-text">
            <strong>System Benchmarks:</strong><br>
            • {benchmarks['dice_line']}<br>
            • {benchmarks['resnet_line']}<br>
            • {benchmarks['judge_line']}<br>
            <br>
            <em>Antigravity MedComp Research Prototype &bull; For investigational evaluation only &bull; Not for clinical diagnosis.</em>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
