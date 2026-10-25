"""Streamlit Multipage: Model Performance & Cohort Evaluation.

Displays comprehensive diagnostic classification metrics, ROC curves, confusion matrices,
cross-scanner transfer analysis, compression-vs-diagnosis rate-distortion curves,
and segmentation performance without requiring GPU acceleration.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medcomp.config import RESULTS_DIR
from medcomp.metrics_report import (
    bootstrap_ci,
    classification_metrics,
    load_oof,
    load_results,
    roc_points,
)

# Page configuration
st.set_page_config(
    page_title="Model Performance & Validation | MedComp",
    page_icon="📈",
    layout="wide",
)

# Custom styling
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
        font-size: 0.85rem;
        color: #6c757d;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-ci {
        font-size: 0.8rem;
        color: #495057;
        margin-top: 4px;
    }
    .footer-text {
        font-size: 0.85rem;
        color: #6c757d;
        border-top: 1px solid #dee2e6;
        padding-top: 15px;
        margin-top: 35px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def get_oof_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and cache out-of-fold prediction CSVs."""
    df_res = load_oof(RESULTS_DIR / "classifier_oof.csv")
    df_jd = load_oof(RESULTS_DIR / "judge_oof.csv")
    return df_res, df_jd


@st.cache_data
def get_cached_bootstrap(
    y_true_tuple: tuple[int, ...],
    prob_tuple: tuple[float, ...],
    threshold: float,
) -> dict[str, tuple[float, float]]:
    """Cached wrapper for non-parametric bootstrap confidence intervals."""
    return bootstrap_ci(
        np.array(y_true_tuple),
        np.array(prob_tuple),
        threshold=threshold,
        n=1000,
        seed=42,
    )


@st.cache_data
def get_all_results() -> dict[str, Any]:
    """Load and cache all benchmark result CSV tables."""
    return load_results(RESULTS_DIR)


def render_csv_download(df: pd.DataFrame, filename: str, label: str) -> None:
    """Render a standard CSV download button for a DataFrame."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


def main() -> None:
    # -------------------------------------------------------------------------
    # Section A: Header & Validation Protocol
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="banner-warning">
            ⚠️ <strong>RESEARCH PROTOTYPE ONLY &mdash; NOT A CLINICAL TOOL</strong><br>
            All classification probabilities and validation metrics presented here are for technical research evaluation only.
            They are strictly not intended for primary clinical diagnosis or medical decision making.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("📈 Model Performance & Diagnostic Validation")
    st.markdown(
        """
        **Clinical Task:** Binary Tuberculosis (TB) classification (*TB suspected* vs *Normal*) from 2D chest radiographs (CXRs).  
        **Evaluation Dataset:** **800** total chest X-rays from two public international clinical cohorts:  
        - **Shenzhen Hospital, China:** 662 images (326 TB positive, 336 normal)  
        - **Montgomery County, Maryland, USA:** 138 images (58 TB positive, 80 normal)  
        **Cross-Validation Protocol:** **5-fold patient-level stratified cross-validation**.  
        *Every prediction displayed on this page is strictly out-of-fold (OOF)*: each patient's image was evaluated 
        solely by the model checkpoint trained on the other 4 folds, ensuring completely unbiased evaluation with zero data leakage.
        """
    )
    st.markdown("---")

    # Load OOF data
    try:
        df_resnet, df_judge = get_oof_data()
    except Exception as e:
        st.error(f"Failed to load out-of-fold prediction files: {e}")
        return

    res_tables = get_all_results()

    # -------------------------------------------------------------------------
    # Section B: Interactive Metrics & Live Threshold Recomputation
    # -------------------------------------------------------------------------
    st.subheader("1. Interactive Out-of-Fold Model Diagnostics")
    st.caption("Recomputes all diagnostic metrics, confidence intervals, confusion matrix counts, and ROC curves live from OOF predictions.")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 2.0])

    with ctrl_col1:
        model_choice = st.selectbox(
            "Select Classifier Model:",
            ["ResNet18 (Steering Classifier)", "EfficientNet-B0 (Judge Classifier)"],
            index=0,
        )
    with ctrl_col2:
        source_choice = st.selectbox(
            "Filter Patient Cohort:",
            ["All Sources (N = 800)", "Shenzhen Cohort (N = 662)", "Montgomery Cohort (N = 138)"],
            index=0,
        )
    with ctrl_col3:
        threshold = st.slider(
            "Decision Threshold (τ):",
            min_value=0.05,
            max_value=0.95,
            value=0.50,
            step=0.01,
            help="Threshold for binary positive prediction: positive if P(TB) >= τ.",
        )

    # Filter data
    active_df = df_resnet if "ResNet18" in model_choice else df_judge
    if "Shenzhen" in source_choice:
        sub_df = active_df[active_df["source"] == "Shenzhen"].copy()
    elif "Montgomery" in source_choice:
        sub_df = active_df[active_df["source"] == "Montgomery"].copy()
    else:
        sub_df = active_df.copy()

    y_true = sub_df["label"].values.astype(int)
    prob_tb = sub_df["prob_tb"].values.astype(float)

    # Compute live metrics & 95% bootstrap CI
    m = classification_metrics(y_true, prob_tb, threshold=threshold)
    ci = get_cached_bootstrap(tuple(y_true), tuple(prob_tb), threshold=threshold)

    # Metric cards display
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    cards = [
        (mc1, "Accuracy", f"{m['accuracy']*100:.2f}%", f"95% CI: [{ci['accuracy'][0]*100:.1f}%, {ci['accuracy'][1]*100:.1f}%]"),
        (mc2, "Sensitivity (Recall)", f"{m['recall']*100:.2f}%", f"95% CI: [{ci['recall'][0]*100:.1f}%, {ci['recall'][1]*100:.1f}%]"),
        (mc3, "Specificity", f"{m['specificity']*100:.2f}%", f"95% CI: [{ci['specificity'][0]*100:.1f}%, {ci['specificity'][1]*100:.1f}%]"),
        (mc4, "Precision (PPV)", f"{m['precision']*100:.2f}%", f"95% CI: [{ci['precision'][0]*100:.1f}%, {ci['precision'][1]*100:.1f}%]"),
        (mc5, "F1-Score", f"{m['f1']:.4f}", f"95% CI: [{ci['f1'][0]:.3f}, {ci['f1'][1]:.3f}]"),
        (mc6, "ROC AUC", f"{m['auc']:.4f}", f"95% CI: [{ci['auc'][0]:.3f}, {ci['auc'][1]:.3f}]"),
    ]
    for col, title, val, ci_str in cards:
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{title}</div>
                    <div class="metric-val">{val}</div>
                    <div class="metric-ci">{ci_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Plots: Confusion Matrix, ROC Curve, Probability Histogram
    pcol1, pcol2, pcol3 = st.columns(3)

    # 1. Confusion Matrix Heatmap
    with pcol1:
        st.markdown("##### Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(4.2, 3.6))
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=int)
        cax = ax_cm.imshow(cm, cmap="Blues", interpolation="nearest")
        fig_cm.colorbar(cax, ax=ax_cm, fraction=0.046, pad=0.04)

        ax_cm.set_xticks([0, 1])
        ax_cm.set_yticks([0, 1])
        ax_cm.set_xticklabels(["Pred Normal", "Pred TB"], fontsize=9)
        ax_cm.set_yticklabels(["True Normal", "True TB"], fontsize=9)
        ax_cm.set_title(f"Confusion Matrix (τ = {threshold:.2f})", fontsize=11, fontweight="bold")

        # Cell text
        max_val = max(1, int(np.max(cm)))
        for i in range(2):
            for j in range(2):
                val_c = cm[i, j]
                txt_color = "white" if val_c > max_val * 0.55 else "black"
                ax_cm.text(j, i, f"{val_c:,}", ha="center", va="center", color=txt_color, fontsize=12, fontweight="bold")

        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

    # 2. ROC Curve
    with pcol2:
        st.markdown("##### ROC Curve with Operating Point")
        roc_df = roc_points(y_true, prob_tb)
        fig_roc, ax_roc = plt.subplots(figsize=(4.2, 3.6))

        if not roc_df.empty:
            ax_roc.plot(roc_df["fpr"], roc_df["tpr"], color="#1f77b4", lw=2.2, label=f"ROC (AUC = {m['auc']:.4f})")

            # Current threshold operating point
            curr_fpr = m["fp"] / max(1, (m["tn"] + m["fp"]))
            curr_tpr = m["tp"] / max(1, (m["tp"] + m["fn"]))
            ax_roc.plot(curr_fpr, curr_tpr, marker="o", markersize=8, color="#d62728", label=f"τ = {threshold:.2f}")

        ax_roc.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1.2, label="Chance")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate (1 - Spec)", fontsize=9)
        ax_roc.set_ylabel("True Positive Rate (Sens)", fontsize=9)
        ax_roc.set_title("Receiver Operating Characteristic", fontsize=11, fontweight="bold")
        ax_roc.legend(loc="lower right", fontsize=8)
        ax_roc.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close()

    # 3. Probability Histogram per Class
    with pcol3:
        st.markdown("##### Class Probability Distribution")
        fig_hist, ax_hist = plt.subplots(figsize=(4.2, 3.6))
        probs_normal = prob_tb[y_true == 0]
        probs_tb = prob_tb[y_true == 1]

        bins = np.linspace(0.0, 1.0, 26)
        ax_hist.hist(probs_normal, bins=bins, alpha=0.6, color="#2ca02c", label=f"Normal (N={len(probs_normal)})", density=True)
        ax_hist.hist(probs_tb, bins=bins, alpha=0.6, color="#d62728", label=f"TB (N={len(probs_tb)})", density=True)
        ax_hist.axvline(threshold, color="black", linestyle="--", lw=1.8, label=f"τ = {threshold:.2f}")

        ax_hist.set_xlim([0.0, 1.0])
        ax_hist.set_xlabel("Predicted P(TB)", fontsize=9)
        ax_hist.set_ylabel("Density", fontsize=9)
        ax_hist.set_title("Probability Distribution by Label", fontsize=11, fontweight="bold")
        ax_hist.legend(loc="upper center", fontsize=8)
        ax_hist.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_hist)
        plt.close()

    # Side-by-side comparison table of both models at chosen threshold
    st.markdown("##### Side-by-Side Model Comparison at Current Threshold")
    m_res = classification_metrics(
        df_resnet[df_resnet["source"].isin(sub_df["source"].unique())]["label"].values,
        df_resnet[df_resnet["source"].isin(sub_df["source"].unique())]["prob_tb"].values,
        threshold=threshold,
    )
    m_jd = classification_metrics(
        df_judge[df_judge["source"].isin(sub_df["source"].unique())]["label"].values,
        df_judge[df_judge["source"].isin(sub_df["source"].unique())]["prob_tb"].values,
        threshold=threshold,
    )

    comp_df = pd.DataFrame([
        {
            "Model Architecture": "ResNet18 (Steering)",
            "Cohort": source_choice,
            "Threshold": threshold,
            "N": len(sub_df),
            "Accuracy": f"{m_res['accuracy']*100:.2f}%",
            "Sensitivity": f"{m_res['recall']*100:.2f}%",
            "Specificity": f"{m_res['specificity']*100:.2f}%",
            "Precision": f"{m_res['precision']*100:.2f}%",
            "F1-Score": f"{m_res['f1']:.4f}",
            "ROC AUC": f"{m_res['auc']:.4f}",
            "TN": m_res["tn"],
            "FP": m_res["fp"],
            "FN": m_res["fn"],
            "TP": m_res["tp"],
        },
        {
            "Model Architecture": "EfficientNet-B0 (Judge)",
            "Cohort": source_choice,
            "Threshold": threshold,
            "N": len(sub_df),
            "Accuracy": f"{m_jd['accuracy']*100:.2f}%",
            "Sensitivity": f"{m_jd['recall']*100:.2f}%",
            "Specificity": f"{m_jd['specificity']*100:.2f}%",
            "Precision": f"{m_jd['precision']*100:.2f}%",
            "F1-Score": f"{m_jd['f1']:.4f}",
            "ROC AUC": f"{m_jd['auc']:.4f}",
            "TN": m_jd["tn"],
            "FP": m_jd["fp"],
            "FN": m_jd["fn"],
            "TP": m_jd["tp"],
        },
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    render_csv_download(comp_df, f"model_comparison_threshold_{threshold:.2f}.csv", "📥 Download Comparison Table (CSV)")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # Section C: Cross-Scanner Generalization
    # -------------------------------------------------------------------------
    st.subheader("2. Cross-Scanner Generalization Analysis")
    st.markdown(
        """
        > **Key Clinical Finding:** **AUC transfers across scanners but the 0.5 threshold does not.**  
        > When a model trained strictly on Shenzhen CXRs is tested on Montgomery CXRs (or vice-versa), 
        > ranking discriminability remains high (AUC 0.88 to 0.90), but baseline probability calibration drifts substantially.
        > Consequently, a fixed decision cutoff of 0.50 yields either severe under-calling or excessive false positives across different acquisition hardware.
        """
    )
    df_xscan = res_tables.get("classifier_crossscanner")
    if df_xscan is not None:
        st.dataframe(df_xscan, use_container_width=True, hide_index=True)
        render_csv_download(df_xscan, "classifier_crossscanner.csv", "📥 Download Cross-Scanner Table (CSV)")
    else:
        st.warning("classifier_crossscanner.csv not found in results directory.")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # Section D: Compression vs Diagnosis
    # -------------------------------------------------------------------------
    st.subheader("3. Compression vs Diagnostic Fidelity")
    st.caption("Interactive rate-distortion analysis examining diagnostic sensitivity retention across lossy codecs.")

    df_bench = res_tables.get("benchmark_summary")
    df_bench_src = res_tables.get("benchmark_summary_per_source")

    if df_bench is not None:
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            metric_target = st.selectbox(
                "Diagnostic Metric to Plot vs Bitrate (bpp):",
                ["auc", "recall", "specificity", "flip_rate"],
                format_func=lambda x: {
                    "auc": "ROC AUC",
                    "recall": "Sensitivity (Recall)",
                    "specificity": "Specificity",
                    "flip_rate": "Decision Flip Rate",
                }[x],
                index=0,
            )
        with b_col2:
            src_toggle = st.selectbox(
                "Cohort Subset:",
                ["Overall (All 800)", "Shenzhen Cohort", "Montgomery Cohort"],
                index=0,
            )

        # Select data
        if src_toggle == "Overall (All 800)":
            plot_bench_df = df_bench.copy()
        elif df_bench_src is not None:
            chosen_src = "Shenzhen" if "Shenzhen" in src_toggle else "Montgomery"
            plot_bench_df = df_bench_src[df_bench_src["source"] == chosen_src].copy()
        else:
            plot_bench_df = df_bench.copy()

        # Find original value
        orig_row = plot_bench_df[plot_bench_df["codec"] == "original"]
        orig_val = float(orig_row[metric_target].values[0]) if not orig_row.empty else None

        # Plot rate-distortion
        fig_rd, ax_rd = plt.subplots(figsize=(10, 4.8))
        codec_names = {
            "jpeg": ("JPEG", "#1f77b4", "o-"),
            "jpeg2000": ("JPEG 2000", "#ff7f0e", "s-"),
            "dct": ("Custom DCT", "#2ca02c", "^-"),
            "dwt": ("Custom DWT", "#d62728", "D-"),
            "roi_dwt": ("ROI-DWT (Near-lossless)", "#9467bd", "v-"),
        }

        for c_key, (c_label, c_color, c_marker) in codec_names.items():
            sub_c = plot_bench_df[plot_bench_df["codec"] == c_key].sort_values("mean_bpp")
            if not sub_c.empty:
                ax_rd.plot(
                    sub_c["mean_bpp"],
                    sub_c[metric_target],
                    c_marker,
                    color=c_color,
                    lw=2.0,
                    label=c_label,
                )

        if orig_val is not None:
            ax_rd.axhline(
                orig_val,
                color="black",
                linestyle="--",
                lw=1.5,
                label=f"Original Baseline ({orig_val:.4f})",
            )

        metric_labels = {
            "auc": "ROC AUC",
            "recall": "Sensitivity (Recall)",
            "specificity": "Specificity",
            "flip_rate": "Flip Rate (Fraction of Decisions Flipped)",
        }
        ax_rd.set_xlabel("Mean Bitrate (bpp — Bits Per Pixel)", fontsize=11)
        ax_rd.set_ylabel(metric_labels[metric_target], fontsize=11)
        ax_rd.set_title(f"{metric_labels[metric_target]} vs Bitrate across Codecs ({src_toggle})", fontsize=13, fontweight="bold")
        ax_rd.legend(loc="best", fontsize=10)
        ax_rd.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_rd)
        plt.close()

        # Operating points table
        st.markdown("##### Clinical Safe Operating Points")
        st.markdown(
            "> **Caption & Methodology:** The table below indicates the minimum bits-per-pixel (bpp) threshold required "
            "for each compression codec to guarantee that diagnostic decision flip rate remains below 5% or 2%, "
            "or that ROC AUC loss remains within 0.01 relative to uncompressed native images."
        )
        df_ops = res_tables.get("benchmark_operating_points")
        if df_ops is not None:
            st.dataframe(df_ops, use_container_width=True, hide_index=True)
            render_csv_download(df_ops, "benchmark_operating_points.csv", "📥 Download Operating Points (CSV)")
        else:
            st.warning("benchmark_operating_points.csv not found.")

        # Download benchmark summary
        render_csv_download(plot_bench_df, "benchmark_summary.csv", "📥 Download Full Benchmark Summary (CSV)")
    else:
        st.warning("benchmark_summary.csv not found in results directory.")
    st.markdown("---")

    # -------------------------------------------------------------------------
    # Section E: Lung Field Segmentation Performance
    # -------------------------------------------------------------------------
    st.subheader("4. Anatomical Lung Field Segmentation Evaluation")
    st.caption("Quantitative U-Net segmentation accuracy evaluated across 704 patient cases with expert radiologist ground-truth masks.")

    df_seg = res_tables.get("segmentation_summary")
    if df_seg is not None:
        st.dataframe(df_seg, use_container_width=True, hide_index=True)
        render_csv_download(df_seg, "segmentation_summary.csv", "📥 Download Segmentation Summary (CSV)")
    else:
        st.warning("segmentation_summary.csv not found in results directory.")

    # -------------------------------------------------------------------------
    # Section F: Research Disclaimer Footer
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="footer-text">
            <strong>RESEARCH PROTOTYPE DISCLAIMER:</strong><br>
            This application and its underlying neural network and codec architectures are investigational research prototypes 
            developed strictly for evaluating rate-distortion and diagnostic stability in medical imaging pipelines. 
            It is NOT a medical device, is NOT approved or cleared by any regulatory authority (including FDA or CE), 
            and is strictly not certified for primary clinical diagnosis, screening, triage, or clinical decision making.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
