"""Streamlit Multipage: Live Mini-Benchmark on Labelled Dataset Samples.

Conducts an interactive, statistically valid compression experiment on real labelled CXRs:
- Stratified sampling from data/manifest.csv.
- Zero-leakage out-of-fold inference using the held-out fold checkpoint for each image.
- Paired bootstrap confidence intervals for ΔAUC and ΔRecall.
- Full per-image audit trail export.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from medcomp.app_logic import (
    LoadedModels,
    classify_heldout,
    compress,
    load_models,
)
from medcomp.config import DATA_DIR, PROCESSED_IMAGES_DIR, WORK_DIR
from medcomp.metrics_report import classification_metrics

PRED_MASKS_DIR = WORK_DIR / "pred" / "mask"

# Page configuration
st.set_page_config(
    page_title="Mini-Benchmark Experiment | MedComp",
    page_icon="🧪",
    layout="wide",
)

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
        font-size: 1.5rem;
        font-weight: 700;
        color: #212529;
    }
    .metric-sub {
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


@st.cache_resource
def get_cached_models() -> LoadedModels:
    """Load neural network fold checkpoints on GPU."""
    return load_models()


def stratified_sample_manifest(
    manifest_df: pd.DataFrame,
    n_samples: int,
    source_filter: str = "All",
    seed: int = 42,
) -> pd.DataFrame:
    """Perform balanced stratified sampling by (source, label).

    Args:
        manifest_df: DataFrame loaded from data/manifest.csv.
        n_samples: Total number of samples to select.
        source_filter: 'All', 'Shenzhen', or 'Montgomery'.
        seed: Random seed for deterministic reproducibility.

    Returns:
        pd.DataFrame containing sampled rows.
    """
    df = manifest_df.copy()
    if source_filter != "All":
        df = df[df["source"] == source_filter].copy()

    total_available = len(df)
    n_target = min(n_samples, total_available)

    rng = np.random.default_rng(seed)

    # Group by strata (source and label)
    strata = df.groupby(["source", "label"])
    sampled_indices: list[int] = []

    # Proportional allocation
    for _, group in strata:
        group_indices = group.index.values
        proportion = len(group) / total_available
        n_group = max(1, int(round(proportion * n_target)))
        chosen = rng.choice(group_indices, size=min(n_group, len(group_indices)), replace=False)
        sampled_indices.extend(chosen.tolist())

    # Adjust exact length to n_target
    if len(sampled_indices) > n_target:
        sampled_indices = rng.choice(sampled_indices, size=n_target, replace=False).tolist()
    elif len(sampled_indices) < n_target:
        remaining = [i for i in df.index.values if i not in sampled_indices]
        if remaining:
            fill = rng.choice(remaining, size=min(n_target - len(sampled_indices), len(remaining)), replace=False)
            sampled_indices.extend(fill.tolist())

    return df.loc[sampled_indices].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def paired_bootstrap_delta(
    y_true: np.ndarray,
    prob_orig: np.ndarray,
    prob_dec: np.ndarray,
    threshold: float = 0.50,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute 95% paired bootstrap confidence intervals for ΔAUC and ΔRecall.

    Resamples patient index pairs with replacement.

    Returns:
        ((delta_auc_low, delta_auc_high), (delta_recall_low, delta_recall_high))
    """
    n = len(y_true)
    rng = np.random.default_rng(seed)

    delta_aucs: list[float] = []
    delta_recalls: list[float] = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        po = prob_orig[idx]
        pd_ = prob_dec[idx]

        m_o = classification_metrics(yt, po, threshold=threshold)
        m_d = classification_metrics(yt, pd_, threshold=threshold)

        if not np.isnan(m_d["auc"]) and not np.isnan(m_o["auc"]):
            delta_aucs.append(float(m_d["auc"] - m_o["auc"]))
        if not np.isnan(m_d["recall"]) and not np.isnan(m_o["recall"]):
            delta_recalls.append(float(m_d["recall"] - m_o["recall"]))

    if len(delta_aucs) > 0:
        d_auc_ci = (float(np.percentile(delta_aucs, 2.5)), float(np.percentile(delta_aucs, 97.5)))
    else:
        d_auc_ci = (float("nan"), float("nan"))

    if len(delta_recalls) > 0:
        d_rec_ci = (float(np.percentile(delta_recalls, 2.5)), float(np.percentile(delta_recalls, 97.5)))
    else:
        d_rec_ci = (float("nan"), float("nan"))

    return d_auc_ci, d_rec_ci


def main() -> None:
    st.markdown(
        """
        <div class="banner-warning">
            ⚠️ <strong>RESEARCH PROTOTYPE ONLY &mdash; NOT A CLINICAL TOOL</strong><br>
            This mini-benchmark executes live compression and deep neural network inference on real labelled images.
            Evaluation is strictly out-of-fold using held-out fold models with zero training leakage.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.title("🧪 Live Mini-Benchmark Experiment")
    st.markdown(
        """
        This tool executes a **statistically rigorous live experiment** on real patient chest X-rays.  
        **Leakage-Free Protocol:** For every sampled image, classification is executed **strictly using the model checkpoint of the fold in which that patient was held out** during 5-fold cross-validation.  
        *Neither the 5-fold ensemble average nor any model trained on the patient is ever used.*
        """
    )

    # Small-sample warning (always displayed)
    st.warning(
        "⚠️ **Small-Sample Notice:** This tool evaluates an interactive sub-cohort ($N = 10$ to $100$) for rapid parametric testing. "
        "Due to finite sample variance, confidence intervals are wider than the definitive full-cohort ($N = 800$) benchmarks."
    )
    st.markdown("---")

    manifest_path = DATA_DIR / "manifest.csv"
    if not manifest_path.is_file():
        st.error(f"data/manifest.csv not found at {manifest_path}!")
        return

    manifest_df = pd.read_csv(manifest_path)

    # -------------------------------------------------------------------------
    # Experiment Controls
    # -------------------------------------------------------------------------
    st.subheader("1. Experiment Parameters & Codec Configuration")

    c1, c2, c3 = st.columns(3)
    with c1:
        n_samples = st.slider("Sample Size N (images):", min_value=10, max_value=100, value=40, step=5)
    with c2:
        seed = st.number_input("Random Seed:", min_value=0, max_value=999999, value=42, step=1)
    with c3:
        source_filter = st.selectbox("Cohort Source Filter:", ["All", "Shenzhen", "Montgomery"], index=0)

    mode_options = {
        "graded": "Graded Near-Lossless (ROI + Band + DWT)",
        "lossless_dilated": "Lossless Dilated (Lossless Lung + DWT)",
        "dwt": "Tuned Custom DWT (bior2.2, 4 Levels)",
        "dct": "Custom Block DCT (8x8 JPEG Matrix)",
        "jpeg": "Standard JPEG (DCT)",
        "jpeg2000": "Standard JPEG 2000 (OpenJPEG DWT)",
        "png": "Lossless PNG (DEFLATE)",
    }

    mcol1, mcol2 = st.columns([1.5, 2.5])
    with mcol1:
        selected_mode = st.selectbox(
            "Compression Mode:",
            list(mode_options.keys()),
            format_func=lambda k: mode_options[k],
            index=0,
        )

    params: dict[str, Any] = {}
    with mcol2:
        if selected_mode == "graded":
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                params["delta_core"] = st.slider("Delta Core", 0, 10, 2)
            with g2:
                params["delta_band"] = st.slider("Delta Band", 0, 15, 4)
            with g3:
                params["band_px"] = st.slider("Band Radius", 1, 20, 8)
            with g4:
                params["bg_quality"] = st.slider("BG Quality", 10, 95, 50)
        elif selected_mode == "lossless_dilated":
            l1, l2 = st.columns(2)
            with l1:
                params["band_px"] = st.slider("Dilation Radius", 1, 20, 8)
            with l2:
                params["bg_quality"] = st.slider("BG Quality", 10, 95, 50)
        elif selected_mode in ("dwt", "dct", "jpeg"):
            params["quality"] = st.slider(f"{selected_mode.upper()} Quality Factor", 1, 100, 50)
        elif selected_mode == "jpeg2000":
            params["ratio"] = st.slider("JPEG 2000 Ratio", 2.0, 80.0, 10.0, step=1.0)

    run_btn = st.button("🚀 Run Live Mini-Benchmark", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # Execution & Session State Caching
    # -------------------------------------------------------------------------
    if run_btn:
        with st.spinner("Initializing models on RTX 2050 GPU..."):
            try:
                models = get_cached_models()
            except Exception as e:
                st.error(f"GPU model initialization failed: {e}")
                return

        sample_df = stratified_sample_manifest(
            manifest_df,
            n_samples=n_samples,
            source_filter=source_filter,
            seed=seed,
        )

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        records = []
        total_items = len(sample_df)

        for i, (_, row) in enumerate(sample_df.iterrows()):
            stem = str(row["stem"])
            label = int(row["label"])
            fold = int(row["fold"])
            source = str(row["source"])

            status_text.text(f"Processing image {i+1}/{total_items}: {stem} (Fold {fold})...")
            progress_bar.progress((i + 1) / total_items)

            # 1. Load image
            img_path = PROCESSED_IMAGES_DIR / f"{stem}.png"
            if not img_path.is_file():
                # Fallback to Lung Segmentation directory
                raw_path = SRC_DIR.parent / str(row["image_path"])
                raw_img = cv2.imread(str(raw_path), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(raw_img, (512, 512))
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            # 2. Mask loading (if required)
            mask_bool = None
            if selected_mode in ("graded", "lossless_dilated"):
                mask_path = PRED_MASKS_DIR / f"{stem}.png"
                if mask_path.is_file():
                    m_arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    mask_bool = (m_arr > 127)
                else:
                    mask_bool = np.ones_like(img, dtype=bool)

            # 3. Compression & bitstream-only decompression
            c_res = compress(img, mask=mask_bool, mode=selected_mode, params=params)
            recon = c_res["recon"]

            # 4. Strictly out-of-fold heldout classification
            c_orig = classify_heldout(img, fold=fold, models=models)
            c_dec = classify_heldout(recon, fold=fold, models=models)

            p_ro = float(c_orig["resnet_prob"])
            p_rd = float(c_dec["resnet_prob"])
            p_jo = float(c_orig["judge_prob"])
            p_jd = float(c_dec["judge_prob"])

            flip_res = bool((p_rd >= 0.50) != (p_ro >= 0.50))
            flip_jd = bool((p_jd >= 0.50) != (p_jo >= 0.50))

            records.append({
                "stem": stem,
                "source": source,
                "label": label,
                "fold": fold,
                "bytes": int(c_res["num_bytes"]),
                "bpp": float(c_res["bpp"]),
                "ratio_vs_raw": float(c_res["ratio_vs_raw"]),
                "prob_orig_resnet": p_ro,
                "prob_dec_resnet": p_rd,
                "flip_resnet": flip_res,
                "prob_orig_judge": p_jo,
                "prob_dec_judge": p_jd,
                "flip_judge": flip_jd,
            })

        progress_bar.empty()
        status_text.empty()

        # Cache in session state
        st.session_state["mini_bench_results"] = {
            "df": pd.DataFrame(records),
            "mode": selected_mode,
            "params": params,
            "n_samples": len(records),
            "seed": seed,
            "source_filter": source_filter,
        }

    # -------------------------------------------------------------------------
    # Render Benchmark Results
    # -------------------------------------------------------------------------
    if "mini_bench_results" in st.session_state:
        bench_data = st.session_state["mini_bench_results"]
        df_res = bench_data["df"]
        N = len(df_res)

        st.markdown("---")
        st.subheader(f"2. Evaluation Results (N = {N} Images, Mode: {bench_data['mode'].upper()})")

        if N < 40:
            st.info("ℹ️ **Statistical Note:** Sample size N < 40. ROC AUC estimates have wide variance; paired bootstrap intervals are enabled for N >= 40.")

        y_true = df_res["label"].values.astype(int)
        p_res_orig = df_res["prob_orig_resnet"].values.astype(float)
        p_res_dec = df_res["prob_dec_resnet"].values.astype(float)
        p_jd_orig = df_res["prob_orig_judge"].values.astype(float)
        p_jd_dec = df_res["prob_dec_judge"].values.astype(float)

        # Compression efficiency summary
        mean_bpp = float(np.mean(df_res["bpp"]))
        mean_cr = float(np.mean(df_res["ratio_vs_raw"]))
        flip_rate_res = float(np.mean(df_res["flip_resnet"])) * 100.0
        flip_rate_jd = float(np.mean(df_res["flip_judge"])) * 100.0

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Mean Bitrate", f"{mean_bpp:.3f} bpp")
        r2.metric("Mean Compression Ratio", f"{mean_cr:.2f}×")
        r3.metric("ResNet18 Decision Flip Rate", f"{flip_rate_res:.1f}%")
        r4.metric("Judge Decision Flip Rate", f"{flip_rate_jd:.1f}%")

        # Side-by-side performance table for Original vs Decoded
        m_ro = classification_metrics(y_true, p_res_orig, threshold=0.50)
        m_rd = classification_metrics(y_true, p_res_dec, threshold=0.50)
        m_jo = classification_metrics(y_true, p_jd_orig, threshold=0.50)
        m_jd = classification_metrics(y_true, p_jd_dec, threshold=0.50)

        st.markdown("##### Side-by-Side Model Diagnostics: Original vs Decoded")
        side_table = pd.DataFrame([
            {
                "Model Architecture": "ResNet18 (Steering)",
                "Image Domain": "Original (Native)",
                "Accuracy": f"{m_ro['accuracy']*100:.1f}%",
                "Sensitivity": f"{m_ro['recall']*100:.1f}%",
                "Specificity": f"{m_ro['specificity']*100:.1f}%",
                "Precision": f"{m_ro['precision']*100:.1f}%",
                "F1-Score": f"{m_ro['f1']:.3f}",
                "ROC AUC": f"{m_ro['auc']:.3f}" if not np.isnan(m_ro["auc"]) else "N/A",
            },
            {
                "Model Architecture": "ResNet18 (Steering)",
                "Image Domain": f"Decoded ({bench_data['mode']})",
                "Accuracy": f"{m_rd['accuracy']*100:.1f}%",
                "Sensitivity": f"{m_rd['recall']*100:.1f}%",
                "Specificity": f"{m_rd['specificity']*100:.1f}%",
                "Precision": f"{m_rd['precision']*100:.1f}%",
                "F1-Score": f"{m_rd['f1']:.3f}",
                "ROC AUC": f"{m_rd['auc']:.3f}" if not np.isnan(m_rd["auc"]) else "N/A",
            },
            {
                "Model Architecture": "EfficientNet-B0 (Judge)",
                "Image Domain": "Original (Native)",
                "Accuracy": f"{m_jo['accuracy']*100:.1f}%",
                "Sensitivity": f"{m_jo['recall']*100:.1f}%",
                "Specificity": f"{m_jo['specificity']*100:.1f}%",
                "Precision": f"{m_jo['precision']*100:.1f}%",
                "F1-Score": f"{m_jo['f1']:.3f}",
                "ROC AUC": f"{m_jo['auc']:.3f}" if not np.isnan(m_jo["auc"]) else "N/A",
            },
            {
                "Model Architecture": "EfficientNet-B0 (Judge)",
                "Image Domain": f"Decoded ({bench_data['mode']})",
                "Accuracy": f"{m_jd['accuracy']*100:.1f}%",
                "Sensitivity": f"{m_jd['recall']*100:.1f}%",
                "Specificity": f"{m_jd['specificity']*100:.1f}%",
                "Precision": f"{m_jd['precision']*100:.1f}%",
                "F1-Score": f"{m_jd['f1']:.3f}",
                "ROC AUC": f"{m_jd['auc']:.3f}" if not np.isnan(m_jd["auc"]) else "N/A",
            },
        ])
        st.dataframe(side_table, use_container_width=True, hide_index=True)

        # Paired Bootstrap Differences (for N >= 40)
        if N >= 40:
            st.markdown("##### Paired Bootstrap Degradation Analysis (ΔAUC & ΔRecall)")
            d_auc_res = float(m_rd["auc"] - m_ro["auc"]) if not (np.isnan(m_rd["auc"]) or np.isnan(m_ro["auc"])) else float("nan")
            d_rec_res = float(m_rd["recall"] - m_ro["recall"])
            d_auc_ci_r, d_rec_ci_r = paired_bootstrap_delta(y_true, p_res_orig, p_res_dec)

            d_auc_jd = float(m_jd["auc"] - m_jo["auc"]) if not (np.isnan(m_jd["auc"]) or np.isnan(m_jo["auc"])) else float("nan")
            d_rec_jd = float(m_jd["recall"] - m_jo["recall"])
            d_auc_ci_j, d_rec_ci_j = paired_bootstrap_delta(y_true, p_jd_orig, p_jd_dec)

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown(
                    f"""
                    **ResNet18 Degradation:**
                    - **ΔAUC:** `{d_auc_res:+.4f}` [95% CI: `{d_auc_ci_r[0]:+.4f}` to `{d_auc_ci_r[1]:+.4f}`]
                    - **ΔRecall:** `{d_rec_res*100:+.2f}%` [95% CI: `{d_rec_ci_r[0]*100:+.2f}%` to `{d_rec_ci_r[1]*100:+.2f}%`]
                    """
                )
            with pcol2:
                st.markdown(
                    f"""
                    **EfficientNet-B0 Judge Degradation:**
                    - **ΔAUC:** `{d_auc_jd:+.4f}` [95% CI: `{d_auc_ci_j[0]:+.4f}` to `{d_auc_ci_j[1]:+.4f}`]
                    - **ΔRecall:** `{d_rec_jd*100:+.2f}%` [95% CI: `{d_rec_ci_j[0]*100:+.2f}%` to `{d_rec_ci_j[1]*100:+.2f}%`]
                    """
                )

        # Confusion Matrices Display
        st.markdown("##### Confusion Matrices Comparison")
        cm_col1, cm_col2 = st.columns(2)

        with cm_col1:
            fig_r, (ax_ro, ax_rd) = plt.subplots(1, 2, figsize=(7.0, 3.2))
            cm_ro = np.array([[m_ro["tn"], m_ro["fp"]], [m_ro["fn"], m_ro["tp"]]])
            cm_rd = np.array([[m_rd["tn"], m_rd["fp"]], [m_rd["fn"], m_rd["tp"]]])

            ax_ro.imshow(cm_ro, cmap="Blues")
            ax_ro.set_title("ResNet18 (Orig)", fontsize=10, fontweight="bold")
            ax_ro.set_xticks([0, 1]); ax_ro.set_yticks([0, 1])
            ax_ro.set_xticklabels(["Neg", "Pos"]); ax_ro.set_yticklabels(["Neg", "Pos"])
            for (r, c), v in np.ndenumerate(cm_ro):
                ax_ro.text(c, r, str(v), ha="center", va="center", fontweight="bold")

            ax_rd.imshow(cm_rd, cmap="Blues")
            ax_rd.set_title("ResNet18 (Dec)", fontsize=10, fontweight="bold")
            ax_rd.set_xticks([0, 1]); ax_rd.set_yticks([0, 1])
            ax_rd.set_xticklabels(["Neg", "Pos"]); ax_rd.set_yticklabels(["Neg", "Pos"])
            for (r, c), v in np.ndenumerate(cm_rd):
                ax_rd.text(c, r, str(v), ha="center", va="center", fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig_r)
            plt.close()

        with cm_col2:
            fig_j, (ax_jo, ax_jd) = plt.subplots(1, 2, figsize=(7.0, 3.2))
            cm_jo = np.array([[m_jo["tn"], m_jo["fp"]], [m_jo["fn"], m_jo["tp"]]])
            cm_jd = np.array([[m_jd["tn"], m_jd["fp"]], [m_jd["fn"], m_jd["tp"]]])

            ax_jo.imshow(cm_jo, cmap="Greens")
            ax_jo.set_title("Judge (Orig)", fontsize=10, fontweight="bold")
            ax_jo.set_xticks([0, 1]); ax_jo.set_yticks([0, 1])
            ax_jo.set_xticklabels(["Neg", "Pos"]); ax_jo.set_yticklabels(["Neg", "Pos"])
            for (r, c), v in np.ndenumerate(cm_jo):
                ax_jo.text(c, r, str(v), ha="center", va="center", fontweight="bold")

            ax_jd.imshow(cm_jd, cmap="Greens")
            ax_jd.set_title("Judge (Dec)", fontsize=10, fontweight="bold")
            ax_jd.set_xticks([0, 1]); ax_jd.set_yticks([0, 1])
            ax_jd.set_xticklabels(["Neg", "Pos"]); ax_jd.set_yticklabels(["Neg", "Pos"])
            for (r, c), v in np.ndenumerate(cm_jd):
                ax_jd.text(c, r, str(v), ha="center", va="center", fontweight="bold")

            plt.tight_layout()
            st.pyplot(fig_j)
            plt.close()

        # -------------------------------------------------------------------------
        # Per-Image Audit Table & CSV Download
        # -------------------------------------------------------------------------
        st.markdown("##### Detailed Per-Image Audit Trail")
        st.dataframe(df_res, use_container_width=True, hide_index=True)

        csv_bytes = df_res.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Per-Image Benchmark Audit (CSV)",
            data=csv_bytes,
            file_name=f"minibenchmark_n{N}_{bench_data['mode']}_audit.csv",
            mime="text/csv",
            use_container_width=False,
        )

    # -------------------------------------------------------------------------
    # Research Disclaimer Footer
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
