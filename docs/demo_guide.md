# MedComp Streamlit Demo Guide

> **DISCLAIMER**:
> This software is an **investigational research prototype, NOT a clinical tool**.
> It is strictly not intended for primary clinical diagnosis, screening, patient triage, or medical decision making. All machine learning outputs represent synthetic probability metrics intended solely for benchmarking compression fidelity.

---

## 1. Quick Launch Instructions

### Prerequisites
- **Operating System**: Windows 11 / Windows Server
- **Hardware**: NVIDIA GeForce RTX 2050 (4 GB VRAM) with CUDA 12+ support
- **Virtual Environment**: `.venv` with PyTorch (CUDA-enabled) and Streamlit installed

### Launching the Application

1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd C:\project\medical
   ```

2. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Launch the Streamlit application:
   ```powershell
   streamlit run app/streamlit_app.py
   ```

4. The application will start and automatically open in your default web browser at:
   ```
   http://localhost:8501
   ```

5. **Headless Execution (for remote servers or automated smoke testing)**:
   ```powershell
   streamlit run app/streamlit_app.py --server.headless true --server.port 8501
   ```

---

## 2. Report Screenshot Checklist (4 Tabs)

When documenting or preparing experimental reports, capture high-resolution screenshots of each of the four main interface tabs:

- [ ] **Tab 1: Visual Inspection (`🖼️ Image Inspection`)**
  - **What to capture**: Side-by-side trio of Original (512x512), Decoded Reconstruction, and Absolute Difference map amplified by $\times 10$.
  - **Key visual focus**: High-contrast lung parenchyma boundaries and loss of fine trabecular texture under heavy compression.

- [ ] **Tab 2: Compression & Rate-Distortion (`📊 Compression & Size`)**
  - **What to capture**: Top metric cards (Compressed Size in bytes, Bitrate in bpp, Ratio vs Raw, Ratio vs PNG).
  - **Key visual focus**: The **Zone Byte Breakdown Stacked Bar Chart** (showing header, mask, core, band, and background portions for graded modes) and the **Full Mode Comparison Table**.

- [ ] **Tab 3: Anatomical Segmentation & Uncertainty (`🫁 Lung Diagnostics`)**
  - **What to capture**: The green contour overlay of predicted lung boundaries from the 5-fold U-Net ensemble alongside the **Monte Carlo Dropout Epistemic Uncertainty Heatmap** (magma colormap).
  - **Key visual focus**: Lower metric cards showing intra-lung reconstruction fidelity (Max Core Error, 99th percentile Core Error, Exact Lung Pixel %).

- [ ] **Tab 4: Clinical Flip Diagnosis (`🩺 Clinical Flip Diagnosis`)**
  - **What to capture**: Dual model prediction cards for ResNet-18 (Steering) and EfficientNet-B0 (Independent Judge).
  - **Key visual focus**: Original probability, decoded probability, numerical drift ($\Delta p$), and the color-coded decision status badge (`✅ Decision Preserved` vs `⚠️ DECISION FLIPPED`).

---

## 3. Demonstration Scenarios

### Scenario 1: Montgomery Chest X-ray at Standard DWT ($Q=50$)
* **Objective**: Illustrate standard wavelet compression baseline on a high-resolution clinical acquisition.
* **Input Image**:
  - Source: `Pick Dataset Sample`
  - Filename: `MCUCXR_0001_0.png`
* **Configuration**:
  - Mode: `Tuned Custom DWT (bior2.2, 4 Levels)`
  - Quality Factor: `50`
  - Fast Segmentation: `Unchecked` (Full 5-fold ensemble with 8-pass MC dropout)
* **Expected Results**:
  - **Bitrate**: $\approx 0.62\text{--}0.68\text{ bpp}$ (Compression Ratio $\approx 12\text{--}13\times$ vs raw 8-bit).
  - **Visuals**: Clean general lung visualization; slight edge attenuation visible in the $\times 10$ amplified difference map.
  - **Fidelity**: Decision preserved across both ResNet18 and Judge models without decision boundary crossing.

---

### Scenario 2: Shenzhen Chest X-ray with Graded Near-Lossless Encoding ($\delta_{\text{core}} = 2$)
* **Objective**: Demonstrate clinical ROI protection where lung fields have strict mathematical error bounds while background achieves high compression.
* **Input Image**:
  - Source: `Pick Dataset Sample`
  - Filename: `CHNCXR_0001_0.png`
* **Configuration**:
  - Mode: `Graded Near-Lossless (ROI + Band + DWT)`
  - Delta Core ($\delta_{\text{core}}$): `2` (Guarantees every lung pixel satisfies $|x - \hat{x}| \le 2$)
  - Delta Band ($\delta_{\text{band}}$): `4`
  - Band Dilation Radius: `8 pixels`
  - Background Quality: `50`
* **Expected Results**:
  - **Bitrate**: $\approx 1.20\text{--}1.25\text{ bpp}$.
  - **Tab 2 Breakdown**: Stacked bar shows compact mask representation, dense core byte allocation, and lossy background savings.
  - **Tab 3 Lung Diagnostics**: Max Core Error strictly $= 2.0$, 99th percentile Core Error $\le 2.0$.
  - **Fidelity**: 100% decision preservation on both diagnostic classifiers.

---

### Scenario 3: Heavy Compression Triggering Diagnostic Flip
* **Objective**: Demonstrate the critical failure mode of aggressive compression where non-ROI loss or severe blur triggers an automated false positive / false negative.
* **Input Image**:
  - Source: `Pick Dataset Sample`
  - Filename: `CHNCXR_0034_0.png`
* **Configuration**:
  - Mode: `Tuned Custom DWT (bior2.2, 4 Levels)`
  - Quality Factor: `10`
* **Expected Results**:
  - **Bitrate**: $\approx 0.056\text{ bpp}$ ($> 140\times$ compression).
  - **Tab 1 Visuals**: Extreme wavelet blur, loss of fine parenchymal lung features, high difference intensity across chest cavity.
  - **Tab 4 Diagnosis**:
    - **ResNet-18 (Steering)**:
      - Original Probability: $\approx 0.368$ (Negative TB, $< 0.50$).
      - Reconstructed Probability: $\approx 0.872$ (Positive TB, $\ge 0.50$).
      - Status: **`⚠️ CLASSIFIER DECISION FLIPPED!`**
    - **EfficientNet-B0 (Judge)**:
      - Original Probability: $\approx 0.446$ (Negative TB, $< 0.50$).
      - Reconstructed Probability: $\approx 0.551$ (Positive TB, $\ge 0.50$).
      - Status: **`⚠️ JUDGE DECISION FLIPPED!`**
  - **Clinical Implication**: Severe whole-image lossy compression falsely induces high-probability disease predictions, highlighting the necessity of closed-loop rate controllers and ROI-guaranteed codecs.

---

## 4. Troubleshooting & Notes

- **CUDA Out-of-Memory (OOM)**:
  - If multiple large applications occupy GPU VRAM, check the "Fast segmentation preview" checkbox in the sidebar. This switches the U-Net from the 5-model 8-pass ensemble to a single model 1-pass execution ($\approx 35\text{ MB}$ footprint).
- **Dataset Image Warning**:
  - When picking samples from the dataset dropdown, the sidebar displays an amber warning badge. This correctly reminds reviewers that the selected sample belonged to the cross-validation training/held-out folds.
- **Exporting Bitstreams**:
  - Use the download buttons in the bottom export section to obtain the exact serialized `.bin` bitstream, decoded `.png` image, and the comprehensive `.json` audit report containing every metric displayed on screen.
