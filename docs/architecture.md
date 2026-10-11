# System Architecture and Bitstream Specification

**Project:** Region-Aware Lossy–Lossless Compression of Chest Radiographs with Diagnostic Fidelity Evaluation  
**Package:** `medcomp` (`src/medcomp/`)  
**Standard:** Production research prototype specification.

---

## 1. Module Map (`src/medcomp/`)

| Filename | Responsibility / Description |
|:---|:---|
| [`__init__.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/__init__.py) | Package initialization, public symbol exports, and semantic version definition. |
| [`app_logic.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/app_logic.py) | Application orchestration layer connecting compression, segmentation, and classification for the Streamlit demo. |
| [`baselines.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/baselines.py) | Standard codec wrappers for baseline evaluation (lossless PNG, classical JPEG, and JPEG 2000 Part 1). |
| [`classifier.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/classifier.py) | ResNet-18 TB steering classifier implementation, inference, out-of-fold scoring, and bootstrap CI metrics. |
| [`cls_data.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/cls_data.py) | Dataset loading, normalization transforms, and PyTorch dataloaders for classifier training and evaluation. |
| [`config.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/config.py) | Centralized filesystem directory paths, quality ladders, hardware configurations, and global constants. |
| [`controller.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/controller.py) | Closed-loop adaptive bitrate controller algorithms (V0–V4), bisection solvers, and oracle judge baselines. |
| [`data_utils.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/data_utils.py) | Manifest parsing, cohort filtering, stratified 5-fold cross-validation generation, and DICOM/PNG discovery. |
| [`dct_codec.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/dct_codec.py) | Custom 8×8 block Discrete Cosine Transform codec with JPEG quantization scaling and canonical Huffman coding. |
| [`dwt_codec.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/dwt_codec.py) | Tuned multi-resolution DWT codec with biorthogonal wavelets (`bior2.2`), dead-zone quantization, and context coding. |
| [`entropy.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/entropy.py) | Canonical Huffman table generator, min-heap builder, BitReader, BitWriter, and Shannon entropy calculations. |
| [`graded_codec.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/graded_codec.py) | Multi-zone graded compression engine combining near-lossless lung core, safety buffer band, and lossy background. |
| [`io_utils.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/io_utils.py) | Resilient image I/O supporting 8-bit, 16-bit, and DICOM VOI LUT windowing, zero-padding, and dimensional clipping. |
| [`judge.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/judge.py) | Independent EfficientNet-B0 evaluation judge architecture and inference harness for cross-model validation. |
| [`lossless_codec.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/lossless_codec.py) | Bit-exact ROI lossless codec using Median Edge Detection (MED) prediction and 8-bit modular residual wrapping. |
| [`mask_codec.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/mask_codec.py) | Compact binary segmentation mask coder utilizing bounding-box localization and run-length bitstream packing. |
| [`metrics.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/metrics.py) | Evaluates distortion and rate metrics: PSNR, SSIM, masked ROI SSIM, BD-PSNR, bits per pixel, and compression ratio. |
| [`nearlossless.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/nearlossless.py) | Deterministic bounded-error spatial coding engine ($L_\infty \le \delta$) with causal feedback and boundary clipping. |
| [`roi_codec.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/roi_codec.py) | Multiplexing container combining bit-exact lossless ROI lungs, compressed masks, and lossy background streams. |
| [`seg_data.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/seg_data.py) | PyTorch Dataset pipelines, spatial augmentations, and mask alignment tensors for U-Net segmentation training. |
| [`unet.py`](file:///c:/Users/kondr/Downloads/Medical-Image-Compression1-main%2024/Medical-Image-Compression1-main/src/medcomp/unet.py) | PyTorch U-Net neural network with skip connections and test-time Monte Carlo dropout for epistemic uncertainty. |

---

## 2. Bitstream Container Layouts

All custom codecs in this repository serialize into self-contained binary byte arrays. Reported file sizes, bitrates, and compression ratios strictly measure the total byte length of the serialized container from byte `0` to `EOF`, including all magic numbers, headers, metadata tables, and packed payloads.

```
+--------------------------------------------------------------------------------+
|  Magic Bytes  |  Header / Geometry Fields  |  Table / Metadata  |  Bitstream   |
+--------------------------------------------------------------------------------+
```

### 2.1 Custom DCT Codec (`src/medcomp/dct_codec.py`)
- **Magic Identifier:** `b"MEDDCT10"` (8 bytes ASCII)
- **Header Structure:** 5 bytes (`>HHB`)
  - `orig_h` (uint16, 2 bytes): Original image height in pixels.
  - `orig_w` (uint16, 2 bytes): Original image width in pixels.
  - `quality` (uint8, 1 byte): Scaling quality factor $Q \in [1, 100]$.
- **Table Data Section:**
  - `dc_lengths` (16 bytes): Canonical Huffman code lengths for DC categories (0–15).
  - `ac_lengths` (256 bytes): Canonical Huffman code lengths for AC `(run, size)` symbols (0–255).
- **Bitstream Section:**
  - Variable-length packed bitstream containing DC differential categories + amplitude bits, followed by AC `(run, size)` Huffman codes + amplitude bits, terminated by block `EOB`.
- **Size Accounting:** Total bytes = $8 + 5 + 16 + 256 + \text{len}(\text{bitstream})$.

---

### 2.2 Tuned DWT Codec (`src/medcomp/dwt_codec.py`)
- **Magic Identifier:**
  - `b"MEDDWT10"` (8 bytes ASCII, Legacy uniform DWT)
  - `b"MEDDWT20"` (8 bytes ASCII, Tuned subband-weighted DWT)
- **Header Structure:**
  - Legacy (`>HHBBfB`, 11 bytes): `height`, `width`, `wavelet_id`, `levels`, `quality`, `bias`, `scheme_id`.
  - Tuned (`>HHBBBHHBB`, 13 bytes): `height`, `width`, `wavelet_id`, `levels`, `quality`, `theta_u16`, `bias_u16`, `scheme_id`, `context_id`.
- **Subband / Geometry Section:**
  - Quantization step table $\Delta_{b}$ for each decomposition level and orientation.
- **Bitstream Section:**
  - Canonical Huffman tables followed by context-conditioned run-length bitstream for subband coefficients.
- **Size Accounting:** Total bytes = $8 + 13 + \text{header\_tables} + \text{len}(\text{bitstream})$.

---

### 2.3 Lossless ROI Codec (`src/medcomp/lossless_codec.py`)
- **Magic Identifier:** `b"MEDLS10"` (7 bytes ASCII)
- **Header Structure:** 11 bytes (`>HHBBIB`)
  - `h` (uint16, 2 bytes): Image height.
  - `w` (uint16, 2 bytes): Image width.
  - `has_mask` (uint8, 1 byte): Boolean flag ($1 = \text{ROI raster stream}$, $0 = \text{Full image}$).
  - `context_flag` (uint8, 1 byte): Gradient-adaptive context conditioning ($1 = \text{Active}$, $0 = \text{Off}$).
  - `n_pixels` (uint32, 4 bytes): Total number of encoded pixels.
  - `num_contexts` (uint8, 1 byte): Number of context categories (1 or 4).
- **Table Data Section:**
  - Serialized canonical Huffman code length arrays (256 bytes per active context).
- **Bitstream Section:**
  - Bit-packed stream of modular 8-bit residuals ($e = (x - \hat{x}) \bmod 256$).
- **Size Accounting:** Total bytes = $7 + 11 + \text{table\_bytes} + \text{len}(\text{bitstream})$.

---

### 2.4 Binary Mask Codec (`src/medcomp/mask_codec.py`)
- **Magic Identifier:** `b"MEDMSK10"` (8 bytes ASCII)
- **Header Structure:** 5 bytes (`>HHB`)
  - `h` (uint16, 2 bytes): Mask grid height.
  - `w` (uint16, 2 bytes): Mask grid width.
  - `bbox_mode` (uint8, 1 byte): $1 = \text{Bounding Box Cropped}$, $0 = \text{Full Frame}$.
- **Bounding Box Bounds (if `bbox_mode == 1`):** 8 bytes (`>HHHH`)
  - `r_min`, `r_max`, `c_min`, `c_max` (uint16 each).
- **Bitstream Section:**
  - Row-wise run-length packed bitstream encoding alternating 0/1 run intervals.
- **Size Accounting:** Total bytes = $8 + 5 + (8 \text{ if bbox else } 0) + \text{len}(\text{rle\_bitstream})$. Mean size = 1,618.8 bytes.

---

### 2.5 Hybrid ROI Container Codec (`src/medcomp/roi_codec.py`)
- **Magic Identifier:** `b"MEDROI10"` (8 bytes ASCII)
- **Header Structure:** 14 bytes (`>BBIII`)
  - `method_id` (uint8, 1 byte): Background codec type ($1 = \text{DCT}$, $2 = \text{DWT}$).
  - `bg_quality` (uint8, 1 byte): Lossy background quality setting $Q \in [10, 90]$.
  - `mask_len` (uint32, 4 bytes): Length of embedded `MEDMSK10` stream in bytes.
  - `roi_len` (uint32, 4 bytes): Length of embedded `MEDLS10` stream in bytes.
  - `bg_len` (uint32, 4 bytes): Length of embedded background bitstream (`MEDDCT10`/`MEDDWT20`) in bytes.
- **Payload Streams:**
  - `[0 : mask_len]`: Complete standalone binary mask stream.
  - `[0 : roi_len]`: Complete standalone bit-exact lossless ROI stream.
  - `[0 : bg_len]`: Complete standalone lossy background stream.
- **Size Accounting:** Total bytes = $8 + 14 + \text{mask\_len} + \text{roi\_len} + \text{bg\_len}$.

---

### 2.6 Near-Lossless Codec (`src/medcomp/nearlossless.py`)
- **Magic Identifier:** `b"MEDNL10"` (7 bytes ASCII)
- **Header Structure:** 22 bytes (`>HHBBIIII`)
  - `h`, `w` (uint16, 4 bytes): Image dimensions.
  - `delta1`, `delta2` (uint8, 2 bytes): Deterministic error bounds for Zone 1 and Zone 2.
  - `n_z1`, `n_z2` (uint32, 8 bytes): Number of pixels in Zone 1 and Zone 2.
  - `core_bytes`, `band_bytes` (uint32, 8 bytes): Payload lengths of Zone 1 and Zone 2 bitstreams.
- **Payload Streams:**
  - Zone 1 and Zone 2 quantizer residual streams with closed-loop causal clipping.
- **Size Accounting:** Total bytes = $7 + 22 + \text{core\_bytes} + \text{band\_bytes}$.

---

### 2.7 Graded Multi-Zone Codec (`src/medcomp/graded_codec.py`)
- **Magic Identifier:** `b"MEDGRD10"` (8 bytes ASCII)
- **Header Structure:** 24 bytes (`>HHBBBBBBIIII`)
  - `h`, `w` (uint16, 4 bytes): Image dimensions.
  - `delta_core` (uint8, 1 byte): Hard error bound for lung core ($\delta \in \{0, 1, 2, 3\}$).
  - `delta_band` (uint8, 1 byte): Hard error bound for safety buffer band.
  - `band_px` (uint8, 1 byte): Morphological dilation radius in pixels ($M \in [0, 20]$).
  - `method_id` (uint8, 1 byte): Background lossy codec identifier.
  - `bg_quality` (uint8, 1 byte): Quality factor for background channel.
  - `has_extra` (uint8, 1 byte): Extension flag.
  - `mask_len`, `core_len`, `band_len`, `bg_len` (uint32 each, 16 bytes): Stream segment byte lengths.
- **Payload Streams:**
  - Sequentially concatenates: `mask_bytes` + `core_bytes` + `band_bytes` + `bg_bytes`.
- **Size Accounting:** Total bytes = $8 + 24 + \text{mask\_len} + \text{core\_len} + \text{band\_len} + \text{bg\_len}$.

---

## 3. Definitions and Metric Conventions

### 3.1 Rate Metrics: bpp and Compression Ratio

1. **Full-Image Bits Per Pixel ($\text{bpp}$):**
   Standardized metric normalizing total compressed bytes across the full image grid ($512 \times 512 = 262,144$ pixels):
   $$\text{bpp} = \frac{8 \times \text{total\_compressed\_bytes}}{512 \times 512}$$
   Used across all whole-image codecs (DCT, DWT, JPEG, JPEG 2000), hybrid codecs (`roi_codec.py`), and baseline summaries.

2. **ROI-Pixel Bits Per Pixel ($\text{bpp}_{\text{ROI}}$):**
   Measures bits allocated strictly per lung parenchyma pixel:
   $$\text{bpp}_{\text{ROI}} = \frac{8 \times \text{roi\_bytes}}{\text{roi\_pixels}}$$
   On our dataset, mean $\text{roi\_pixels} = 60,976.1$ pixels (23.27% of image area). Lossless lung coding consumes $\text{mean bpp}_{\text{ROI}} = 4.1544$ bits/pixel.

3. **Compression Ratio ($\text{CR}$):**
   The ratio of raw uncompressed image bytes to total compressed file bytes:
   $$\text{CR} = \frac{\text{uncompressed\_bytes}}{\text{compressed\_bytes}} = \frac{262,144}{\text{total\_compressed\_bytes}} = \frac{8.00}{\text{bpp}}$$

---

### 3.2 Distortion Metrics: PSNR and SSIM

1. **Peak Signal-to-Noise Ratio ($\text{PSNR}$):**
   Logarithmic ratio of peak intensity ($MAX = 255$) to Mean Squared Error ($\text{MSE}$):
   $$\text{MSE} = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} [I(i, j) - \hat{I}(i, j)]^2$$
   $$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{255^2}{\text{MSE}} \right) \quad [\text{dB}]$$
   If $\text{MSE} = 0$, $\text{PSNR} = \infty$.

2. **Structural Similarity Index ($\text{SSIM}$):**
   Evaluates luminance ($l$), contrast ($c$), and structure ($s$) using an 11×11 Gaussian window ($\sigma = 1.5$):
   $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
   where $C_1 = (0.01 \times 255)^2 = 6.5025$ and $C_2 = (0.03 \times 255)^2 = 58.5225$.

3. **Masked ROI Distortion ($\text{PSNR}_{\text{ROI}}$, $\text{SSIM}_{\text{ROI}}$):**
   Evaluates distortion strictly over pixels where the binary lung mask $M(i, j) == 1$. For bit-exact ROI coding, $\text{MSE}_{\text{ROI}} = 0$, $\text{PSNR}_{\text{ROI}} = \infty$, and $\text{SSIM}_{\text{ROI}} \equiv 1.0000$.

4. **Bjontegaard Delta PSNR ($\text{BD-PSNR}$):**
   Integrates the average vertical PSNR difference between two rate-distortion curves over their overlapping logarithmic bitrate span using cubic polynomial curve fitting. Positive values indicate superior reconstruction quality at equivalent bitrate.

---

### 3.3 ROI and Multi-Zone Terminology

```
+-------------------------------------------------------------+
| Background (Zone 0)                                         |
|    Lossy DCT / DWT (Q = 10..90)                             |
|                                                             |
|         +-----------------------------------------+         |
|         | Safety Buffer Band (Zone 2)             |         |
|         |    Morphological Dilation (M px)        |         |
|         |    Near-lossless (|e| <= delta_band)    |         |
|         |                                         |         |
|         |      +---------------------------+      |         |
|         |      | Lung Core (Zone 1)        |      |         |
|         |      |    True Lung Parenchyma   |      |         |
|         |      |    |e| <= delta_core      |      |         |
|         |      +---------------------------+      |         |
|         +-----------------------------------------+         |
+-------------------------------------------------------------+
```

- **Lung ROI (Region of Interest / Diagnostic Core):**
  The anatomical lung parenchyma delineated by manual ground truth or U-Net segmentation. Carries 85–90% of downstream diagnostic signal for pulmonary pathology.
- **Safety Buffer Band:**
  A morphological boundary zone generated by dilating the lung mask by radius $M \in [1, 20]$ pixels. Absorbs segmentation boundary errors, preventing peripheral subpleural infiltrates from leaking into aggressive background quantization.
- **Background Zone:**
  The thoracic cavity outside the lung masks—including mediastinum, ribs, clavicles, subdiaphragmatic space, and unexposed collimation borders. Compressed using aggressive lossy DCT or DWT.
- **Near-Lossless Bounded Error ($\delta$):**
  A certified deterministic maximum reconstruction bound guaranteeing $|I(i, j) - \hat{I}(i, j)| \leq \delta$ for all pixels within the designated zone.
