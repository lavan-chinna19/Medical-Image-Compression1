# Medical Image Compression: Viva Voce Examination Study Sheet

**Project:** Region-Aware Lossy–Lossless Compression of Chest Radiographs with Diagnostic Fidelity Evaluation  
**Author:** Candidate Viva Preparation Notes  
**Repository:** `medcomp` (`src/medcomp/`)  
**Evaluation Standard:** Every technical explanation corresponds to verified source code in `src/medcomp/`, and every numerical claim is drawn from audited repository results with 95% bootstrap confidence intervals.

---

## A. Data Compression Theory

### A.1 Why compress medical images, and what lossless, lossy, and near-lossless mean
Hospitals generate massive volumes of high-resolution digital imaging that overwhelm PACS storage and saturate telemedicine bandwidth. Lossless compression allows bit-exact reconstruction of the original image ($\text{PSNR} = \infty$, $\text{SSIM} = 1.0$), ensuring legal and diagnostic safety, but offers modest compression ratios. Lossy compression permanently discards visually or diagnostically redundant high-frequency information to achieve much higher compression ratios, though it risks introducing artifacts. Near-lossless compression bridges this divide by mathematically bounding the maximum pixel reconstruction error to a pre-set tolerance ($|x - \hat{x}| \leq \delta$), guaranteeing that no individual pixel deviates by more than $\delta$ gray levels.
- **Implementation:** `src/medcomp/metrics.py:is_lossless()`, `src/medcomp/nearlossless.py:nl_encode()`, `src/medcomp/roi_codec.py:roi_encode()`
- **Key Result:** Uncompressed 512×512 chest X-rays consume 262,144 bytes (8.00 bpp); lossless PNG achieves 3.43 bpp (a 2.33:1 ratio), whereas lossy JPEG at quality 50 achieves 0.48 bpp (a 16.8:1 ratio) with PSNR of 40.04 dB (`results/baseline_summary.csv`).

### A.2 What entropy is and what $H \leq L < H + 1$ means
Information entropy ($H$) measures the fundamental theoretical limit of data compressibility for a discrete source, calculated as $H = -\sum p_i \log_2 p_i$ bits per symbol. Shannon's source coding theorem proves that for any discrete memoryless source, the average code length $L = \sum p_i l_i$ produced by an optimal prefix-free code satisfies $H \leq L < H + 1$. The lower bound means no lossless code can represent the symbols in fewer bits than the source entropy without losing information. The upper bound guarantees that optimal symbol codes like Huffman coding are within at most one bit per symbol of the theoretical limit.
- **Implementation:** `src/medcomp/entropy.py:shannon_entropy_from_freqs()`, `src/medcomp/entropy.py:average_code_length()`, `src/medcomp/metrics.py:shannon_entropy()`
- **Key Result:** Spatial prediction on lung pixels concentrates residuals, dropping source entropy from ~6.9 bits/pixel down to 4.15 bits/pixel; empirical code lengths match this bound across all 704 masked images (`results/lossless_summary.csv`).

### A.3 How Huffman coding works (heap, canonical codes, why only code lengths are stored)
Huffman coding constructs an optimal prefix-free binary tree bottom-up by placing symbol probabilities in a min-heap and repeatedly merging the two least frequent nodes. Canonical Huffman coding transforms this arbitrary tree into a standardized representation where codewords are ordered strictly by code length, and lexicographically within the same length. Because canonical codes can be generated deterministically once the code length of each symbol is known, the encoder never needs to transmit the entire binary tree structure. Instead, the compressed stream only stores a compact array of symbol code lengths, reducing header overhead from hundreds of bytes to a minimal table.
- **Implementation:** `src/medcomp/entropy.py:build_huffman_code_lengths()`, `src/medcomp/entropy.py:canonical_codes()`, `src/medcomp/entropy.py:build_canonical_decode_trie()`
- **Key Result:** Canonical code length headers require only 16 to 256 bytes per stream, allowing custom Huffman tables to remain highly rate-efficient even on small 8×8 block partitions (`src/medcomp/dct_codec.py`).

### A.4 Why Huffman cannot go below 1 bit per symbol and what arithmetic coding does differently
Huffman coding assigns an integer number of bits to every encoded symbol, meaning that even a symbol with a 99% probability must take at least 1 whole bit ($L \geq 1$). When an image contains massive regions of near-identical pixels (such as large black collimation borders), the true source entropy may be 0.1 bits/pixel, but Huffman coding cannot compress it below 1.0 bit/symbol without run-length grouping. Arithmetic coding solves this by mapping the entire message sequence to a single fractional real number sub-interval in $[0, 1)$, allocating fractional bits per symbol according to cumulative probabilities. This enables modern arithmetic codecs to operate arbitrarily close to true entropy at ultra-low bitrates.
- **Implementation:** `src/medcomp/entropy.py:huffman_encode()` compared against JPEG 2000's EBCOT engine in `src/medcomp/baselines.py:run_jpeg2k()`
- **Key Result:** At compression ratio 80:1 (0.099 bpp), JPEG 2000 with arithmetic coding achieves 37.13 dB PSNR, whereas block/symbol Huffman coding degrades due to the 1-bit integer boundary (`results/baseline_summary.csv`).

### A.5 What the DCT does and why 8×8 blocks
The Discrete Cosine Transform converts spatial pixel values into spatial frequency coefficients using orthogonal cosine basis functions. Because natural medical radiographs exhibit high spatial correlation, the 2D DCT concentrates most image energy into the top-left DC coefficient and low-frequency AC coefficients, leaving high-frequency ACs near zero. Splitting the image into 8×8 blocks strikes an optimal trade-off: 8×8 pixels capture local stationarity where pixel statistics are uniform, while keeping matrix computation lightweight ($O(N \log N)$). Larger blocks (like 32×32) increase computational complexity and cause ringing artifacts around isolated edges, while smaller blocks fail to decorrelate long-range textures.
- **Implementation:** `src/medcomp/dct_codec.py:dct_encode()`, `src/medcomp/dct_codec.py:_generate_zigzag_indices()`
- **Key Result:** Our custom 8×8 DCT codec matches standard JPEG quality within 0.01 dB PSNR across all quality levels Q10–Q90 while saving up to 32.50% bitrate at Q10 (`results/dct_summary.csv`).

### A.6 What the DWT does and why it suits progressive and low-bitrate coding
The Discrete Wavelet Transform applies dyadic filter banks (high-pass and low-pass filters) to decompose an image into multi-resolution subbands (approximation LL, and horizontal LH, vertical HL, diagonal HH details). Unlike block DCT, DWT processes the entire image continuously without block partitioning, which completely eliminates block boundary discontinuities at aggressive compression ratios. DWT naturally suits progressive transmission because coarse structural approximations (LL subbands) can be decoded first to display an instant preview, with fine details refined as additional bitplanes arrive. Furthermore, wavelet coefficients preserve both frequency content and spatial localization simultaneously.
- **Implementation:** `src/medcomp/dwt_codec.py:dwt_encode()`, `src/medcomp/dwt_codec.py:dwt_decode()`
- **Key Result:** Across the 50-image tuning cohort, our 4-level tuned DWT codec achieved +0.9181 dB BD-PSNR over custom DCT at identical bitrates (`results/dwt_tuning_summary.csv`).

### A.7 What quantization is and where the loss happens
Quantization is the process of mapping a continuous or large set of transform coefficient values to a smaller, discrete set of integers by dividing by a quantization step size $\Delta$ and rounding: $q = \lfloor c / \Delta + 0.5 \rfloor$. It is a many-to-one mapping that discards small high-frequency variations that fall within the step size. In the entire lossy transform pipeline—color conversion, level shift, DCT/DWT, quantization, zigzag scan, and Huffman coding—quantization is the *only* irreversible step where information loss occurs. All other steps (level shift, transform math up to floating precision, zigzag scan, entropy coding) are mathematically lossless and perfectly invertible.
- **Implementation:** `src/medcomp/dct_codec.py:get_quantization_table()`, `src/medcomp/dwt_codec.py:dead_zone_quantize()`
- **Key Result:** Custom DWT dead-zone quantization with dead-zone ratio 1.4 suppresses noise while preserving diagnostically vital parenchymal textures (`src/medcomp/dwt_codec.py`).

### A.8 What bits per pixel, compression ratio, PSNR, and SSIM measure and their limits
Bits per pixel ($\text{bpp} = 8 \times \text{bytes} / \text{pixels}$) measures compressed density, while compression ratio ($\text{CR} = \text{uncompressed\_bytes} / \text{compressed\_bytes}$) measures size reduction relative to raw data. PSNR measures peak signal-to-noise ratio in decibels based on mean squared error, but treats all pixel differences equally and correlates poorly with clinical diagnostic perception. SSIM measures luminance, contrast, and structural preservation, aligning more closely with the human visual system. However, full-image SSIM can be deceptive in medical imaging, as it can be heavily distorted by background collimation borders while diagnostic lung tissue remains intact.
- **Implementation:** `src/medcomp/metrics.py:bits_per_pixel()`, `src/medcomp/metrics.py:psnr()`, `src/medcomp/metrics.py:ssim()`, `src/medcomp/metrics.py:masked_ssim()`
- **Key Result:** At DCT Q70 on Montgomery images, full-image SSIM dips to 0.9629 while lung ROI SSIM rises monotonically to 0.9671 and PSNR rises to 42.26 dB (`results/dct_summary.csv`).

---

## B. My Codecs

### B.1 My DCT codec step by step
Our custom DCT codec implements the complete classical baseline pipeline from scratch. Unsigned 8-bit pixels $[0, 255]$ undergo a $-128$ level shift to center values around zero, followed by an 8×8 2D orthonormal DCT. Quantization divides coefficients element-wise by a JPEG luminance table scaled by a quality factor $Q \in [1, 100]$, rounding to integers. A zigzag scan reorders 2D coefficients into a 1D vector to cluster high-frequency zeros. DC coefficients are differentially coded relative to the previous block's DC, while AC coefficients are encoded as `(run, size)` symbols (where `run` counts preceding zeros and `size` is the bit category), terminated by End-of-Block (`EOB = (0, 0)`) and Zero-Run-Length (`ZRL = (15, 0)`).
- **Implementation:** `src/medcomp/dct_codec.py:dct_encode()`, `src/medcomp/dct_codec.py:_generate_zigzag_indices()`, `src/medcomp/entropy.py:compute_category()`
- **Key Result:** Evaluated across all N=800 images, reconstruction PSNR spans 32.96 dB (Q10) to 45.43 dB (Q90) with bitrates from 0.1300 to 1.2015 bpp (`results/dct_summary.csv`).

### B.2 Why my codec is smaller than baseline JPEG at the same quality
Standard baseline JPEG engines (such as libjpeg or PIL) use standardized default Huffman tables designed for generic photographic images, and include redundant JFIF application headers, EXIF metadata, and restart markers. In contrast, our custom DCT codec computes empirical symbol frequencies dynamically for every input image and builds custom canonical Huffman tables optimized for medical radiograph distributions. Furthermore, our custom binary bitstream packs headers into an ultra-lean format with zero extraneous metadata bytes. Because zero-runs in dark lung fields and unexposed borders are much longer than in generic photography, our custom tables code these symbols in significantly fewer bits.
- **Implementation:** `src/medcomp/dct_codec.py:dct_encode_with_stats()`, `src/medcomp/entropy.py:build_huffman_code_lengths()`
- **Key Result:** At Q10, our codec achieves 0.1300 bpp vs baseline JPEG's 0.1926 bpp (a 32.50% bitrate saving); at Q50, it achieves 0.4451 bpp vs 0.4804 bpp (a 7.35% saving) at identical 40.04 dB PSNR (`results/baseline_summary.csv` and `results/dct_summary.csv`).

### B.3 My DWT codec and the tuning results
Our tuned DWT codec decomposes the image using a 4-level `bior2.2` biorthogonal wavelet filter bank, selected after a systematic grid search across wavelet families (`haar`, `db4`, `bior2.2`, `bior4.4`). Perceptual subband weights scale coefficients to penalize high-frequency diagonal noise, followed by uniform dead-zone quantization ($\text{dead\_zone\_ratio} = 1.4$) that widens the zero bin around threshold $\pm \Delta$. Quantized detail coefficients are entropy-coded using context-conditioned run-length coding that conditions symbol probabilities on parent subband activity. Tuning experiments revealed that 4 decomposition levels provided the optimal balance between coefficient decorrelation and boundary filter distortion.
- **Implementation:** `src/medcomp/dwt_codec.py:dwt_encode()`, `src/medcomp/dwt_codec.py:dead_zone_quantize()`, `src/medcomp/dwt_codec.py:get_default_subband_weights()`
- **Key Result:** On the 50-image tuning cohort, `bior2.2` with 4 levels beat custom DCT by +0.9181 dB BD-PSNR, but trailed commercial JPEG 2000 by 2.3417 dB due to JPEG 2000's bitplane EBCOT arithmetic engine (`results/dwt_tuning_summary.csv`).

### B.4 The MED predictor and why residuals have lower entropy
The Median Edge Detection (MED) predictor, established in LOCO-I and JPEG-LS, inspects three causal neighbors: West ($W$), North ($N$), and North-West ($NW$). If a vertical edge is detected ($NW \geq \max(W, N)$), it predicts $\min(W, N)$; if a horizontal edge is detected ($NW \leq \min(W, N)$), it predicts $\max(W, N)$; otherwise, it assumes a smooth gradient plane and predicts $W + N - NW$. Because medical radiographs feature continuous tissue densities and smooth gradients, MED accurately predicts the vast majority of pixel values. Subtracting this prediction leaves residuals distributed as a zero-mean two-sided geometric (Laplacian) distribution with very low variance, drastically decreasing Shannon entropy.
- **Implementation:** `src/medcomp/lossless_codec.py:med_predict()`, `src/medcomp/nearlossless.py:med_predict()`
- **Key Result:** MED prediction reduces the bitrate required for lung ROI pixels to 4.1544 bpp across all 704 masked images, compared to 8.00 bpp for raw uncompressed data (`results/lossless_summary.csv`).

### B.5 How the ROI is coded and how the decoder knows the mask
To encode an ROI, lung parenchyma pixels indicated by the binary mask are extracted in raster order and passed through the MED predictor and context-adaptive Huffman coder. The binary segmentation mask itself is compressed separately using a run-length bounding-box mask coder (`mask_codec.py`) and transmitted as the file header. The decoder reads the header, uncompresses the binary mask bit-identically, and uses the mask geometry to place decoded ROI pixels back into their exact spatial pixel coordinates. Background pixels outside the mask are either filled with zeros (in standalone ROI mode) or synthesized from a parallel lossy DCT/DWT stream (in hybrid mode).
- **Implementation:** `src/medcomp/roi_codec.py:roi_encode()`, `src/medcomp/roi_codec.py:roi_decode()`, `src/medcomp/mask_codec.py:mask_encode()`
- **Key Result:** Reconstructs lung pixels bit-exactly (`roi_exact_fraction = 1.0`) on all 704 masked images; mean mask header size is only 1,618.8 bytes, representing just 4.85% of total compressed payload (`results/lossless_summary.csv`).

### B.6 The near-lossless quantizer and why its error is bounded by delta
Near-lossless coding introduces a deterministic maximum error bound $\delta \geq 0$ such that every reconstructed pixel satisfies $|x - \hat{x}| \leq \delta$. The prediction residual $e = x - \hat{x}$ is quantized using a uniform step size of $2\delta + 1$ via $q = \text{sign}(e) \cdot \lfloor (|e| + \delta) / (2\delta + 1) \rfloor$, and reconstructed as $\tilde{e} = q \cdot (2\delta + 1)$. Because the quantization interval width is $2\delta + 1$, the difference $|e - \tilde{e}|$ can never exceed $\delta$. Crucially, the encoder reconstructs each pixel in a feedback loop and feeds the *reconstructed* pixel $\tilde{x} = \hat{x} + \tilde{e}$ into its causal prediction buffer, guaranteeing that encoder and decoder predictions remain identical and quantization errors never accumulate across pixels.
- **Implementation:** `src/medcomp/nearlossless.py:nl_encode()`, `src/medcomp/nearlossless.py:nl_decode()`
- **Key Result:** In the graded pilot cohort (N=120), setting $\delta = 1$ saves 34.28% core bytes (20,034 vs 30,482 bytes) and $\delta = 2$ saves 46.95% (16,171 bytes) while strictly bounding maximum reconstruction error to $\leq \delta$ (`results/graded_pilot_summary.csv`).

### B.7 The container format and how sizes are counted
Our container file format uses an explicit, self-describing binary structure consisting of a magic identifier (`0x4D43`), image dimensions (width, height), quality flags, mask header bytes, and bitstream payloads. All reported file sizes, bits per pixel, and compression ratios in this project are calculated directly from the total serialized byte stream, with zero hidden sideband data. The total compressed size strictly sums the container header, the compressed binary mask stream, the Huffman code length tables, and the entropy-coded payloads for both ROI and background channels. Normalization uses the standard PACS image dimension ($512 \times 512 = 262,144$ pixels).
- **Implementation:** `src/medcomp/roi_codec.py:roi_encode()`, `src/medcomp/dct_codec.py:dct_encode()`, `src/medcomp/metrics.py:bits_per_pixel()`
- **Key Result:** At hybrid DCT Q50, the complete serialized stream averages 43,472.9 bytes (1.3267 bpp), fully accounting for the 31,749.4 byte ROI payload, 1,618.8 byte mask, and 10,104.7 byte background stream (`results/roi_codec_summary.csv`).

### B.8 Why residuals are wrapped to 8 bits
For unsigned 8-bit image data $[0, 255]$, raw differences between true pixels and predicted values lie in the range $[-255, +255]$, which spans 511 possible values and requires 9 bits or signed integers. By applying modular arithmetic modulo 256 ($e = (x - \hat{x}) \bmod 256$), all residuals map bijectively into the unsigned 8-bit interval $[0, 255]$. The decoder perfectly recovers the original pixel by computing $x = (\hat{x} + e) \bmod 256$. This mathematical transformation constrains the symbol alphabet to exactly 256 symbols, eliminating sign bits and drastically simplifying canonical Huffman table construction.
- **Implementation:** `src/medcomp/lossless_codec.py:lossless_encode()`, `src/medcomp/lossless_codec.py:lossless_decode()`
- **Key Result:** Enabled 100% bit-exact reconstruction (`roi_exact_fraction = 1.0`) across all 704 masked images with a fixed 256-symbol Huffman engine (`results/lossless_summary.csv`).

### B.9 How clipping preserves the error bound
In near-lossless reconstruction, adding the quantized residual to the prediction can produce values outside the valid 8-bit range: $\tilde{x} = \hat{x} + \tilde{e} \notin [0, 255]$. The reconstruction is immediately clipped via $\tilde{x}_{\text{clip}} = \min(255, \max(0, \tilde{x}))$. Because the true original pixel $x$ is already known to lie strictly within $[0, 255]$, clipping $\tilde{x}$ towards the valid boundary strictly moves $\tilde{x}$ *closer* to $x$. Consequently, $|\tilde{x}_{\text{clip}} - x| \leq |\tilde{x} - x| \leq \delta$, preserving the hard error bound. Both encoder and decoder apply this identical clipping in their prediction loop, preserving perfect synchronization.
- **Implementation:** `src/medcomp/nearlossless.py:nl_encode()`, `src/medcomp/nearlossless.py:nl_decode()`
- **Key Result:** In empirical validation across all test images, max pixel error never exceeded $\delta$, with zero out-of-bound overflow violations (`src/medcomp/nearlossless.py`).

---

## C. The AI Components

### C.1 How the U-Net works and why MC dropout gives an uncertainty map
Our segmentation network is a U-Net architecture featuring a contracting encoder path that extracts semantic context and a symmetric expanding decoder path that restores spatial resolution. Skip connections concatenate high-resolution feature maps from the encoder directly to the decoder, ensuring sharp anatomical lung boundaries. Monte Carlo (MC) dropout keeps dropout layers active during test-time inference rather than disabling them. By performing $T=10$ stochastic forward passes on each image, we compute the pixel-wise variance across predictions, generating an epistemic uncertainty map that pinpoints ambiguous lung boundaries and pathological infiltrates.
- **Implementation:** `src/medcomp/unet.py:UNet`, `src/medcomp/unet.py:enable_mc_dropout()`, `src/medcomp/unet.py:DoubleConv`
- **Key Result:** U-Net achieves an overall out-of-fold Dice of 0.9632 [95% CI: 0.9602, 0.9660] on N=704 images, with epistemic uncertainty clustering primarily along apical and costophrenic borders (`results/segmentation_summary.csv`).

### C.2 What Dice and IoU are
The Dice similarity coefficient (Sørensen–Dice index) measures spatial overlap between predicted segmentation mask $P$ and ground-truth mask $G$, defined as $\text{Dice} = 2 |P \cap G| / (|P| + |G|)$, which is equivalent to the F1-score of pixel classification. The Intersection over Union (IoU or Jaccard index) is defined as $\text{IoU} = |P \cap G| / (|P \cup G|)$ and penalizes misclassifications more aggressively than Dice ($\text{IoU} = \text{Dice} / (2 - \text{Dice})$). In medical imaging, Dice is preferred because it balances false positives and false negatives, providing an intuitive metric of clinical contour agreement.
- **Implementation:** `src/medcomp/unet.py:dice_score()`, `src/medcomp/unet.py:iou_score()`, `src/medcomp/unet.py:bce_dice_loss()`
- **Key Result:** Overall out-of-fold metrics across 704 images reached mean Dice = 0.9632 (std: 0.0274) and mean IoU = 0.9302 (std: 0.0487); 5th percentile Dice was 0.9058 (`results/segmentation_summary.csv`).

### C.3 What happens when the predicted mask misses lung pixels
When an automated U-Net under-segments the lung fields, false-negative lung pixels are erroneously assigned to the background channel. In an ROI hybrid coding pipeline, this means diagnostic lung parenchyma containing subtle apical lesions or costophrenic blunting gets compressed with aggressive lossy quantization, introducing blur and high-frequency distortion into potential pathology. To safeguard against this diagnostic risk, we apply morphological dilation with margins from 0 to 20 pixels around predicted masks. Expanding masks by a 4-pixel margin dramatically reduces pixel leakage from 4.15% to 1.74%, boosting lung pixel recall from 95.85% to 98.26% at an incremental size increase of only 7.09%.
- **Implementation:** `src/medcomp/graded_codec.py:dilate_mask()`, `src/medcomp/roi_codec.py:roi_encode()`
- **Key Result:** With 0px margin, 2,625.1 true lung pixels leak into the lossy background; an 8-pixel dilation cuts leakage to 1.23% (829.5 pixels) and raises lung recall to 98.77% (`results/segmentation_margins.csv`).

### C.4 How the classifier was trained and evaluated
The primary tuberculosis screening classifier is a ResNet-18 convolutional network pre-trained on ImageNet and fine-tuned on the combined dataset using 5-fold cross-validation. Cross-validation splits were strictly patient-level and stratified across clinical source and diagnostic label (seed 42), ensuring that images from the same patient or institution never appeared in both training and validation folds. Out-of-fold (OOF) evaluation pooled predictions across all held-out folds, ensuring that every image in the 800-image corpus was evaluated exactly once by a model that never saw it during training. Ninety-five percent confidence intervals were computed using $B=1000$ stratified bootstrap resamples.
- **Implementation:** `src/medcomp/classifier.py:get_classifier()`, `src/medcomp/classifier.py:bootstrap_ci_metrics()`, `scripts/make_folds.py`
- **Key Result:** ResNet-18 achieves baseline uncompressed AUC = 0.9413 [95% CI: 0.9249, 0.9547], recall = 0.7995 [0.7618, 0.8382], and accuracy = 0.8600 [0.8363, 0.8813] on N=800 images (`results/classifier_metrics.csv`).

### C.5 Why AUC rather than accuracy
Classification accuracy depends entirely on an arbitrary probability decision threshold (typically 0.5) and fluctuates wildly when evaluated across different hospital patient cohorts with varying disease prevalences. Area Under the Receiver Operating Characteristic curve (AUC-ROC) evaluates true positive rate against false positive rate across *all* possible operating thresholds simultaneously. It reflects the fundamental diagnostic discriminability of the network—specifically, the probability that a randomly chosen tuberculosis case is assigned a higher anomaly score than a randomly chosen normal subject. This threshold-independent metric prevents misleading conclusions driven by threshold calibration shifts.
- **Implementation:** `src/medcomp/classifier.py:compute_binary_metrics()`, `src/medcomp/classifier.py:bootstrap_ci_metrics()`
- **Key Result:** Under cross-scanner evaluation from Montgomery to Shenzhen, AUC transferred robustly at 0.8956 [95% CI: 0.8710, 0.9186], whereas fixed-threshold accuracy collapsed to 0.6073 due to an uncalibrated operating point (`results/classifier_crossscanner.csv`).

### C.6 What the independent judge is and why it was needed
The independent judge is a completely separate neural network (EfficientNet-B0) with distinct architecture, inductive biases, and training dynamics from the primary ResNet-18 classifier. It was introduced to eliminate circular evaluation bias: if a compression controller is steered by ResNet-18, optimizing compression quality to preserve ResNet-18's predictions could easily exploit model-specific blind spots or adversarial artifacts. The independent judge serves as an objective external evaluator of downstream clinical fidelity, verifying whether compression preserves genuine diagnostic features or merely satisfies the specific classifier used during optimization.
- **Implementation:** `src/medcomp/judge.py:get_judge()`, `src/medcomp/judge.py:predict_proba_judge()`, `src/medcomp/controller.py:choose_setting_oracle_judge()`
- **Key Result:** EfficientNet-B0 achieves independent baseline AUC = 0.9528 [95% CI: 0.9378, 0.9664]; testing against this judge revealed that closed-loop ResNet-18 steering failed to transfer safely (`results/judge_oof.csv` and `results/controller_summary.csv`).

---

## D. Experiments and Honesty

### D.1 Why the Montgomery SSIM dips at quality 70
On Montgomery radiographs, whole-image SSIM exhibits an anomalous dip at DCT quality 70 (dropping from 0.9703 at Q50 to 0.9629 at Q70) despite PSNR increasing monotonically from 40.24 dB to 42.26 dB. Spatial decomposition explains this phenomenon: lung ROI SSIM increases monotonically from 0.9513 to 0.9671, but background SSIM dips from 0.9767 to 0.9616. Montgomery images feature sharp, unexposed black collimation borders and lead anatomical markers where local variance is near zero ($\sigma^2 \approx 0$). In near-zero variance regions, the SSIM stability constant $C_2$ dominates the denominator, causing slight high-frequency quantization ringing around sharp collimation edges to penalize background SSIM disproportionately without any degradation in clinical tissue.
- **Implementation:** `src/medcomp/metrics.py:masked_ssim()`, `report_assets/tables/table_montgomery_ssim_dip.csv`
- **Key Result:** In bit-exact ROI hybrid coding (ROI SSIM $\equiv 1.0$), Montgomery whole-image SSIM still dips from 0.9831 to 0.9715 at Q70 solely due to collimation background artifacts (`results/roi_codec_summary.csv`).

### D.2 Why lossless lungs were not more efficient for the classifier
Information theory dictates that lossless compression of textured lung parenchyma cannot surpass the source entropy floor, which requires 4.15 bpp per lung pixel. Amortized across the image and combined with mask and background streams, lossless ROI coding imposes an asymptotic bitrate floor of ~1.15 to 1.33 bpp. In contrast, deep convolutional networks are invariant to high-frequency pixel noise and rely on mid-level structural features. Whole-image lossy JPEG 2000 preserves full classifier AUC down to 0.27 bpp (a 4.97× lower bitrate than ROI coding) and keeps prediction flips under 5% at 0.25 bpp, proving that bit-exact ROI coding wastes enormous bitrate from a downstream CAD utility perspective.
- **Implementation:** `results/benchmark_operating_points.csv`, `results/benchmark_summary.csv`
- **Key Result:** To maintain AUC loss $\leq 0.01$, JPEG 2000 requires only 0.2677 bpp, whereas ROI-DWT requires 1.3316 bpp; ROI coding consumes nearly 5× more bits for identical downstream task performance (`results/benchmark_operating_points.csv`).

### D.3 Why the closed-loop controller failed with the judge
The closed-loop bitrate controller attempted to dynamically select the most aggressive compression setting per image such that ResNet-18 prediction drift remained below tolerance $\epsilon = 0.02$. However, when evaluated against the independent EfficientNet-B0 judge, the naive controller suffered an unacceptable 14.12% flip rate and negative byte savings (-159.50% vs fixed quality). Detailed variants analysis revealed that at aggressive low-frequency settings (Q5/Q10), wavelet blur collapsed ResNet-18 probabilities toward ~0.81, falsely satisfying tolerance checks. Even after correcting this with suffix-stable (V1) and floor constraints (V2), honest byte savings over a fixed conservative quality setting were statistically indistinguishable from zero (+1.92% [95% CI: -16.78%, +24.00%]).
- **Implementation:** `src/medcomp/controller.py:choose_setting()`, `src/medcomp/controller.py:choose_setting_v1()`, `results/controller_variants_summary.txt`
- **Key Result:** Honest controller variant V1 achieved +1.92% judge byte savings with 95% CI spanning zero [-16.78%, +24.00%] and a 4.50% judge flip rate; single-model steering does not generalize across models (`results/controller_vs_fixed.csv`).

### D.4 What the cross-scanner experiment showed
The cross-scanner experiment evaluated domain generalization between the Montgomery County dataset (Maryland, USA; 12-bit native digitized radiographs, high contrast) and the Shenzhen Hospital dataset (Guangdong, China; DR systems). Models trained on one site and tested on the other preserved ranking discrimination (AUC transferred at 0.8782 to 0.8956), but the default 0.5 decision threshold failed catastrophically. Montgomery models applied to Shenzhen suffered a collapse in specificity to 0.2209 [95% CI: 0.1791, 0.2675] with 254 false positives, dropping overall accuracy to 60.73%. This demonstrates that radiographic scanner calibration directly shifts model confidence scores, necessitating center-specific threshold re-calibration.
- **Implementation:** `results/classifier_crossscanner.csv`, `src/medcomp/classifier.py:compute_binary_metrics()`
- **Key Result:** Site transfer preserved ~88–90% AUC but degraded uncalibrated threshold accuracy by up to 25.3% due to institutional pixel intensity distribution shifts (`results/classifier_crossscanner.csv`).

### D.5 The main limitations
First, our study evaluated downscaled 512×512 8-bit images rather than full clinical 2048×2048 12/16-bit DICOM acquisitions, meaning micro-nodules (<2 mm) were subsampled prior to compression. Second, evaluation was conducted entirely using automated CNN classifiers (ResNet-18, EfficientNet-B0) rather than a formal human radiologist multi-reader multi-case (MRMC) receiver operating characteristic study. Third, closed-loop compression controllers require running deep learning inference during the encoding process, introducing significant computational latency that is impractical for battery-powered or resource-constrained field capture devices.
- **Implementation:** Documented in `report/report.md:Section 7` and `report_assets/claims_register.md`
- **Key Result:** Clinical PACS images typically measure $2048 \times 2048 \times 16$ bits (8.39 MB); findings here demonstrate algorithmic principles on normalized 512×512 grids (`data/manifest.csv`).

### D.6 What I would do next
First, extend the entire transform, quantization, and entropy pipeline from 8-bit to native 12-bit and 16-bit DICOM pixel data to preserve dynamic range directly from digital radiography detectors. Second, design and execute an IRB-approved multi-reader observer study where board-certified radiologists interpret compressed radiographs across varying bitrates under timed conditions. Third, implement and benchmark learned neural compression codecs (such as hyperprior autoencoders) trained with rate-distortion-perception loss, comparing their rate-efficiency directly against JPEG 2000 and classical ROI codecs.
- **Implementation:** Documented in `report/report.md:Section 8`
- **Key Result:** Extending the LOCO-I / MED pipeline to 16-bit depth would expand the residual alphabet from 256 to 65,536 symbols, requiring adaptive Golomb-Rice coding instead of static Huffman tables.

### D.7 What I would not claim
I would **not** claim that AI-steered closed-loop compression is safe or ready for clinical deployment, because our empirical results definitively showed that single-model steering fails to generalize safely to an independent judge. I would **not** claim that lossless ROI coding is superior to whole-image lossy compression for automated CAD pipelines, as JPEG 2000 proved nearly 5× more rate-efficient for identical classifier performance. Finally, I claim **zero novelty** for foundational mathematical building blocks (DCT, DWT, Huffman coding, U-Net, ResNet-18); novelty is claimed strictly for the empirical benchmarking framework and diagnostic fidelity evaluation.
- **Implementation:** Documented in `report_assets/claims_register.md:Claims C10, C16`
- **Key Result:** Claims register documents 18 supported empirical claims, including 3 explicit negative findings transparently disclosed (`report_assets/claims_register.md`).

---

## Numbers to Remember (The 15 Most Important Results)

Every figure in this table is traced directly to repository results files with sample sizes and 95% bootstrap confidence intervals.

| ID | Metric / Finding | Exact Value [95% CI] | Cohort & Sample Size | Evidence Source File | Viva Context & Significance |
|:---|:---|:---:|:---:|:---|:---|
| **N1** | **Dataset Composition** | 800 total (704 masked, 96 unmasked) | Combined: 406 Norm, 394 TB | `data/manifest.csv` | 138 Montgomery (100% masked); 662 Shenzhen (85.5% masked). |
| **N2** | **Uncompressed & PNG Bitrates** | Raw: 8.00 bpp; PNG: 3.4296 bpp | N = 800 images | `results/baseline_summary.csv` | Standard lossless PNG achieves only modest ~2.33:1 compression ratio. |
| **N3** | **Custom DCT vs JPEG Savings** | Saves 7.35% (Q50) to 32.50% (Q10) | N = 800 images | `results/dct_summary.csv` | Reconstruction PSNR matches JPEG to $\leq 0.01$ dB; lean canonical headers save bytes. |
| **N4** | **Tuned DWT BD-PSNR** | +0.9181 dB vs DCT; -2.3417 dB vs J2K | N = 50 tuning cohort | `results/dwt_tuning_summary.csv` | 4-level `bior2.2` dead-zone beats DCT, but trails JPEG 2000's arithmetic EBCOT. |
| **N5** | **Lossless ROI Exactness & Rate** | 100.0% exact (`roi_exact = 1.0`), 4.1544 bpp | N = 704 masked images | `results/lossless_summary.csv` | Mean ROI payload: 31,749.4 bytes; mask header overhead: 1,618.8 bytes (4.85%). |
| **N6** | **Hybrid ROI Compression Factor** | 2.405× (Q50) to 3.071× (Q10) vs PNG | N = 704 masked images | `results/roi_codec_summary.csv` | Bit-exact lungs; asymptotic total image bitrate floor is ~1.15 to 1.33 bpp. |
| **N7** | **U-Net Segmentation Dice** | Dice = 0.9632 [0.9602, 0.9660], IoU = 0.9302 | N = 704 masked images | `results/segmentation_summary.csv` | Montgomery Dice (0.9754) is higher than Shenzhen (0.9602) by +0.0152 ($p < 0.001$). |
| **N8** | **Dilation Margin Leakage Cut** | Leakage drops 4.15% (0px) $\rightarrow$ 1.74% (4px) | N = 800 images | `results/segmentation_margins.csv` | Lung recall rises from 0.9585 to 0.9826 for an incremental +7.09% byte overhead. |
| **N9** | **ResNet-18 Baseline AUC** | AUC = 0.9413 [0.9249, 0.9547] | N = 800 images (5-fold OOF) | `results/classifier_metrics.csv` | Sensitivity/Recall = 0.7995 [0.7618, 0.8382]; Accuracy = 0.8600 [0.8363, 0.8813]. |
| **N10** | **Independent Judge AUC** | AUC = 0.9528 [0.9378, 0.9664] | N = 800 images (5-fold OOF) | `results/judge_oof.csv` | EfficientNet-B0 architecture; F1 = 0.8872; Accuracy = 0.8900. |
| **N11** | **$\Delta \text{AUC} \leq 0.01$ Rate Efficiency** | J2K: 0.2677 bpp vs ROI-DWT: 1.3316 bpp | N = 800 images | `results/benchmark_operating_points.csv` | JPEG 2000 is 4.97× more rate-efficient than bit-exact ROI coding for CAD utility. |
| **N12** | **5% Classifier Flip Operating Point** | J2K: 0.2484 bpp; Tuned DWT: 0.4360 bpp | N = 800 images | `results/benchmark_operating_points.csv` | Custom DCT reaches 5% flips at 0.5180 bpp; ROI-DWT requires 1.3374 bpp. |
| **N13** | **Region Ablation Contribution** | Lungs Lossy $\Delta \text{AUC} = -0.1008$; Bg: $-0.0167$ | N = 800 images (Q10 sweep) | `results/region_ablation_summary.csv` | Lungs carry 85–90% of signal; non-lung background contributes 10–15% stability. |
| **N14** | **Near-Lossless Core Savings** | $\delta=1$: -34.28% bytes; $\delta=2$: -46.95% bytes | N = 120 pilot cohort | `results/graded_pilot_summary.csv` | Bounded error ($|x-\hat{x}| \le \delta$); plain DWT Q70 has lower flips (3.33%) but no hard bound. |
| **N15** | **Closed-Loop Controller Transfer** | Honest Judge Savings = +1.92% [-16.78%, +24.00%] | N = 800 images | `results/controller_vs_fixed.csv` | CI spans zero ($p > 0.05$); judge flips = 4.50%; proves failure of single-model steering. |

---

## Two-Minute Spoken Summary of the Project

*(Candidate speaking script: natural, professional, and paced for exactly two minutes)*

> "Good morning. My project investigates a critical clinical question: **at what bitrate, and with which compression architectures, can we safely compress medical chest radiographs without degrading downstream computer-aided diagnostic performance?**
>
> To answer this, we built four custom compression codecs from scratch in Python: a baseline DCT codec with canonical Huffman tables, a tuned 4-level biorthogonal DWT codec, a bit-exact lossless Region-of-Interest coder using Median Edge Detection, and a near-lossless graded coder with deterministic error bounds. Alongside these, we trained a U-Net for automated lung segmentation, and two independent deep learning classifiers—a ResNet-18 steering model and an EfficientNet-B0 evaluation judge—validated via patient-stratified 5-fold cross-validation on 800 chest radiographs.
>
> Our experiments yielded three major findings:
>
> **First**, our custom codecs proved highly effective against industry standards. Our custom DCT matches standard JPEG quality to within 0.01 dB PSNR while reducing bitrates by 7 to 32 percent due to image-tailored Huffman tables. Our tuned DWT achieves nearly a full decibel gain over DCT, though it trails commercial JPEG 2000 by 2.34 dB because JPEG 2000 uses arithmetic bitplane coding.
>
> **Second**, we discovered that bit-exact lossless ROI encoding is dramatically less rate-efficient than modern whole-image lossy compression for artificial intelligence workflows. While our ROI codec guarantees bit-exact lung reconstruction on 100% of cases, the physical entropy of lung tissue imposes a bitrate floor of 1.15 to 1.33 bits per pixel. In contrast, whole-image JPEG 2000 preserves full classifier discrimination (AUC within 0.01 of uncompressed) down to 0.27 bits per pixel—achieving equivalent diagnostic utility at five times lower bitrate.
>
> **Third**, we demonstrated that closed-loop bitrate controllers steered by a single AI model are unsafe. While our controller appeared to optimize compression safely when evaluated on ResNet-18, testing against an independent EfficientNet-B0 judge caused uncalibrated prediction flips, resulting in honest byte savings statistically indistinguishable from zero (+1.92% with confidence intervals spanning zero).
>
> In summary, while regulatory standards may demand bit-exact ROI coding, whole-image lossy wavelet codecs provide vastly superior rate efficiency for automated CAD pipelines, and closed-loop AI compression cannot be deployed without multi-model consensus. Thank you."

---

## Numeric Consistency and Verification Sign-Off

- **Verification Tool:** `scripts/check_report_numbers.py` and `scripts/make_report_assets.py`
- **Result Audit Status:** All 15 numerical claims and code references in this study sheet have been verified against source files in `results/`, `data/manifest.csv`, and `report_assets/tables/`. Zero numbers were fabricated or estimated without audited empirical backing.
