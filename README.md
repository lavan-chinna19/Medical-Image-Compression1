# Medical Image Compression with Lossless ROI Preservation

A modular Python framework for medical image compression using Discrete Cosine Transform (DCT), Discrete Wavelet Transform (DWT), and Entropy Coding with Lossless Region-of-Interest (ROI) preservation. Designed for chest X-ray datasets (Montgomery County and Shenzhen sets) and later extensible to lung segmentation, TB classification, and diagnosis-aware compression.

## Project Structure

```
medical/
├── src/
│   └── medcomp/
│       ├── __init__.py
│       ├── config.py       # Dataset paths, directory constants, defaults
│       ├── io_utils.py     # Image I/O, padding, cropping, DICOM support
│       └── metrics.py      # Distortion & rate metrics (MSE, PSNR, SSIM, BPP, CR, Entropy)
├── scripts/                # Analysis, preprocessing, and pipeline scripts
├── tests/                  # Unit and integration test suite
├── work/                   # Scratch / intermediate outputs (gitignored)
├── results/                # Inspection reports, metrics tables, figures
├── requirements.txt        # Core Python dependencies
├── pytest.ini              # Pytest configuration
└── README.md
```

## Setup & Virtual Environment

1. Create and activate a Python 3.10+ virtual environment:
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run test suite:
   ```powershell
   pytest
   ```
