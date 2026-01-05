# Retinal Vessel Segmentation and Feature Extraction

This repository contains the official implementation of the segmentation model and feature extraction pipeline used in the study "Precision Oculomics".

## Contents

1.  **`model_plain_deep_sup.py`**: The segmentation model architecture (UNet with Encoder and Deep Supervision).
2.  **`calculate_retinal_features.py`**: A unified script for extracting retinal vascular features (CRAE, CRVE, Tortuosity, Fractal Dimension, Branching Angles) using standard methodology and PVBM for optic disc detection.

## Requirements

```bash
pip install torch torchvision timm segmentation-models-pytorch
pip install opencv-python scikit-image scipy pandas tqdm
pip install PVBM  # Required for Optic Disc detection
```

## Usage

### 1. Segmentation Model

The model uses a encoder with a plain UNet decoder and deep supervision.

```python
import torch
from model_plain_deep_sup import create_model_plain_deep_sup

# Initialize model
# num_classes=2 for Artery/Vein segmentation
model = create_model_plain_deep_sup(model_size="base", num_classes=2, pretrained=False)

# Input tensor: [Batch, Channels, Height, Width]
x = torch.randn(1, 3, 512, 512)
output = model(x)

print(output.shape) # torch.Size([1, 2, 512, 512])
```

### 2. Feature Extraction

The script calculates vascular features from binary segmentation masks of arteries and veins.

**Input Directory Structure:**
The script expects a base directory containing two subfolders:
*   `artery_bin/`: Contains binary masks for arteries.
*   `vein_bin/`: Contains binary masks for veins.
*   `summary.csv` (Optional): Contains metadata about original images (crop coordinates) for accurate optic disc detection.

**Command:**

```bash
python calculate_all_retinal_features.py \
    --base_dir /path/to/segmentations \
    --output results.csv
```

**Arguments:**
*   `--base_dir`: Path to the directory containing `artery_bin` and `vein_bin` folders.
*   `--output`: Path to save the resulting CSV file.
*   `--no_pvbm`: (Optional) Disable PVBM-based optic disc detection and use fallback method.
*   `--swap_av`: (Optional) Swap artery and vein labels if necessary.

## Features Calculated

*   **Caliber:** CRAE (Central Retinal Arteriolar Equivalent), CRVE (Central Retinal Venular Equivalent), AVR (Artery-Vein Ratio).
*   **Geometry:** Simple Tortuosity, Branching Angles.
*   **Complexity:** Fractal Dimension (Box-counting).
