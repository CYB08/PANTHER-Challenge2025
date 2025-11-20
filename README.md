# Pancreatic Tumor Segmentation

A deep learning framework for automatic pancreatic tumor segmentation from MRI scans using semi-supervised learning with the BLUNet architecture integrated into nnUNet.

## 📋 Overview

This project implements a semi-supervised learning pipeline for pancreatic tumor segmentation that:
- Uses **BLUNet** as the backbone architecture
- Generates high-quality pseudo-labels from unlabeled MRI data
- Iteratively improves model performance by selecting reliable predictions
- Integrates seamlessly with the nnUNet framework

### Key Features

- **Advanced Architecture**: BLUNet with hierarchical gated convolutions for improved boundary detection
- **Semi-Supervised Learning**: Leverages both labeled and unlabeled MRI data
- **Quality-Based Selection**: Automatic filtering of high-confidence pseudo-labels
- **ROI-Based Processing**: Pancreas-focused cropping for efficient training
- **nnUNet Integration**: Built on the robust nnUNet framework

## 🏗️ Architecture

### BLUNet

```
Input MRI Scan
      ↓
┌─────────────────────────────────────────────┐
│  Encoder (with Residual + hgcnblocks)      │
│  - Stem convolution                         │
│  - Stage 0-1: Residual blocks               │
│  - Stage 2-5: Residual + hgcnblocks        │
│  - Progressive downsampling                 │
└─────────────────────────────────────────────┘
      ↓ (skip connections)
┌─────────────────────────────────────────────┐
│  Decoder (with Residual + hgcnblocks)      │
│  - Progressive upsampling                   │
│  - Skip connection fusion                   │
│  - Stage-matched hgcnblocks                │
│  - Segmentation heads (deep supervision)    │
└─────────────────────────────────────────────┘
      ↓
Tumor Segmentation Output
```

**Key Components**:
- **hgConv**: High-order gated convolution for multi-scale feature extraction
- **hgcnblock**: Hierarchical gated convolutional block with residual connections
- **Layer Normalization**: Spatial normalization for stable training

## 📁 Project Structure

```
TASK1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data_utils.py                      # Data preprocessing utilities
│
├── training/                          # Semi-supervised training pipeline
│   ├── network.py                     # BLUNet architecture definition
│   ├── filter.py                      # Pseudo-label generation
│   └── evaluate.py                    # Pseudo-label quality evaluation
│
├── trainer/                           # nnUNet trainer implementations
│   └── nnUNetTrainerBLUNet.py        # Custom BLUNet trainer
│
├── MRSegmentator/                     # MRI preprocessing tools https://github.com/hhaentze/MRSegmentator.git
├── nnUNet_results/                    # Trained model checkpoints
└── Dockerfile                         # Docker container configuration
```

## Installation

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd TASK1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install nnUNet
cd nnUNet
pip install -e .
cd ..

# Set up environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

## Dataset Preparation

### Expected Data Structure

```
nnUNet_raw/
└── Dataset<ID>_PancreasTumor/
    ├── imagesTr/              # Training images
    │   ├── case001_0000.nii.gz
    │   ├── case002_0000.nii.gz
    │   └── ...
    ├── labelsTr/              # Training labels
    │   ├── case001.nii.gz
    │   ├── case002.nii.gz
    │   └── ...
    ├── imagesTs/              # Test images (unlabeled)
    │   └── ...
    └── dataset.json           # Dataset configuration
```

### Data Preprocessing

```python
import SimpleITK as sitk
from data_utils import resample_img, CropPancreasROI

# Load MRI image
image = sitk.ReadImage("case001_0000.nii.gz")

# Resample to target spacing (e.g., 3.0x3.0x6.0 mm)
resampled = resample_img(image, out_spacing=[3.0, 3.0, 6.0], is_label=False)

# Crop pancreas ROI (requires pancreas mask)
pancreas_mask = sitk.ReadImage("pancreas_mask.nii.gz")
cropped, coords = CropPancreasROI(resampled, pancreas_mask, margins=(10, 10, 10))

# Save preprocessed image
sitk.WriteImage(cropped, "case001_preprocessed.nii.gz")
```

## 🔄 Semi-Supervised Learning Pipeline

### Stage 1: Pseudo-Label Generation

```bash
# Filter domain adaptation weights from trained model
python training/filter.py
```

### Stage 2: Quality Evaluation & Selection

```bash
# Evaluate pseudo-label quality
python training/evaluate.py
```

### Stage 3: Model Retraining

```bash
# Train BLUNet with selected pseudo-labels
nnUNetv2_train DATASET_ID 3d_fullres 0 \
    -tr nnUNetTrainerBLUNet \
    --npz  # Use with additional pseudo-labeled data
```

## 🎯 Model Training

### Training

```bash
# Plan and preprocess
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# Train BLUNet
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerBLUNet
```

## Inference

### Single Case Prediction

```python
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Initialize predictor
predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_device=True,
    device='cuda',
    verbose=False
)

# Load model
predictor.initialize_from_trained_model_folder(
    "path/to/model/folder",
    use_folds=('0',),
    checkpoint_name='checkpoint_final.pth'
)

# Run prediction
predictor.predict_from_files(
    "path/to/input",
    "path/to/output",
    save_probabilities=False,
    overwrite=True
)
```

### Batch Prediction

```bash
# Use nnUNet command-line tool
nnUNetv2_predict \
    -i INPUT_FOLDER \
    -o OUTPUT_FOLDER \
    -d DATASET_ID \
    -c 3d_fullres \
    -tr nnUNetTrainerBLUNet \
    -f 0 \
    --disable_tta  # Disable test-time augmentation for speed
```

### Pancreas ROI Cropping

```python
from data_utils import CropPancreasROI, restore_to_full_size

# Crop pancreas region with margins
cropped_image, crop_coords = CropPancreasROI(
    image=original_image,
    low_res_segmentation=pancreas_mask,
    margins=(10, 10, 10)  # Margins in mm
)

# After prediction, restore to full size
full_size_prediction = restore_to_full_size(
    cropped_mask=predicted_mask,
    original_image=original_image,
    crop_coordinates=crop_coords
)
```
