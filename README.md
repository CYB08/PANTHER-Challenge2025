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

### BLUNet (Bi-Level U-Net)

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
│
├── MRSegmentator/                     # MRI preprocessing tools
├── nnUNet_results/                    # Trained model checkpoints
└── Dockerfile                         # Docker container configuration
```

## 🚀 Installation

### Prerequisites

- Python 3.9+
- CUDA 11.0+ (for GPU support)
- 16GB+ RAM
- 50GB+ disk space

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

## 📊 Dataset Preparation

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

# Generate pseudo-labels on unlabeled data
# Configuration in filter.py:
# - MODEL_FOLDER: Path to trained model
# - INPUT_FOLDER: Unlabeled MRI scans
# - OUTPUT_FOLDER: Output pseudo-labels
```

**What it does**:
1. Removes domain adaptation layers from model checkpoints
2. Initializes nnUNet predictor with cleaned weights
3. Generates predictions on unlabeled MRI scans
4. Saves pseudo-labels for evaluation

### Stage 2: Quality Evaluation & Selection

```bash
# Evaluate pseudo-label quality
python training/evaluate.py

# Configuration in evaluate.py:
# - PSEUDO_LABELS: Generated pseudo-labels
# - VALIDATION_GT: Validation ground truth
# - DICE_THRESHOLD: Quality threshold (e.g., 0.7)
# - SELECTED_OUTPUT: High-quality pseudo-labels
```

**What it does**:
1. Computes Dice scores for pseudo-labels vs validation set
2. Filters cases with Dice ≥ threshold
3. Copies selected pseudo-labels to output folder
4. Saves selection list for tracking

### Stage 3: Model Retraining

```bash
# Train BLUNet with selected pseudo-labels
nnUNetv2_train DATASET_ID 3d_fullres 0 \
    -tr nnUNetTrainerBLUNet \
    --npz  # Use with additional pseudo-labeled data
```

### Iterative Refinement

```bash
# Repeat the cycle for continuous improvement
for iteration in {1..5}; do
    echo "=== Iteration $iteration ==="
    python training/filter.py
    python training/evaluate.py
    nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerBLUNet
done
```

## 🎯 Model Training

### Standard Training (Supervised Only)

```bash
# Plan and preprocess
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# Train BLUNet
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerBLUNet

# Options:
# --c: Use specific configuration
# -p: Use specific plans file
# --npz: Enable deep supervision
```

### Semi-Supervised Training

```bash
# 1. Initial training on labeled data
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerBLUNet

# 2. Generate pseudo-labels
python training/filter.py

# 3. Select high-quality cases
python training/evaluate.py

# 4. Retrain with augmented dataset
nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerBLUNet --continue_training
```

## 🔮 Inference

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

## 📈 Evaluation

### Compute Metrics

```python
from training.evaluate import PseudoLabelSelector

selector = PseudoLabelSelector(dice_threshold=0.7, labels=[1, 2])

# Evaluate predictions
metrics = selector.evaluate_predictions(
    pred_folder="predictions/",
    gt_folder="ground_truth/",
    output_json="results.json"
)

# Select high-quality cases
selected = selector.select_high_quality_cases(
    metrics_json="results.json",
    pred_folder="predictions/",
    output_folder="selected/"
)
```

### Expected Metrics

| Label | Class | Target Dice |
|-------|-------|-------------|
| 1     | Tumor | ≥ 0.70      |
| 2     | Cyst  | ≥ 0.70      |

## 🛠️ Data Utilities

### Resampling

```python
from data_utils import resample_img

# Resample MRI to uniform spacing
resampled_image = resample_img(
    image, 
    out_spacing=[3.0, 3.0, 6.0],  # Target spacing in mm
    is_label=False  # Use B-spline interpolation
)

# Resample segmentation mask
resampled_mask = resample_img(
    mask,
    out_spacing=[3.0, 3.0, 6.0],
    is_label=True  # Use nearest-neighbor interpolation
)
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

## 🐳 Docker Support

```bash
# Build Docker image
docker build -t pancreas-segmentation .

# Run container
docker run --gpus all \
    -v /path/to/data:/data \
    -v /path/to/results:/results \
    pancreas-segmentation \
    nnUNetv2_train DATASET_ID 3d_fullres 0 -tr nnUNetTrainerBLUNet
```

## 📝 Configuration

### Model Hyperparameters

Located in `training/network.py`:

```python
# BLUNet architecture parameters
n_stages = 6                           # Number of encoder/decoder stages
features_per_stage = [32, 64, 128, 256, 320, 320]  # Channels per stage
order_range = [1, 2, 3, 4, 5, 5]      # hgConv orders
drop_path_rate = 0.5                   # Stochastic depth rate
```

### Training Parameters

Located in `trainer/nnUNetTrainerBLUNet.py`:

```python
initial_lr = 1e-2                      # Initial learning rate
num_epochs = 1000                      # Training epochs
batch_size = 2                         # Batch size
deep_supervision = True                # Enable deep supervision
```

## 🔬 Technical Details

### BLUNet Components

1. **High-order Gated Convolution (hgConv)**
   - Multi-scale feature extraction
   - Hierarchical channel splitting
   - Learnable gating mechanism

2. **Hierarchical Gated Convolutional Block (hgcnblock)**
   - Channel transformation
   - LayerNorm + hgConv with residual
   - Standard convolutions
   - Drop path regularization

3. **Residual Encoder/Decoder**
   - Stacked residual blocks
   - Progressive resolution changes
   - Skip connections
   - Symmetric architecture

### Label Definitions

| Value | Structure | Description |
|-------|-----------|-------------|
| 0     | Background | Non-pancreatic tissue |
| 1     | Tumor | Pancreatic ductal adenocarcinoma (PDAC) |
| 2     | Cyst | Pancreatic cystic lesion |

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@article{blunet_pancreas_2025,
  title={Semi-Supervised Pancreatic Tumor Segmentation with BLUNet},
  author={Your Name},
  journal={Medical Image Analysis},
  year={2025}
}
```

## 📄 License

This project incorporates components from:
- **nnUNet**: Apache License 2.0
- **MRSegmentator**: Apache License 2.0
- **Data utilities**: Apache License 2.0 (Radboud UMC)

See individual LICENSE files in subdirectories for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- nnUNet framework by Fabian Isensee et al.
- MRSegmentator by Radboud UMC
- PANORAMA dataset contributors
- Medical Image Analysis community

---

**Last Updated**: November 2025  
**Status**: Active Development  
**Version**: 1.0.0

