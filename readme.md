# Glacier Semantic Segmentation with Attention DeepLab V3+

This repository contains a state-of-the-art glacier semantic segmentation implementation based on DeepLab V3+ with Convolutional Block Attention Module (CBAM) and Test-Time Augmentation (TTA) for the GlacierHack 2025 competition.

## Overview

The model performs binary semantic segmentation to distinguish glacier pixels from non-glacier pixels in multispectral satellite imagery using 5 spectral bands: B2 (Blue), B3 (Green), B4 (Red), B6 (SWIR), and B10 (TIR1).

## Architecture Features

### 1. Attention DeepLab V3+

- **Backbone**: ResNet-34 encoder for feature extraction
- **ASPP with CBAM**: Atrous Spatial Pyramid Pooling enhanced with Convolutional Block Attention Module
- **Decoder**: Depthwise separable convolutions for efficient upsampling
- **Skip Connections**: Low-level features fusion for better boundary preservation

### 2. CBAM (Convolutional Block Attention Module)

- **Channel Attention**: Focuses on important feature channels
- **Spatial Attention**: Emphasizes relevant spatial locations
- **Integration**: Applied to each ASPP branch for enhanced feature representation

### 3. Test-Time Augmentation (TTA)

- **Transformations**: Original, horizontal flip, vertical flip, both flips
- **Ensemble**: Averages predictions from all augmented versions
- **Robustness**: Improves model consistency and performance

## Files Description

### 1. `glacier_segmentation_training.ipynb`

**Google Colab notebook for model training**

**Features:**

- Complete training pipeline from data loading to model evaluation
- Data augmentation with albumentations
- Combined BCE + Dice loss for handling class imbalance
- Comprehensive metrics: MCC, IoU, F1-score, RÂ²
- Training visualization and model checkpointing
- Model compression to stay under 200MB limit
- TTA implementation and evaluation

**Key Sections:**

- Model architecture definition (Attention DeepLab V3+ with CBAM)
- Custom dataset class for .npy file handling
- Training loop with early stopping
- Validation and testing with/without TTA
- Results visualization and model saving

### 2. `solution.py`

**Competition inference script**

**Features:**

- Follows exact competition requirements
- Loads pre-trained model weights (`model.pth`)
- Processes multi-band TIFF images
- Implements TTA for robust predictions
- Saves binary masks in required format

**Key Functions:**

- `AttentionDeepLabV3Plus`: Complete model architecture
- `TTAWrapper`: Test-time augmentation implementation
- `maskgeration`: Main inference function
- `load_and_preprocess_image`: Image preprocessing pipeline

## Data Format

### Training Data

- **Input**: `slice_x_image_y.npy` (512Ã—512Ã—15) â†’ Extract bands [1,2,3,5,9] â†’ (512Ã—512Ã—5)
- **Target**: `slice_x_mask_y.npy` (512Ã—512Ã—3) â†’ Compress to (512Ã—512Ã—1)
  - Channel 0: Clean-ice glacier
  - Channel 1: Debris-covered glacier
  - Channel 2: HKH region mask
  - **Final target**: (clean-ice OR debris-covered) AND hkh_region

### Inference Data

- **Structure**:
  ```
  dataset/
    Band1/
      img001.tif
      img002.tif
      ...
    Band2/
      ...
    Band5/
      ...
  ```

## Installation & Usage

### Training (Google Colab)

1. **Setup Environment**:

   ```bash
   !pip install torch torchvision tqdm opencv-python pillow scikit-learn numpy tifffile segmentation-models-pytorch
   !pip install albumentations matplotlib seaborn
   ```

2. **Mount Google Drive**:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Update Data Path**:

   ```python
   DATA_DIR = '/content/drive/MyDrive/glacier_data'  # Update this path
   ```

4. **Run Notebook**: Execute all cells in `glacier_segmentation_training.ipynb`

### Inference

1. **Install Dependencies**:

   ```bash
   pip install torch torchvision tqdm opencv-python pillow scikit-learn numpy tifffile albumentations
   ```

2. **Run Inference**:
   ```bash
   python solution.py --data /path/to/test/dataset --masks /path/to/masks --out /path/to/output
   ```

## Model Performance

### Evaluation Metrics

- **Primary**: Matthews Correlation Coefficient (MCC) - optimized for balanced performance
- **Secondary**: IoU, F1-score, RÂ² for comprehensive evaluation

### Expected Performance

- **MCC**: >0.85 on validation set
- **IoU**: >0.80 for glacier segmentation
- **F1-score**: >0.90 for binary classification
- **Model Size**: <200MB (competition requirement)

## Key Implementation Details

### 1. Loss Function

```python
Combined Loss = Î± Ã— BCE Loss + (1-Î±) Ã— Dice Loss
```

- Handles class imbalance effectively
- BCE for pixel-wise classification
- Dice for overlap optimization

### 2. Data Augmentation

- Horizontal/Vertical flips
- 90Â° rotations
- Brightness/Contrast adjustments
- Gaussian noise injection

### 3. Band Selection

Selected 5 most informative bands for glacier detection:

- **Band 1 (B2)**: Blue - Snow/ice detection
- **Band 2 (B3)**: Green - Vegetation contrast
- **Band 3 (B4)**: Red - Rock/soil discrimination
- **Band 4 (B6)**: SWIR - Ice/snow separation
- **Band 5 (B10)**: TIR - Thermal signature

### 4. Model Compression

- State dict only (no optimizer)
- Post-training quantization if needed
- Weight pruning for size optimization

## Competition Requirements Compliance

âœ… **Script Name**: `solution.py`  
âœ… **Function Name**: `maskgeration(imagepath, out_dir)`  
âœ… **Model Weights**: `model.pth` (<200MB)  
âœ… **Dependencies**: Listed in script header  
âœ… **Output Format**: Binary TIFF masks  
âœ… **File Naming**: Matches input filenames  
âœ… **Arguments**: `--data`, `--masks`, `--out`

## Architecture Diagram

```
Input (5 bands) â†’ ResNet-34 Encoder â†’ ASPP + CBAM â†’ Decoder â†’ Binary Mask
                      â†“                    â†“             â†“
                 Low-level feat.    High-level feat.   Skip Connection
                      â†“                    â†“             â†“
                  Projection         Attention       Upsampling
                      â†“                    â†“             â†“
                      â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â†’ Concatenate â†â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

## References

1. **DeepLab V3+**: Chen, L. C., et al. "Encoder-decoder with atrous separable convolution for semantic image segmentation." ECCV 2018.

2. **CBAM**: Woo, S., et al. "CBAM: Convolutional block attention module." ECCV 2018.

3. **Glacier Extraction**: Chu, X., et al. "Glacier extraction based on high spatial resolution remote sensing images using a deep learning approach with attention mechanism." The Cryosphere 2022.

## Citation

```bibtex
@article{glacier_segmentation_2025,
  title={Glacier Semantic Segmentation with Attention DeepLab V3+},
  author={Your Name},
  journal={GlacierHack 2025},
  year={2025}
}
```

## Contact

For questions or issues, please contact: [your.email@domain.com]

---

**Note**: This implementation is designed specifically for the GlacierHack 2025 competition requirements and follows all submission guidelines.

---

# Core deep learning frameworks

torch>=1.9.0
torchvision>=0.10.0

# Data processing and manipulation

numpy>=1.21.0
pillow>=8.3.0
opencv-python>=4.5.0
scikit-learn>=1.0.0
tifffile>=2021.7.2

# Data augmentation and visualization

albumentations>=1.1.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Progress bars and utilities

tqdm>=4.62.0

# Optional: For segmentation models pytorch (if using pre-built models)

# segmentation-models-pytorch>=0.2.0
