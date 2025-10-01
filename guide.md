# Glacier Segmentation Training Guide

## File Structure

Your dataset should be organized as follows:

```
dataset/
â”œâ”€â”€ train/
â”‚   â”œâ”€â”€ B2/
â”‚   â”‚   â”œâ”€â”€ image001.tif
â”‚   â”‚   â”œâ”€â”€ image002.tif
â”‚   â”‚   â””â”€â”€ ...
â”‚   â”œâ”€â”€ B3/
â”‚   â”‚   â”œâ”€â”€ image001.tif
â”‚   â”‚   â”œâ”€â”€ image002.tif
â”‚   â”‚   â””â”€â”€ ...
â”‚   â”œâ”€â”€ B4/
â”‚   â”‚   â””â”€â”€ ...
â”‚   â”œâ”€â”€ B6/
â”‚   â”‚   â””â”€â”€ ...
â”‚   â””â”€â”€ B10/
â”‚       â””â”€â”€ ...
â”œâ”€â”€ train_masks/
â”‚   â”œâ”€â”€ image001.tif
â”‚   â”œâ”€â”€ image002.tif
â”‚   â””â”€â”€ ...
â”œâ”€â”€ val/
â”‚   â”œâ”€â”€ B2/
â”‚   â”œâ”€â”€ B3/
â”‚   â”œâ”€â”€ B4/
â”‚   â”œâ”€â”€ B6/
â”‚   â””â”€â”€ B10/
â””â”€â”€ val_masks/
    â”œâ”€â”€ image001.tif
    â””â”€â”€ ...
```

## Dependencies Installation

```bash
pip install torch torchvision tqdm opencv-python pillow scikit-learn numpy tifffile
pip install albumentations matplotlib seaborn
```

## Training the Model

### Basic Training Command

```bash
python glacier_training.py \
    --train_data ./dataset/train \
    --train_masks ./dataset/train_masks \
    --val_data ./dataset/val \
    --val_masks ./dataset/val_masks \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-4 \
    --save_path model_tf.weights.h5
```

### Advanced Training Options

```bash
# For faster training with larger batch size (requires more GPU memory)
python glacier_training.py \
    --train_data ./dataset/train \
    --train_masks ./dataset/train_masks \
    --val_data ./dataset/val \
    --val_masks ./dataset/val_masks \
    --epochs 100 \
    --batch_size 16 \
    --lr 2e-4 \
    --save_path model_tf.weights.h5

# For slower, more careful training
python glacier_training.py \
    --train_data ./dataset/train \
    --train_masks ./dataset/train_masks \
    --val_data ./dataset/val \
    --val_masks ./dataset/val_masks \
    --epochs 200 \
    --batch_size 4 \
    --lr 5e-5 \
    --save_path model_tf.weights.h5
```

## Running Inference

After training, use the solution.py script for inference:

```bash
python solution.py \
    --data ./test_dataset \
    --masks ./dummy_masks \
    --out ./predictions
```

## Data Preprocessing Notes

1. **Band Selection**: The model expects exactly 5 bands in this order:

   - B2 (Blue)
   - B3 (Green)
   - B4 (Red)
   - B6 (SWIR)
   - B10 (TIR1)

2. **Image Format**:

   - Input images should be TIFF files
   - Expected size: 512x512 pixels
   - Data type: uint16 or float32
   - Value range: 0-10000 (typical Landsat range)

3. **Mask Format**:
   - Binary masks with values 0 (non-glacier) and 255 (glacier)
   - Same size as input images (512x512)
   - TIFF format

## Model Features

- **Architecture**: DeepLab V3+ with ResNet-34 backbone
- **Attention**: CBAM (Convolutional Block Attention Module)
- **ASPP**: Atrous Spatial Pyramid Pooling with attention
- **TTA**: Test-Time Augmentation for better inference
- **Loss Function**: Combined BCE + Dice Loss
- **Metrics**: MCC, IoU, F1-score, Precision, Recall

## Training Monitoring

The training script will:

- Save the best model based on validation MCC
- Generate training curves plot
- Save training history as pickle file
- Implement early stopping with patience=10 epochs
- Use learning rate scheduling

## Model Size Optimization

The model is designed to stay under 200MB by:

- Using ResNet-34 (smaller backbone)
- Saving only state_dict (no optimizer)
- Efficient architecture design

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch_size to 4 or 2
2. **Model too large**: The script automatically checks model size
3. **Missing bands**: Script will create dummy zero-filled bands and warn you
4. **File format issues**: Ensure all images are TIFF format

### Performance Tips:

1. **Data Loading**: Use num_workers=4 for faster data loading
2. **Mixed Precision**: Can be added for faster training
3. **Gradient Accumulation**: For effective larger batch sizes

## Expected Performance

- **Training Time**: ~2-4 hours on single GPU (depending on dataset size)
- **Validation MCC**: >0.85 for good model
- **Model Size**: <200MB
- **Inference Speed**: ~0.5-1 second per image with TTA

## File Outputs

After training:

- `model_tf.weights.h5`: Trained model weights
- `training_history.pkl`: Training metrics history
- `training_curves.png`: Training visualization plots

## Example Training Session

```
Starting training...

Epoch 1/50
Training: 100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 100/100 [02:30<00:00,  1.50s/it, loss=0.4521, iou=0.6234, mcc=0.5891]
Validation: 100%|â–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆâ–ˆ| 25/25 [00:30<00:00,  1.20s/it, loss=0.3921, iou=0.7123, mcc=0.6789]
Train Loss: 0.4521, Val Loss: 0.3921
Train MCC: 0.5891, Val MCC: 0.6789
Train IoU: 0.6234, Val IoU: 0.7123
Train F1: 0.7456, Val F1: 0.8123
New best MCC: 0.6789 - Model saved!

...

Training completed! Best MCC: 0.8567
Model size: 156.78 MB
```

This guide should help you successfully train and deploy the glacier segmentation model with your specific dataset format.
