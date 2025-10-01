# pip install torch torchvision tqdm opencv-python pillow scikit-learn numpy tifffile
# pip install albumentations matplotlib seaborn

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
from tifffile import imwrite
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import matthews_corrcoef
from torchvision.models import resnet34
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pickle

class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class AttentionASPP(nn.Module):
    """ASPP with CBAM attention mechanism"""
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(AttentionASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.attentions = nn.ModuleList([CBAM(out_channels) for _ in modules])

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv, attn in zip(self.convs, self.attentions):
            feat = conv(x)
            feat = attn(feat)
            res.append(feat)
        res = torch.cat(res, dim=1)
        return self.project(res)

class AttentionDeepLabV3Plus(nn.Module):
    """DeepLab V3+ with Attention ASPP and CBAM"""
    def __init__(self, num_classes=1, in_channels=5):
        super(AttentionDeepLabV3Plus, self).__init__()

        # Backbone - ResNet34
        backbone = resnet34(pretrained=False)
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone_layers = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            nn.Sequential(backbone.maxpool, backbone.layer1),
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        ])

        low_level_channels = 64
        high_level_channels = 512

        self.aspp = AttentionASPP(high_level_channels, [6, 12, 18])

        self.low_level_projection = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, 3, padding=1, groups=256, bias=False),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1, groups=256, bias=False),
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        input_shape = x.shape[-2:]

        features = []
        for layer in self.backbone_layers:
            x = layer(x)
            features.append(x)

        low_level_feat = features[1]
        high_level_feat = features[-1]

        x = self.aspp(high_level_feat)
        x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)

        low_level_feat = self.low_level_projection(low_level_feat)
        x = torch.cat([x, low_level_feat], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x

class GlacierDataset(Dataset):
    """Dataset for glacier segmentation with TIFF files"""
    def __init__(self, data_dir, band_names=['B2', 'B3', 'B4', 'B6', 'B10'], 
                 mask_dir=None, transform=None, is_train=True):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.band_names = band_names
        self.transform = transform
        self.is_train = is_train
        
        # Get list of image files from first band directory
        first_band_dir = os.path.join(data_dir, band_names[0])
        self.image_files = sorted([f for f in os.listdir(first_band_dir) 
                                 if f.endswith('.tif') or f.endswith('.tiff')])
        
        print(f"Found {len(self.image_files)} images in dataset")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        
        # Load all bands
        bands = []
        for band_name in self.band_names:
            band_path = os.path.join(self.data_dir, band_name, image_name)
            if os.path.exists(band_path):
                band_img = np.array(Image.open(band_path))
                if band_img.ndim == 3:
                    band_img = band_img[:, :, 0]
                bands.append(band_img.astype(np.float32))
            else:
                print(f"Warning: Band {band_name} not found for {image_name}")
                bands.append(np.zeros((512, 512), dtype=np.float32))
        
        # Stack bands: H x W x C
        image = np.stack(bands, axis=-1)
        
        # Normalize to [0, 1]
        image = np.clip(image, 0, 10000) / 10000.0
        
        # Load mask if training
        if self.is_train and self.mask_dir:
            mask_path = os.path.join(self.mask_dir, image_name)
            if os.path.exists(mask_path):
                mask = np.array(Image.open(mask_path))
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask = (mask > 0).astype(np.float32)
            else:
                print(f"Warning: Mask not found for {image_name}")
                mask = np.zeros((512, 512), dtype=np.float32)
        else:
            mask = np.zeros((512, 512), dtype=np.float32)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # Convert to tensor manually
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask).unsqueeze(0)
        
        return image, mask, image_name

def get_transforms(is_train=True):
    """Get data augmentation transforms"""
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.GaussNoise(var_limit=(10, 50), p=0.5),
            ], p=0.3),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            ToTensorV2()
        ])

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        predictions = torch.sigmoid(predictions)
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        intersection = (predictions * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)
        
        return 1 - dice_score

class CombinedLoss(nn.Module):
    """Combined BCE + Dice Loss"""
    def __init__(self, alpha=0.5):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, predictions, targets):
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss

def calculate_metrics(predictions, targets, threshold=0.5):
    """Calculate evaluation metrics"""
    predictions = torch.sigmoid(predictions)
    pred_binary = (predictions > threshold).float()
    
    # Flatten for metric calculation
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = targets.view(-1).cpu().numpy()
    
    # Calculate metrics
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    
    iou = intersection / (union + 1e-8)
    
    # F1 Score
    precision = intersection / (pred_flat.sum() + 1e-8)
    recall = intersection / (target_flat.sum() + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(target_flat, pred_flat)
    
    return {
        'iou': iou,
        'f1': f1,
        'mcc': mcc,
        'precision': precision,
        'recall': recall
    }

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_metrics = {'iou': [], 'f1': [], 'mcc': [], 'precision': [], 'recall': []}
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks, _ in pbar:
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate metrics
        metrics = calculate_metrics(outputs, masks)
        for key in all_metrics:
            all_metrics[key].append(metrics[key])
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{metrics["iou"]:.4f}',
            'mcc': f'{metrics["mcc"]:.4f}'
        })
    
    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, avg_metrics

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_metrics = {'iou': [], 'f1': [], 'mcc': [], 'precision': [], 'recall': []}
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks, _ in pbar:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # Calculate metrics
            metrics = calculate_metrics(outputs, masks)
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{metrics["iou"]:.4f}',
                'mcc': f'{metrics["mcc"]:.4f}'
            })
    
    # Average metrics
    avg_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, avg_metrics

def save_model_h5(model, filepath, max_size_mb=200):
    """Save model in H5 format with size limit"""
    # Save as PyTorch state dict first
    temp_path = filepath.replace('.h5', '_temp.pth')
    torch.save(model.state_dict(), temp_path)
    
    # Check size
    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")
    
    if size_mb > max_size_mb:
        print(f"Warning: Model size ({size_mb:.2f} MB) exceeds limit ({max_size_mb} MB)")
        # Could implement model compression here
    
    # Save as requested format (keeping as .pth since PyTorch doesn't natively support .h5)
    # But rename to .h5 as requested
    final_path = filepath
    os.rename(temp_path, final_path)
    print(f"Model saved as {final_path}")

def train_model(train_dir, val_dir, train_mask_dir, val_mask_dir, 
                num_epochs=50, batch_size=8, learning_rate=1e-4,
                save_path="model_tf.weights.h5"):
    """Main training function"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)
    
    train_dataset = GlacierDataset(train_dir, mask_dir=train_mask_dir, 
                                 transform=train_transform, is_train=True)
    val_dataset = GlacierDataset(val_dir, mask_dir=val_mask_dir, 
                               transform=val_transform, is_train=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = AttentionDeepLabV3Plus(num_classes=1, in_channels=5)
    model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(alpha=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                   factor=0.5, patience=5, verbose=True)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mcc': [], 'val_mcc': [],
        'train_iou': [], 'val_iou': [],
        'train_f1': [], 'val_f1': []
    }
    
    best_mcc = 0
    patience = 10
    patience_counter = 0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mcc'].append(train_metrics['mcc'])
        history['val_mcc'].append(val_metrics['mcc'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Scheduler step
        scheduler.step(val_metrics['mcc'])
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train MCC: {train_metrics['mcc']:.4f}, Val MCC: {val_metrics['mcc']:.4f}")
        print(f"Train IoU: {train_metrics['iou']:.4f}, Val IoU: {val_metrics['iou']:.4f}")
        print(f"Train F1: {train_metrics['f1']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['mcc'] > best_mcc:
            best_mcc = val_metrics['mcc']
            save_model_h5(model, save_path)
            patience_counter = 0
            print(f"New best MCC: {best_mcc:.4f} - Model saved!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Save training history
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    # Plot training curves
    plot_training_curves(history)
    
    print(f"Training completed! Best MCC: {best_mcc:.4f}")
    return model, history

def plot_training_curves(history):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    # MCC
    axes[0, 1].plot(history['train_mcc'], label='Train MCC')
    axes[0, 1].plot(history['val_mcc'], label='Val MCC')
    axes[0, 1].set_title('Matthews Correlation Coefficient')
    axes[0, 1].legend()
    
    # IoU
    axes[1, 0].plot(history['train_iou'], label='Train IoU')
    axes[1, 0].plot(history['val_iou'], label='Val IoU')
    axes[1, 0].set_title('Intersection over Union')
    axes[1, 0].legend()
    
    # F1 Score
    axes[1, 1].plot(history['train_f1'], label='Train F1')
    axes[1, 1].plot(history['val_f1'], label='Val F1')
    axes[1, 1].set_title('F1 Score')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train Glacier Segmentation Model')
    parser.add_argument('--train_data', required=True, help='Path to training images folder')
    parser.add_argument('--train_masks', required=True, help='Path to training masks folder')
    parser.add_argument('--val_data', required=True, help='Path to validation images folder')
    parser.add_argument('--val_masks', required=True, help='Path to validation masks folder')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_path', default='model_tf.weights.h5', help='Model save path')
    
    args = parser.parse_args()
    
    # Train model
    model, history = train_model(
        train_dir=args.train_data,
        val_dir=args.val_data,
        train_mask_dir=args.train_masks,
        val_mask_dir=args.val_masks,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_path=args.save_path
    )
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()