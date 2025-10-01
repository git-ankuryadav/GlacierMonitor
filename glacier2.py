# pip install torch torchvision tqdm opencv-python pillow scikit-learn numpy tifffile
# pip install albumentations

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from tifffile import imwrite
from torchvision.models import resnet34

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

        # Add CBAM attention to each branch
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
            feat = attn(feat)  # Apply CBAM attention
            res.append(feat)
        res = torch.cat(res, dim=1)
        return self.project(res)

class AttentionDeepLabV3Plus(nn.Module):
    """DeepLab V3+ with Attention ASPP and CBAM"""
    def __init__(self, num_classes=1, in_channels=5):
        super(AttentionDeepLabV3Plus, self).__init__()

        # Backbone - ResNet34
        backbone = resnet34(pretrained=False)  # No pretrained for custom input channels

        # Modify first conv layer for 5 input channels
        backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.backbone_layers = nn.ModuleList([
            nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu),
            nn.Sequential(backbone.maxpool, backbone.layer1),
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        ])

        # Low-level features from layer1 (after maxpool)
        low_level_channels = 64

        # High-level features from layer4
        high_level_channels = 512

        # Attention ASPP
        self.aspp = AttentionASPP(high_level_channels, [6, 12, 18])

        # Low-level feature projection
        self.low_level_projection = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Depthwise separable conv
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

        # Encoder
        features = []
        for layer in self.backbone_layers:
            x = layer(x)
            features.append(x)

        low_level_feat = features[1]  # After maxpool + layer1
        high_level_feat = features[-1]  # After layer4

        # ASPP
        x = self.aspp(high_level_feat)

        # Upsample high-level features
        x = F.interpolate(x, size=low_level_feat.shape[-2:], mode='bilinear', align_corners=False)

        # Low-level feature projection
        low_level_feat = self.low_level_projection(low_level_feat)

        # Concatenate
        x = torch.cat([x, low_level_feat], dim=1)

        # Decoder
        x = self.decoder(x)

        # Final upsample
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return x

class TTAWrapper(nn.Module):
    """Test-Time Augmentation wrapper"""
    def __init__(self, model):
        super(TTAWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Original
        pred1 = self.model(x)

        # Horizontal flip
        pred2 = torch.flip(self.model(torch.flip(x, dims=[3])), dims=[3])

        # Vertical flip
        pred3 = torch.flip(self.model(torch.flip(x, dims=[2])), dims=[2])

        # Both flips
        pred4 = torch.flip(torch.flip(self.model(torch.flip(torch.flip(x, dims=[2]), dims=[3])), dims=[3]), dims=[2])

        # Average predictions
        return (pred1 + pred2 + pred3 + pred4) / 4.0

def normalize_bands(image_array):
    """Normalize image bands to [0, 1] range"""
    # Handle potential outliers by clipping to reasonable range
    image_array = np.clip(image_array, 0, 10000)  # Landsat typical range
    return image_array / 10000.0

def load_and_preprocess_image(band_paths, image_name):
    """Load and preprocess multi-band image"""
    bands = []

    # Expected band order: B2, B3, B4, B6, B10 (as specified in your requirements)
    expected_bands = ['B2', 'B3', 'B4', 'B6', 'B10']

    for band_name in expected_bands:
        if band_name in band_paths:
            band_path = os.path.join(band_paths[band_name], image_name)
            if os.path.exists(band_path):
                # Load TIFF image
                band_img = np.array(Image.open(band_path))
                if band_img.ndim == 3:
                    band_img = band_img[:, :, 0]  # Take first channel if RGB
                bands.append(band_img.astype(np.float32))
            else:
                print(f"Warning: Band {band_name} file not found: {band_path}")
                # Create dummy band filled with zeros
                bands.append(np.zeros((512, 512), dtype=np.float32))
        else:
            print(f"Warning: Band {band_name} directory not found in band_paths")
            bands.append(np.zeros((512, 512), dtype=np.float32))

    if len(bands) != 5:
        print(f"Warning: Expected 5 bands, got {len(bands)} for image {image_name}")
        return None

    # Stack bands: H x W x C
    image = np.stack(bands, axis=-1)

    # Normalize
    image = normalize_bands(image)

    # Convert to tensor: C x H x W
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    return image_tensor

def maskgeration(imagepath, out_dir):
    """Generate binary masks for glacier segmentation"""

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionDeepLabV3Plus(num_classes=1, in_channels=5)

    # Load trained weights - try both possible file names
    weight_files = ["model_tf.weights.h5", "model.pth"]
    model_loaded = False
    
    for weight_file in weight_files:
        if os.path.exists(weight_file):
            try:
                model.load_state_dict(torch.load(weight_file, map_location=device))
                print(f"Model loaded successfully from {weight_file}")
                model_loaded = True
                break
            except Exception as e:
                print(f"Failed to load {weight_file}: {e}")
                continue
    
    if not model_loaded:
        print("Error: No valid model weights found!")
        print("Please ensure either 'model_tf.weights.h5' or 'model.pth' exists")
        return

    model.to(device)

    # Wrap with TTA
    tta_model = TTAWrapper(model)
    tta_model.eval()

    print(f"Model loaded successfully on {device}")
    print(f"Processing bands: {list(imagepath.keys())}")

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # Get list of image files from the first band directory
    available_bands = [band for band in imagepath.keys() if os.path.isdir(imagepath[band])]
    if not available_bands:
        print("Error: No valid band directories found!")
        return
    
    first_band = sorted(available_bands)[0]
    image_files = sorted(os.listdir(imagepath[first_band]))
    image_files = [f for f in image_files if f.endswith(('.tif', '.tiff'))]

    print(f"Found {len(image_files)} images to process")

    if len(image_files) == 0:
        print("Warning: No TIFF images found in the dataset!")
        return

    # Process each image
    with torch.no_grad():
        for i, image_name in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_name}...")

            # Load and preprocess image
            image_tensor = load_and_preprocess_image(imagepath, image_name)

            if image_tensor is None:
                print(f"Skipping {image_name} due to loading issues")
                continue

            image_tensor = image_tensor.to(device)

            try:
                # Predict with TTA
                output = tta_model(image_tensor)
                prediction = torch.sigmoid(output).squeeze().cpu().numpy()

                # Convert to binary mask (threshold = 0.5)
                binary_mask = (prediction > 0.5).astype(np.uint8) * 255

                # Save mask with same filename as input
                output_path = os.path.join(out_dir, image_name)

                # Save as TIFF
                imwrite(output_path, binary_mask, dtype=np.uint8)

                print(f"Saved mask: {output_path}")
                
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue

    print("Mask generation completed!")

# Do not update this section
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to test images folder")
    parser.add_argument("--masks", required=True, help="Path to masks folder (unused)")
    parser.add_argument("--out", required=True, help="Path to output predictions")
    args = parser.parse_args()

    # Build band â†’ folder map
    imagepath = {}
    
    print(f"Scanning data directory: {args.data}")
    
    if not os.path.exists(args.data):
        print(f"Error: Data directory {args.data} does not exist!")
        return
        
    for item in os.listdir(args.data):
        item_path = os.path.join(args.data, item)
        if os.path.isdir(item_path):
            imagepath[item] = item_path
            print(f"Found band directory: {item}")

    if not imagepath:
        print("Error: No band directories found in data folder!")
        return

    print(f"Processing bands: {list(imagepath.keys())}")

    # Run mask generation and save predictions
    maskgeration(imagepath, args.out)

if __name__ == "__main__":
    main()