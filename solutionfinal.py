# pip install tensorflow numpy pillow tifffile scikit-learn

import argparse
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import tifffile

def convolution_block(block_input, num_filters=256, kernel_size=3, dilation_rate=1, padding="same", use_bias=False):
    """Build convolution block with batch normalization and ReLU activation"""
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=tf.keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)

def DilatedSpatialPyramidPooling(dspp_input):
    """Dilated Spatial Pyramid Pooling module"""
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size=512, num_classes=1):
    """DeepLabV3+ model for glacier segmentation with custom 5-channel input"""
    model_input = layers.Input(shape=(image_size, image_size, 5))
    
    # Custom preprocessing for 5-channel input (normalize each channel)
    x = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=-1))(model_input)
    
    # Simple encoder backbone for 5-channel input (lightweight version to keep under 200MB)
    # Encoder path
    x = convolution_block(x, num_filters=64, kernel_size=3)
    x = convolution_block(x, num_filters=64, kernel_size=3)
    low_level_features = x  # Save for decoder
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    x = convolution_block(x, num_filters=128, kernel_size=3)
    x = convolution_block(x, num_filters=128, kernel_size=3)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    x = convolution_block(x, num_filters=256, kernel_size=3)
    x = convolution_block(x, num_filters=256, kernel_size=3)
    x = layers.MaxPooling2D(pool_size=2)(x)
    
    # ASPP module
    x = DilatedSpatialPyramidPooling(x)
    
    # Decoder
    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    
    # Process low-level features
    input_b = convolution_block(low_level_features, num_filters=48, kernel_size=1)
    
    # Concatenate and refine
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x, num_filters=256)
    x = convolution_block(x, num_filters=256)
    
    # Final upsampling and output
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    
    # Binary classification output with sigmoid activation
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same", activation="sigmoid")(x)
    
    return models.Model(inputs=model_input, outputs=model_output)

def normalize_band_data(arr):
    """Normalize band data using z-score normalization"""
    arr = arr.astype(np.float32)
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 0:
        return (arr - mean) / std
    return arr

def load_and_stack_bands(imagepath, filename):
    """Load and stack all 5 bands for a given filename"""
    bands = []
    band_folders = sorted([f for f in os.listdir(imagepath) if os.path.isdir(os.path.join(imagepath, f))])
    
    for band_folder in band_folders:
        band_path = os.path.join(imagepath, band_folder, filename)
        if os.path.exists(band_path):
            # Load TIFF image
            arr = np.array(Image.open(band_path))
            if arr.ndim == 3:
                arr = arr[..., 0]  # Take first channel if RGB
            # Normalize the band
            arr = normalize_band_data(arr)
            bands.append(arr)
    
    if len(bands) == 5:
        # Stack bands to create (H, W, 5) array
        return np.stack(bands, axis=-1)
    else:
        return None

def maskgeration(imagepath, out_dir):
    """Generate glacier masks using trained DeepLabV3+ model"""
    # Load the trained model
    model = DeeplabV3Plus(image_size=512, num_classes=1)
    model.load_weights("model_tf.weights.h5")
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Get list of files from the first band folder (reference)
    band_folders = sorted([f for f in os.listdir(imagepath) if os.path.isdir(os.path.join(imagepath, f))])
    if not band_folders:
        print("No band folders found!")
        return
    
    reference_folder = os.path.join(imagepath, band_folders[0])
    filenames = [f for f in os.listdir(reference_folder) if f.endswith('.tif')]
    
    for filename in filenames:
        print(f"Processing {filename}...")
        
        # Load and stack all 5 bands
        input_array = load_and_stack_bands(imagepath, filename)
        
        if input_array is not None:
            # Add batch dimension and predict
            input_batch = np.expand_dims(input_array, axis=0)
            
            # Predict glacier mask
            prediction = model.predict(input_batch, verbose=0)
            
            # Post-process prediction (threshold at 0.5)
            binary_mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)
            
            # Convert to 0-255 range (0 for non-glacier, 255 for glacier)
            output_mask = binary_mask * 255
            
            # Save as TIFF
            output_path = os.path.join(out_dir, filename)
            tifffile.imwrite(output_path, output_mask)
            print(f"Saved mask to {output_path}")
        else:
            print(f"Could not load all bands for {filename}")

# Do not update this section
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to test images folder")
    parser.add_argument("--masks", required=True, help="Path to masks folder (unused)")
    parser.add_argument("--out", required=True, help="Path to output predictions")
    args = parser.parse_args()

    # Build band â†’ folder map
    imagepath = {}
    for band in os.listdir(args.data):
        band_path = os.path.join(args.data, band)
        if os.path.isdir(band_path):
            imagepath[band] = band_path

    print(f"Processing bands: {list(imagepath.keys())}")

    # Run mask generation and save predictions
    maskgeration(imagepath, args.out)

if __name__ == "__main__":
    main()