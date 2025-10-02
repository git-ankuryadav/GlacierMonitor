# Training script for DeepLabV3+ glacier segmentation model
# pip install tensorflow numpy pillow tifffile scikit-learn

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import matthews_corrcoef
import tifffile
import re

print("TensorFlow version:", tf.__version__)

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
    x = layers.ReLU()(x)  # Changed from tf.nn.relu to layers.ReLU()
    return x

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
    model_input = layers.Input(shape=(image_size, image_size, 5))
    x = layers.Lambda(lambda x: tf.keras.utils.normalize(x, axis=-1))(model_input)

    # Encoder
    x = convolution_block(x, 64, 3)
    x = convolution_block(x, 64, 3)
    low_level_features = x
    x = layers.MaxPooling2D(2)(x)

    x = convolution_block(x, 128, 3)
    x = convolution_block(x, 128, 3)
    x = layers.MaxPooling2D(2)(x)

    x = convolution_block(x, 256, 3)
    x = convolution_block(x, 256, 3)
    x = layers.MaxPooling2D(2)(x)

    # ASPP
    x = DilatedSpatialPyramidPooling(x)

    # Decoder: upsample ASPP output to 128×128
    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear"
    )(x)

    # Process low-level features: 1×1 conv then downsample to 128×128
    input_b = convolution_block(low_level_features, 48, 1)
    input_b = layers.MaxPooling2D(2)(input_b)  # 512→256
    input_b = layers.MaxPooling2D(2)(input_b)  # 256→128

    # Concatenate at 128×128
    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x, 256)
    x = convolution_block(x, 256)

    # Final upsampling to original size
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear"
    )(x)

    outputs = layers.Conv2D(num_classes, 1, padding="same", activation="sigmoid")(x)
    return models.Model(inputs=model_input, outputs=outputs)

def normalize_band_data(arr):
    """Normalize band data using z-score normalization"""
    arr = arr.astype(np.float32)
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 0:
        return (arr - mean) / std
    return arr

def extract_tile_id(filename):
    """Extract tile ID from band or mask filename"""
    match = re.search(r'(\d{2}_\d{2})(?=\.tif$)', filename)
    if match:
        return match.group(1)
    return None

def load_files_map(folder):
    """Create a mapping of tile_id to filename for files in a folder"""
    if not os.path.exists(folder):
        return {}
    files = [f for f in os.listdir(folder) if f.endswith('.tif')]
    tile_map = {}
    for f in files:
        tile_id = extract_tile_id(f)
        if tile_id:
            tile_map[tile_id] = f
    return tile_map

def load_band_arrays(imagepath, tile_id, band_file_maps):
    """Load and normalize arrays for all bands for a given tile"""
    band_arrays = []
    H, W = None, None
    for band_name in sorted(imagepath.keys()):
        band_folder = imagepath[band_name]
        band_files = band_file_maps[band_name]
        if tile_id not in band_files:
            return None, None, None
        file_path = os.path.join(band_folder, band_files[tile_id])
        if not os.path.exists(file_path):
            return None, None, None
        arr = np.array(Image.open(file_path))
        if arr.ndim == 3:
            arr = arr[..., 0]  # Take first channel if RGB
        H, W = arr.shape
        arr_normalized = normalize_band_data(arr)
        band_arrays.append(arr_normalized)
    if not band_arrays or len(band_arrays) != 5:
        return None, None, None
    X = np.stack(band_arrays, axis=-1)
    return X, H, W

def load_label_array(label_folder, tile_id):
    """Load and process label array using Y_output_resized_{tile_id}.tif naming"""
    label_filename = f'Y_output_resized_{tile_id}.tif'
    label_path = os.path.join(label_folder, label_filename)
    if not os.path.exists(label_path):
        return None
    label_img = np.array(Image.open(label_path))
    if label_img.ndim == 3:
        label_img = label_img[..., 0]  # Take first channel if RGB
    label_binary = (label_img > 127).astype(np.float32)
    return label_binary

def prepare_dataset(imagepath, label_folder):
    """Prepare dataset for training"""
    print("Preparing dataset...")

    band_file_maps = {b: load_files_map(f) for b, f in imagepath.items()}
    
    # Debug prints
    for b, mapping in band_file_maps.items():
        print(f"Band {b} tile IDs:", list(mapping.keys())[:5])  # Show first 5
    
    # Get common tile IDs from all bands
    if not band_file_maps:
        raise ValueError("No band folders found")
    
    common_tile_ids = set.intersection(*(set(band_file_maps[b].keys()) for b in band_file_maps))
    tiles = sorted(common_tile_ids)

    print(f"Found {len(tiles)} common tiles across all bands")
    
    X_data = []
    y_data = []
    
    for tile_id in tiles:
        X_img, H, W = load_band_arrays(imagepath, tile_id, band_file_maps)
        y_img = load_label_array(label_folder, tile_id)
        
        if X_img is None or y_img is None:
            print(f"Skipping tile {tile_id} due to loading issues")
            continue
        
        if y_img.shape != X_img.shape[:2]:
            print(f"Skipping tile {tile_id} due to shape mismatch: X={X_img.shape}, y={y_img.shape}")
            continue
        
        X_data.append(X_img)
        y_data.append(y_img[..., np.newaxis])  # Add channel dimension for labels
        
        print(f"Loaded tile {tile_id}: X shape={X_img.shape}, y shape={y_img.shape}")
    
    if not X_data:
        raise ValueError("No valid data found.")
    
    X_array = np.array(X_data)
    y_array = np.array(y_data)
    
    print(f"Final dataset shape: X={X_array.shape}, y={y_array.shape}")
    return X_array, y_array

def dice_coefficient(y_true, y_pred):
    """Dice coefficient for model evaluation"""
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + 1e-7) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1e-7)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    """Combined loss: binary crossentropy + dice loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def train_and_save_model(train_imagepath, train_label_folder, weights_path):
    """Train model and save weights"""
    print("Loading training data...")
    X_train, y_train = prepare_dataset(train_imagepath, train_label_folder)
    
    # Create model
    print("Creating model...")
    model = DeeplabV3Plus(image_size=512, num_classes=1)
    
    # Compile model with combined loss
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=['accuracy', dice_coefficient]
    )
    
    # Print model summary
    print("Model summary:")
    model.summary()
    
    # Calculate model size
    param_count = model.count_params()
    estimated_size_mb = (param_count * 4) / (1024 * 1024)  # 4 bytes per float32 parameter
    print(f"Model parameters: {param_count:,}")
    print(f"Estimated model size: {estimated_size_mb:.2f} MB")
    
    if estimated_size_mb > 200:
        print("WARNING: Model size may exceed 200MB limit!")
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            weights_path,
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
    ]
    
    # Train model
    print("Starting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=2,  # Small batch size due to memory constraints
        epochs=50,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Training completed!")
    
    # Save final model weights
    model.save_weights(weights_path)
    print(f"Model weights saved as {weights_path}")
    
    # Check final file size
    if os.path.exists(weights_path):
        file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
        print(f"Actual model file size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 200:
            print("WARNING: Model file size exceeds 200MB limit!")
        else:
            print("✓ Model file size is within the 200MB limit")

def load_and_evaluate_model(test_imagepath, test_label_folder, weights_path):
    """Load model and evaluate on test set"""
    print("Loading test data...")
    X_test, y_test = prepare_dataset(test_imagepath, test_label_folder)
    
    print("Creating model and loading weights...")
    model = DeeplabV3Plus(image_size=512, num_classes=1)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=['accuracy', dice_coefficient]
    )
    model.load_weights(weights_path)
    
    # Predict on test set
    y_pred_prob = model.predict(X_test)
    y_pred_binary = (y_pred_prob > 0.5).astype(np.int32)
    
    # Calculate MCC for each image and average
    mcc_scores = []
    for i in range(len(y_test)):
        y_true_flat = y_test[i].flatten()
        y_pred_flat = y_pred_binary[i].flatten()
        
        # Skip if all predictions are the same class
        if len(np.unique(y_pred_flat)) == 1:
            continue
            
        mcc = matthews_corrcoef(y_true_flat, y_pred_flat)
        mcc_scores.append(mcc)
    
    if mcc_scores:
        avg_mcc = np.mean(mcc_scores)
        print(f"Average Matthews Correlation Coefficient: {avg_mcc:.4f}")
    else:
        print("Could not calculate MCC (all predictions same class)")
    
    return avg_mcc if mcc_scores else 0.0

if __name__ == '__main__':
    train_dir = 'train'
    test_dir = 'test'
    weights_file = 'model_tf.weights.h5'

    train_imagepath = {
        'band1': os.path.join(train_dir, 'Band1'),
        'band2': os.path.join(train_dir, 'Band2'),
        'band3': os.path.join(train_dir, 'Band3'),
        'band4': os.path.join(train_dir, 'Band4'),
        'band5': os.path.join(train_dir, 'Band5'),
    }
    train_label_folder = os.path.join(train_dir, 'Label')

    test_imagepath = {
        'band1': os.path.join(test_dir, 'Band1'),
        'band2': os.path.join(test_dir, 'Band2'),
        'band3': os.path.join(test_dir, 'Band3'),
        'band4': os.path.join(test_dir, 'Band4'),
        'band5': os.path.join(test_dir, 'Band5'),
    }
    test_label_folder = os.path.join(test_dir, 'Actual')

    # train_and_save_model(train_imagepath, train_label_folder, weights_file)
    
    load_and_evaluate_model(test_imagepath, test_label_folder, weights_file)
