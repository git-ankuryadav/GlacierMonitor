# pip install tensorflow numpy pillow tifffile scikit-learn

import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from sklearn.metrics import matthews_corrcoef
import tifffile
import re

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
    # x = layers.Lambda(lambda x: tf.keras.utils.normalize(x, axis=-1))(model_input)
    x = layers.LayerNormalization(axis=-1)(model_input)

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
    # For files like img001.tif, extract "001"
    match = re.search(r"img(\d+)\.tif", filename)
    if match:
        return match.group(1)  # Returns "001", "002", etc.

    # Alternative pattern for other naming conventions
    match = re.search(r"(\d{2}_\d{2})", filename)
    return match.group(1) if match else None

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
def maskgeration(imagepath, model_path):
    """Generate glacier masks using trained DeepLabV3+ model.
    
    Args:
        imagepath (dict): Dictionary mapping band names to folder paths.
        model_path (str): Path to the trained model (.h5).

    Returns:
        dict[str, np.ndarray]: Mapping from tile ID to binary mask array (H×W uint8).
    """
    import os
    import numpy as np
    from PIL import Image
    import tensorflow as tf

    # 1. Load model
    model = tf.keras.models.load_model(model_path, compile=False)

    # 2. Band normalization helper
    def normalize_band(arr):
        arr = arr.astype(np.float32)
        m, s = arr.mean(), arr.std()
        return (arr - m) / s if s > 0 else arr

    # 3. Build band → tile_id → filename map (imagepath is now a dict)
    band_tile_map = {band: {} for band in imagepath}
    for band, folder in imagepath.items():
        if not os.path.exists(folder):
            continue
        files = os.listdir(folder)
        for f in files:
            if f.endswith(".tif"):
                tid = extract_tile_id(f)
                if tid:
                    band_tile_map[band][tid] = f

    # 4. Reference band to enumerate tile IDs
    ref_band = sorted(imagepath.keys())[0]
    tile_ids = sorted(band_tile_map[ref_band].keys())

    masks = {}
    for tid in tile_ids:
        # 5. Load and normalize all 5 bands for this tile
        arrays = []
        H = W = None
        for band_name in sorted(imagepath.keys()):
            if tid not in band_tile_map[band_name]:
                break
            file_path = os.path.join(imagepath[band_name], band_tile_map[band_name][tid])
            if not os.path.exists(file_path):
                break
            arr = np.array(Image.open(file_path))
            if arr.ndim == 3:
                arr = arr[..., 0]
            H, W = arr.shape
            arrays.append(normalize_band(arr))
        else:
            # 6. Stack into H×W×5
            X = np.stack(arrays, axis=-1)
            # 7. Resize to 512×512 if needed
            if (H, W) != (512, 512):
                R = []
                for c in range(5):
                    im = Image.fromarray(X[..., c])
                    im = im.resize((512, 512), Image.BILINEAR)
                    R.append(np.array(im))
                X = np.stack(R, axis=-1)
            # 8. Predict full tile mask
            pred = model.predict(X[np.newaxis], verbose=0)[0, ..., 0]
            mask = (pred > 0.5).astype(np.uint8) * 255
            # 9. Store in dict by tile ID
            masks[tid] = mask

    return masks
