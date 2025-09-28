import os
import re
import numpy as np
from PIL import Image
from sklearn.metrics import matthews_corrcoef
import tensorflow as tf
from tensorflow.keras import layers, models

def extract_tile_id(filename):
    match = re.search(r'(\d{2}_\d{2})', filename)
    if match:
        return match.group(1)
    return None

def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(num_filters, 3, padding='same', activation='relu')(x)
    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = layers.concatenate([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def create_unet(input_shape=(512, 512, 5)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(d4)

    model = models.Model(inputs, outputs, name="U-Net")
    return model

# The other functions remain the same except some modifications to handle 2D input/output

def normalize_band_data(arr):
    arr = arr.astype(np.float32)
    mean = np.mean(arr)
    std = np.std(arr)
    if std > 0:
        return (arr - mean) / std
    return arr

def load_files_map(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.tif')]
    tile_map = {}
    for f in files:
        tile_id = extract_tile_id(f)
        if tile_id:
            tile_map[tile_id] = f
    return tile_map

def load_band_arrays_2d(imagepath, tile_id, band_file_maps):
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
            arr = arr[..., 0]
        H, W = arr.shape
        arr_normalized = normalize_band_data(arr)
        band_arrays.append(arr_normalized)
    if not band_arrays:
        return None, None, None
    # Stack along last dimension to get (H, W, 5)
    X = np.stack(band_arrays, axis=-1)
    return X, H, W

def load_label_array_2d(label_folder, tile_id, label_file_map):
    if tile_id not in label_file_map:
        return None
    label_path = os.path.join(label_folder, label_file_map[tile_id])
    if not os.path.exists(label_path):
        return None
    label_img = np.array(Image.open(label_path))
    if label_img.ndim == 3:
        label_img = label_img[..., 0]
    label_binary = (label_img > 127).astype(np.uint8)
    return label_binary

def prepare_dataset_2d(imagepath, label_folder):
    band_file_maps = {b: load_files_map(f) for b, f in imagepath.items()}
    label_file_map = load_files_map(label_folder)
    common_tile_ids = set.intersection(*(set(band_file_maps[b].keys()) for b in band_file_maps))
    common_tile_ids = common_tile_ids.intersection(set(label_file_map.keys()))
    tiles = sorted(common_tile_ids)

    all_X, all_y = [], []
    for tile_id in tiles:
        X, H, W = load_band_arrays_2d(imagepath, tile_id, band_file_maps)
        if X is None:
            continue
        y = load_label_array_2d(label_folder, tile_id, label_file_map)
        if y is None or y.shape != X.shape[:2]:
            continue
        # Mask to remove clouds if sum of bands == 0 (assumes cloud pixels)
        cloud_mask = np.sum(X, axis=-1) == 0
        # Filter out cloud pixels from X and y by flattening non-cloud pixels only
        X_flat = X[~cloud_mask]
        y_flat = y[~cloud_mask]
        if X_flat.shape[0] == 0:
            continue
        all_X.append(X_flat)
        all_y.append(y_flat)

    if not all_X:
        raise ValueError("No valid data found.")

    # Instead of flattening all data into 1D, we will keep 2D patches for U-Net
    # Here, we load full images without flattening
    # So return lists of arrays to feed directly to the model as batches

    # Alternatively, stack all 2D tiles in batch dimension for training
    X_array = []
    y_array = []
    for tile_id in tiles:
        X_img, H, W = load_band_arrays_2d(imagepath, tile_id, band_file_maps)
        y_img = load_label_array_2d(label_folder, tile_id, label_file_map)
        if X_img is None or y_img is None:
            continue
        X_array.append(X_img)
        y_array.append(y_img[..., np.newaxis])  # add channel dim to labels

    return np.array(X_array), np.array(y_array)

def train_and_save_model(train_imagepath, train_label_folder, weights_path):
    print("Preparing training data...")
    X_train, y_train = prepare_dataset_2d(train_imagepath, train_label_folder)
    print(f"Training samples: {X_train.shape[0]} with image size {X_train.shape[1:]}")

    model = create_unet(input_shape=X_train.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=1)
    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")

def load_and_evaluate_model(test_imagepath, test_label_folder, weights_path):
    print("Preparing test data...")
    X_test, y_test = prepare_dataset_2d(test_imagepath, test_label_folder)
    print(f"Test samples: {X_test.shape[0]} with image size {X_test.shape[1:]}")

    model = create_unet(input_shape=X_test.shape[1:])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(weights_path)

    y_pred_prob = model.predict(X_test).squeeze()  # shape (num_samples, H, W)
    y_pred = (y_pred_prob > 0.5).astype(np.uint8)

    # Flatten for MCC calculation
    mcc = matthews_corrcoef(y_test.flatten(), y_pred.flatten())
    print(f"MCC on test  {mcc}")
    return mcc

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
