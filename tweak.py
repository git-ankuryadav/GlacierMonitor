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

def build_pixel_ann():
    model = models.Sequential([
        layers.Input(shape=(5,)),
        layers.Dense(16, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model

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

def load_band_arrays(imagepath, tile_id, band_file_maps):
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
        band_arrays.append(arr_normalized.flatten())
    if not band_arrays:
        return None, None, None
    X = np.stack(band_arrays, axis=1)
    return X, H, W

def load_label_array(label_folder, tile_id, label_file_map):
    if tile_id not in label_file_map:
        return None
    label_path = os.path.join(label_folder, label_file_map[tile_id])
    if not os.path.exists(label_path):
        return None
    label_img = np.array(Image.open(label_path))
    if label_img.ndim == 3:
        label_img = label_img[..., 0]
    label_binary = (label_img > 127).astype(np.uint8)
    return label_binary.flatten()

def prepare_dataset(imagepath, label_folder):
    band_file_maps = {}
    for band_name, band_folder in imagepath.items():
        band_file_maps[band_name] = load_files_map(band_folder)
    label_file_map = load_files_map(label_folder)
    common_tile_ids = set.intersection(*[set(band_file_maps[b].keys()) for b in band_file_maps])
    common_tile_ids = common_tile_ids.intersection(set(label_file_map.keys()))
    tiles = sorted(common_tile_ids)

    all_X = []
    all_y = []

    for tile_id in tiles:
        X, H, W = load_band_arrays(imagepath, tile_id, band_file_maps)
        if X is None:
            continue
        y = load_label_array(label_folder, tile_id, label_file_map)
        if y is None or y.shape[0] != X.shape[0]:
            continue
        cloud_mask = X.sum(axis=1) == 0
        X_valid = X[~cloud_mask]
        y_valid = y[~cloud_mask]
        if X_valid.shape[0] == 0:
            continue
        all_X.append(X_valid)
        all_y.append(y_valid)
    if not all_X:
        raise ValueError("No valid data found.")
    X_dataset = np.concatenate(all_X, axis=0)
    y_dataset = np.concatenate(all_y, axis=0)
    return X_dataset, y_dataset

def train_model(train_imagepath, train_label_folder, weights_path):
    print("Preparing training data...")
    X_train, y_train = prepare_dataset(train_imagepath, train_label_folder)
    print(f"Training samples: {X_train.shape[0]}")

    model = build_pixel_ann()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    model.save_weights(weights_path)
    print(f"Weights saved to {weights_path}")
    return model

def predict_and_evaluate(model, test_imagepath, test_label_folder):
    print("Preparing test data...")
    X_test, y_test = prepare_dataset(test_imagepath, test_label_folder)
    print(f"Test samples: {X_test.shape[0]}")

    y_pred_prob = model.predict(X_test).squeeze()
    y_pred = (y_pred_prob > 0.5).astype(np.uint8)

    mcc = matthews_corrcoef(y_test, y_pred)
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

    # model = train_model(train_imagepath, train_label_folder, weights_file)
    model = build_pixel_ann()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights('model_tf.weights.h5')
    predict_and_evaluate(model, test_imagepath, test_label_folder)
