# pip install numpy pillow tqdm tensorflow scikit-learn tifffile

import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
from tifffile import imwrite
import argparse


# === Define ANN model in TensorFlow ===
def build_pixel_ann():
    model = models.Sequential(
        [
            layers.Input(shape=(5,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model


def get_tile_id(filename):
    # For files like img001.tif, extract "001"
    match = re.search(r"img(\d+)\.tif", filename)
    if match:
        return match.group(1)  # Returns "001", "002", etc.

    # Alternative pattern for other naming conventions
    match = re.search(r"(\d{2}_\d{2})", filename)
    return match.group(1) if match else None


def maskgeration(imagepath, model_path):
    # Load model
    model = build_pixel_ann()
    model.load_weights("model_tf.h5")  # TensorFlow checkpoint

    def normalize_band(band_data):
        band_data = band_data.astype(np.float32)
        mean = np.mean(band_data)
        std = np.std(band_data)
        if std > 0:
            return (band_data - mean) / std
        return band_data

    # Map band -> tile_id -> filename
    band_tile_map = {band: {} for band in imagepath}
    for band, folder in imagepath.items():
        if not os.path.exists(folder):
            continue

        files = os.listdir(folder)

        for f in files:
            if f.endswith(".tif"):
                tile_id = get_tile_id(f)
                if tile_id:
                    band_tile_map[band][tile_id] = f

    # FIXED: Changed ref_album to ref_band
    ref_band = sorted(imagepath.keys())[0]
    tile_ids = sorted(band_tile_map[ref_band].keys())  # Fixed typo here

    masks = {}

    for tile_id in tile_ids:
        # Collect band arrays in order
        band_arrays = []
        H, W = None, None

        for band_name in sorted(imagepath.keys()):
            if tile_id not in band_tile_map[band_name]:
                continue

            file_path = os.path.join(
                imagepath[band_name], band_tile_map[band_name][tile_id]
            )

            if not os.path.exists(file_path):
                continue

            arr = np.array(Image.open(file_path))
            if arr.ndim == 3:
                arr = arr[..., 0]
            H, W = arr.shape

            # Normalize instead of using scaler
            arr_normalized = normalize_band(arr)
            band_arrays.append(arr_normalized.flatten())

        if not band_arrays:
            continue

        X_test = np.stack(band_arrays, axis=1)

        # Cloud mask
        cloud_mask = X_test.sum(axis=1) == 0

        X_valid = X_test[~cloud_mask]

        if X_valid.shape[0] == 0:
            continue

        probs = model.predict(X_valid).squeeze()
        preds = (probs < 0.5).int().numpy()

        # Reconstruct full mask
        full_mask = np.zeros(H * W, dtype=np.uint8)
        full_mask[~cloud_mask] = preds
        full_mask = full_mask.reshape(H, W) * 255
        masks[tile_id] = full_mask

    return masks
