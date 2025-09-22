# pip install numpy pillow tqdm tensorflow scikit-learn tifffile

import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import matthews_corrcoef
from tifffile import imwrite



def conv_block(x, filters, kernel_size=3, padding="same", activation="relu"):
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x

def unet_plus_plus(input_shape=(128, 128, 5), num_classes=1):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x00 = conv_block(inputs, 32)
    x10 = conv_block(layers.MaxPooling2D()(x00), 64)
    x20 = conv_block(layers.MaxPooling2D()(x10), 128)
    x30 = conv_block(layers.MaxPooling2D()(x20), 256)
    x40 = conv_block(layers.MaxPooling2D()(x30), 512)

    # Decoder with nested skip connections
    x01 = conv_block(layers.Concatenate()([
        layers.UpSampling2D()(x10), x00
    ]), 32)

    x11 = conv_block(layers.Concatenate()([
        layers.UpSampling2D()(x20), x10
    ]), 64)

    x21 = conv_block(layers.Concatenate()([
        layers.UpSampling2D()(x30), x20
    ]), 128)

    x31 = conv_block(layers.Concatenate()([
        layers.UpSampling2D()(x40), x30
    ]), 256)

    x02 = conv_block(layers.Concatenate()([
        layers.UpSampling2D()(x11), x01, x00
    ]), 32)

    x12 = conv_block(layers.Concatenate()([
        layers.UpSampling2D()(x21), x11, x10
    ]), 64)

    x22 = conv_block(layers.Concatenate()([
        layers.UpSampling2D()(x31), x21, x20
    ]), 128)

    x03 = conv_block(layers.Concatenate()([
        layers.UpSampling2D()(x12), x02, x01, x00
    ]), 32)

    # Output layer (sigmoid for binary segmentation)
    output = layers.Conv2D(num_classes, kernel_size=1, activation="sigmoid")(x03)

    model = models.Model(inputs=inputs, outputs=output)
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
    model = unet_plus_plus(input_shape=(128, 128, 5), num_classes=1)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model.fit(
    #     train_images, train_masks,
    #     validation_data=(val_images, val_masks),
    #     batch_size=16,
    #     epochs=50
    # )

    # Save the trained weights
    model.save_weights('model_unetpp.h5')

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

def metrics(true, preds):
    from sklearn.metrics import matthews_corrcoef

    mcc_score = matthews_corrcoef(true, preds)
    print("Matthews Correlation Coefficient:", mcc_score)

