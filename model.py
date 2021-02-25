import numpy as np
import tensorflow as tf
from tensorflow import keras
from setting import setting

lr, epoch, batch, dropout, gpu_num = setting()

def ensemble_models():
    layers = [
        [64, 64, 32, 16, 8],
        [64, 64, 32, 16, 4],
        [64, 64, 32, 8, 4],
        [64, 32, 32, 16, 8],
        [64, 32, 32, 16, 4],
        [64, 32, 32, 8, 4],
        [64, 32, 16, 16, 8],
        [64, 32, 16, 16, 4],
        [64, 16, 16, 8, 4],
        [64, 16, 8, 8, 4]
    ]

    models = []

    for layer in layers:
        models.append(
            keras.Sequential([
                tf.keras.layers.Dense(layer[0], activation = 'relu'),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(layer[1], activation = 'relu'),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(layer[2], activation = 'relu'),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(layer[3], activation = 'relu'),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(layer[4], activation = 'relu'),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(1, activation = 'sigmoid')
            ])
        )
    
    return models

def main_model():
    model = keras.Sequential([
        tf.keras.layers.Dense(64, activation = 'relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(32, activation = 'relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation = 'relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(8, activation = 'relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(4, activation = 'relu'),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])

    return model