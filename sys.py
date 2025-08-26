import tensorflow as tf
from keras import layers, models, callbacks
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont
import os
from sklearn.utils import shuffle
from collections import Counter
import random

# Reproducibility
random.seed(42)
np.random.seed(42)

# ----------------------------
# 1. Build CNN Model
# ----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(15, activation="softmax")  # 10 digits + 5 operators
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# # ----------------------------
# 6. Load Balanced MNIST Digits
# ----------------------------
(X_train_dig, y_train_dig), _ = tf.keras.datasets.mnist.load_data()

# Normalize
X_train_dig = X_train_dig.astype('float32') / 255.0
X_train_dig = X_train_dig.reshape(-1, 28, 28, 1)

# Shuffle
X_combined, y_combined = shuffle(X_train_dig, y_train_dig, random_state=42)

# Train/Val/Test split
train_size = int(0.8 * len(X_combined))
val_size = int(0.1 * len(X_combined))
X_train, y_train = X_combined[:train_size], y_combined[:train_size]
X_val, y_val = X_combined[train_size:train_size+val_size], y_combined[train_size:train_size+val_size]
X_test, y_test = X_combined[train_size+val_size:], y_combined[train_size+val_size:]

print("Train/Val/Test shapes:", X_train.shape, X_val.shape, X_test.shape)

# ----------------------------
# 8. Train Model
# ----------------------------
earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[earlystopping]
)

# ----------------------------
# 9. Save Model
# ----------------------------
model.save("digit_operator_model.h5")
print("✅ Model trained and saved as 'digit_operator_model.h5'")
