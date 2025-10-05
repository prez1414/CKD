import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 256
OUT_MODEL_PATH = "models/kidney_focal_finetuned.keras"
CHECKPOINT_PATH = "models/_tmp_focal_best.h5"

# Function to rebuild the same ResNet head
def build_resnet_head(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=4):
    base_net = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base_net.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_net.input, outputs=out)
    return model

# Build fresh model
model = build_resnet_head(num_classes=4)

# Load best focal-trained weights
model.load_weights(CHECKPOINT_PATH)

# Compile with normal loss (so evaluation works fine)
model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

# Save as .keras for evaluation
model.save(OUT_MODEL_PATH)
print(f"✅ Exported focal fine-tuned model to {OUT_MODEL_PATH}")
