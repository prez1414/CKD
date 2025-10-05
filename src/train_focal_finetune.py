# src/train_focal_finetune.py
import os, math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

# ---- focal loss implementation for categorical (softmax) ----
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    gamma = float(gamma)
    alpha = float(alpha)
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = - y_true * tf.math.log(y_pred)
        weight = alpha * tf.math.pow(1 - y_pred, gamma)
        loss_val = weight * cross_entropy
        return tf.reduce_sum(loss_val, axis=1)
    return loss

# CONFIG
BASE_MODEL_PATH = "models/kidney_resnet_finetuned.keras"  # last-good model
OUT_MODEL_PATH = "models/kidney_focal_finetuned.keras"
IMG_SIZE = 256
BATCH = 24
EPOCHS = 18
UNFREEZE_LAST_N = 60
TRAIN_DIR = "data/processed/train"
VAL_DIR = "data/processed/val"

# data generators (mild augmentation)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1),
    fill_mode='reflect'
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH, class_mode="categorical", shuffle=True)
val_gen = val_datagen.flow_from_directory(VAL_DIR, target_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH, class_mode="categorical", shuffle=False)

# load baseline model (no custom objects)
print("Loading base model:", BASE_MODEL_PATH)
base = load_model(BASE_MODEL_PATH)

# unfreeze last N layers
count = 0
for layer in reversed(base.layers):
    if count < UNFREEZE_LAST_N:
        layer.trainable = True
        count += 1
    else:
        break
print("Unfroze last", count, "layers.")

# compile with focal loss
loss_fn = categorical_focal_loss(gamma=2.0, alpha=0.25)
base.compile(optimizer=Adam(learning_rate=1e-5), loss=loss_fn, metrics=["accuracy"])

# callbacks
checkpoint = ModelCheckpoint("models/_tmp_focal_best.h5", monitor="val_accuracy", save_best_only=True, mode="max")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
earlystop = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

# train
base.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, earlystop]
)

# After training, save weights and also save a Keras-loadable model (no custom loss)
# 1) save weights
weights_path = "models/kidney_focal_weights.h5"
base.save_weights(weights_path)
print("Saved weights to", weights_path)

# 2) rebuild same architecture using a fresh ResNet head (no custom loss), load weights and save model in standard format
def build_resnet_head(input_shape=(IMG_SIZE,IMG_SIZE,3), num_classes=None):
    base_net = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base_net.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_net.input, outputs=out)
    return model

num_classes = len(train_gen.class_indices)
final_model = build_resnet_head(num_classes=num_classes)
# load weights by name where possible
# Note: models built differently may not match exactly if original base had custom top; if names differ, try load_weights with by_name=True
try:
    final_model.load_weights(weights_path)
except Exception as e:
    print("Direct weight load failed, trying by_name:", e)
    final_model.load_weights(weights_path, by_name=True)
# compile with standard categorical crossentropy for evaluation
final_model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])
final_model.save(OUT_MODEL_PATH)
print("Saved final model (no custom loss) to:", OUT_MODEL_PATH)
