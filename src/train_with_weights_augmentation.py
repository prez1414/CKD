# src/train_with_weights_augmentation.py
import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# Optional: CLAHE preprocessing using OpenCV
def clahe_preprocess(img):
    # img: numpy array (H, W, 3), dtype float32 or uint8
    try:
        import cv2
    except Exception:
        return img  # if cv2 missing, just return original

    # Ensure uint8
    arr = img.astype("uint8")
    # convert to LAB color space, apply CLAHE to L channel
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return rgb

# ---------------- CONFIG ----------------
DATA_DIR = "data/processed"       # expects train/ val/ test/ inside
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 256        # used successfully before
BATCH = 24
EPOCHS_HEAD = 8
EPOCHS_FINETUNE = 12
UNFREEZE_LAST_N = 60
BASE_LR = 1e-4
FT_LR = 1e-5
MODEL_NAME = "kidney_resnet_classweight_aug.keras"

# ---------------- AUGMENTATIONS ----------------
train_datagen = ImageDataGenerator(
    preprocessing_function=clahe_preprocess,   # apply CLAHE (helps contrast)
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.08,
    zoom_range=0.18,
    horizontal_flip=True,
    brightness_range=(0.75, 1.25),
    fill_mode="reflect"
)
val_datagen = ImageDataGenerator(preprocessing_function=clahe_preprocess)

# Generators
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, class_mode="categorical", shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH, class_mode="categorical", shuffle=False
)

# ---------------- CLASS WEIGHTS ----------------
y_train = train_gen.classes            # numeric labels from directory
class_ids = np.unique(y_train)
cw = compute_class_weight(class_weight="balanced", classes=class_ids, y=y_train)
class_weights = dict(enumerate(cw))
print("Computed class weights:", class_weights)

# ---------------- MODEL (ResNet50 head) ----------------
def build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=None):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers:
        layer.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)
    return model

num_classes = len(train_gen.class_indices)
model = build_model(num_classes=num_classes)
model.compile(optimizer=Adam(learning_rate=BASE_LR), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------- CALLBACKS ----------------
checkpoint = ModelCheckpoint(os.path.join(OUT_DIR, MODEL_NAME), monitor="val_accuracy", save_best_only=True, mode="max")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
earlystop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

# ---------------- PHASE A: train head ----------------
print("=== PHASE A: training head with class weights + CLAHE augmentations ===")
history_a = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, earlystop]
)

# ---------------- PHASE B: unfreeze and fine-tune ----------------
print("=== PHASE B: unfreezing last layers and fine-tuning ===")
# reload best model
model = load_model(os.path.join(OUT_DIR, MODEL_NAME))
# unfreeze last N layers
count = 0
for layer in reversed(model.layers):
    if count < UNFREEZE_LAST_N:
        layer.trainable = True
        count += 1
    else:
        break
print(f"Unfroze last {count} layers. Recompiling with lr={FT_LR}")
model.compile(optimizer=Adam(learning_rate=FT_LR), loss="categorical_crossentropy", metrics=["accuracy"])

history_b = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, earlystop]
)

print("Training finished. Best model saved to:", os.path.join(OUT_DIR, MODEL_NAME))
