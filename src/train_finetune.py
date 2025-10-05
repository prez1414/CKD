# src/train_finetune.py
import os
import math
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

# ---- CONFIG -----------------------------------------------------------------
DATA_DIR = "data/processed"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "val")
OUT_DIR   = "models"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = 256            # increase from 224 if you want more detail (watch memory)
BATCH_SIZE = 24
EPOCHS_HEAD = 8           # phase A
EPOCHS_FINETUNE = 12      # phase B
UNFREEZE_LAST_N = 60      # layers to unfreeze in backbone during finetune
BASE_LR = 1e-4
FINETUNE_LR = 1e-5
MODEL_NAME = "kidney_resnet_finetuned.keras"   # recommended native Keras format

# ---- AUGMENTATIONS (stronger, targeted) ------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.12,
    height_shift_range=0.12,
    shear_range=0.08,
    zoom_range=0.18,
    horizontal_flip=True,
    vertical_flip=False,           # ultrasound/CT often should not flip vertically, tune as needed
    brightness_range=(0.7, 1.3),   # helps intensity differences (cyst vs stone)
    fill_mode="reflect"
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True
)
val_gen = val_datagen.flow_from_directory(
    VAL_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

# ---- CLASS WEIGHTS ----------------------------------------------------------
train_labels = train_gen.classes
class_weights_raw = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
class_weights = dict(enumerate(class_weights_raw))
print("Class weights:", class_weights)

# ---- BUILD MODEL (transfer learning head) -----------------------------------
def build_resnet_head(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=None):
    base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    # freeze backbone for head training
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
model = build_resnet_head(num_classes=num_classes)
model.summary()

# compile head
model.compile(optimizer=Adam(learning_rate=BASE_LR), loss="categorical_crossentropy", metrics=["accuracy"])

# callbacks
checkpoint = ModelCheckpoint(os.path.join(OUT_DIR, MODEL_NAME), monitor="val_accuracy", save_best_only=True, mode="max")
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
earlystop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)

# ---- PHASE A: train head ----------------------------------------------------
print("=== PHASE A: training head (backbone frozen) ===")
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, earlystop]
)

# ---- PHASE B: unfreeze top layers and fine-tune -----------------------------
print("=== PHASE B: unfreeze last layers and fine-tune ===")
# load best saved model from phase A
model = load_model(os.path.join(OUT_DIR, MODEL_NAME))

# find backbone (ResNet) layers and unfreeze last UNFREEZE_LAST_N layers
# ensure we do not unfreeze BatchNorm layers for stability sometimes (optional)
total_layers = len(model.layers)
print("Total layers in loaded model:", total_layers)

# Unfreeze by iterating from end and setting trainable True for last N
count = 0
for layer in reversed(model.layers):
    if count < UNFREEZE_LAST_N:
        layer.trainable = True
        count += 1
    else:
        break
print(f"Unfroze last {UNFREEZE_LAST_N} layers. Recompiling with lr={FINETUNE_LR}")

model.compile(optimizer=Adam(learning_rate=FINETUNE_LR), loss="categorical_crossentropy", metrics=["accuracy"])

history_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINETUNE,
    class_weight=class_weights,
    callbacks=[checkpoint, reduce_lr, earlystop]
)

print("Training complete. Best model saved to:", os.path.join(OUT_DIR, MODEL_NAME))

