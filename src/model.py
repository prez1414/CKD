# src/model.py

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(224, 224, 3), num_classes=4):
    # Load pretrained ResNet50 (without top classifier layer)
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)

    # Freeze base model layers (so we only train the classifier first)
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers for CKD classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation="softmax")(x)

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
