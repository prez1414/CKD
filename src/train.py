import argparse
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model



def main(args):
    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(args.data_dir, 'train'),
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    val_generator = val_datagen.flow_from_directory(
        os.path.join(args.data_dir, 'val'),
        target_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        class_mode='categorical'
    )

    # Build model
    model = build_model(num_classes=len(train_generator.class_indices))

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # 🔑 Compute class weights
    train_labels = train_generator.classes  # actual labels from generator
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)

    # Train model with class weights
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        class_weight=class_weights
    )

    # Save model
    os.makedirs(args.out_dir, exist_ok=True)
    model.save(os.path.join(args.out_dir, 'kidney_model.h5'))
    print(f"✅ Model saved to {os.path.join(args.out_dir, 'kidney_model.h5')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
