# src/evaluate.py
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate(model_path="models/kidney_model.h5", test_dir="data/processed/test", img_size=224, batch_size=32, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    print("📂 Loading model:", model_path)
    model = load_model(model_path)

    datagen = ImageDataGenerator(rescale=1./255)
    gen = datagen.flow_from_directory(test_dir, target_size=(img_size, img_size),
                                      batch_size=batch_size, class_mode='categorical', shuffle=False)

    steps = math.ceil(gen.samples / batch_size)
    print("🔎 Predicting on test set...")
    preds = model.predict(gen, steps=steps, verbose=1)

    y_true = gen.classes
    y_pred = np.argmax(preds, axis=1)

    class_indices = gen.class_indices
    classes = [None] * len(class_indices)
    for k,v in class_indices.items():
        classes[v] = k

    print("\n📊 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    out_cm = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(out_cm)
    print(f"✅ Saved confusion matrix to: {out_cm}")

    # ROC AUC (multi-class)
    try:
        y_true_bin = np.eye(len(classes))[y_true]
        roc = roc_auc_score(y_true_bin, preds, average='macro', multi_class='ovr')
        print("ROC AUC (macro, OVR):", roc)
    except Exception as e:
        print("ROC AUC not available:", e)

    # Save predictions
    np.savez(os.path.join(out_dir, "predictions.npz"), preds=preds, y_true=y_true, y_pred=y_pred, classes=classes)
    print("📁 Saved raw predictions to", os.path.join(out_dir, "predictions.npz"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/kidney_model.h5")
    parser.add_argument("--test_dir", default="data/processed/test")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    evaluate(args.model, args.test_dir, args.img_size, args.batch_size, args.out_dir)
