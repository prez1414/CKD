import os
import shutil
import random
from pathlib import Path

def prepare_dataset(raw_dir="data/raw", processed_dir="data/processed", 
                    train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    random.seed(seed)

    classes = ["normal", "cyst", "tumor", "stone"]
    for cls in classes:
        files = list(Path(raw_dir, cls).glob("*.*"))
        random.shuffle(files)

        n_total = len(files)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_files = files[:n_train]
        val_files   = files[n_train:n_train + n_val]
        test_files  = files[n_train + n_val:]

        subsets = [("train", train_files), ("val", val_files), ("test", test_files)]

        for subset, file_list in subsets:
            out_dir = Path(processed_dir, subset, cls)
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in file_list:
                shutil.copy(f, out_dir / f.name)

    print("✅ Dataset prepared and split into train/val/test")

if __name__ == "__main__":
    prepare_dataset()
