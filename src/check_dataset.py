import os
from collections import Counter

# Change these paths if needed
train_dir = "data/processed/train"
val_dir = "data/processed/val"
test_dir = "data/processed/test"

def count_images(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            num_images = sum(len(files) for _, _, files in os.walk(class_path))
            class_counts[class_name] = num_images
    return class_counts

print("📊 Dataset Distribution:\n")

print("Train:", count_images(train_dir))
print("Validation:", count_images(val_dir))
print("Test:", count_images(test_dir))
