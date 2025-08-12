import rasterio
import numpy as np
import os
from sklearn.model_selection import train_test_split
import sys

# Add project root to import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.utils.config import config

# === Load labeled image ===
new_final_class_labels = config.new_final_class_labels
train_mask_path = config.train_mask_path
test_mask_path = config.test_mask_path

with rasterio.open(new_final_class_labels) as src:
    labels = src.read(1)
    profile = src.profile

# === Identify unique classes (excluding background = 0)
unique_classes = np.unique(labels)
unique_classes = unique_classes[unique_classes != 2]

# === Create empty train and test masks
train_mask = np.zeros_like(labels, dtype=np.uint8)
test_mask = np.zeros_like(labels, dtype=np.uint8)

# === For each class, randomly sample 1000 pixels and split into train/test
np.random.seed(42)

for cls in unique_classes:
    # Get all pixel coordinates for this class
    y_cls, x_cls = np.where(labels == cls)
    coords_cls = list(zip(y_cls, x_cls))
    

    # Shuffle and take 1000 samples
    selected_coords = np.array(coords_cls)
    np.random.shuffle(selected_coords)
    selected_coords = selected_coords[:1000]

    # Split into 80% train / 20% test
    coords_train, coords_test = train_test_split(
        selected_coords, test_size=0.2, random_state=42
    )

    # Fill train mask
    for y, x in coords_train:
        train_mask[y, x] = cls

    # Fill test mask
    for y, x in coords_test:
        test_mask[y, x] = cls

# === Save the masks
profile.update(dtype='uint8', count=1, compress='lzw')

with rasterio.open(train_mask_path, "w", **profile) as dst:
    dst.write(train_mask, 1)

with rasterio.open(test_mask_path, "w", **profile) as dst:
    dst.write(test_mask, 1)

print(f"✅ Train and test masks saved with 1000 samples per class across {len(unique_classes)} classes.")