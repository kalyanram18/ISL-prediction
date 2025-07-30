import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

IMG_SIZE = 64  # size of image (64x64)

def load_images(folder="data"):
    images, labels = [], []
    for class_dir in Path(folder).iterdir():
        if not class_dir.is_dir():
            continue
        label = class_dir.name
        for img_path in class_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img.flatten())  # flatten to 1D
            labels.append(label)
    return np.array(images, dtype=np.float32), np.array(labels)

def train_test_split_80_20(X, y, seed=42):
    return train_test_split(X, y, test_size=0.20, random_state=seed, stratify=y)
