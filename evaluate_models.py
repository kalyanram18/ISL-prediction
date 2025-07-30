import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
load_model = tf.keras.models.load_model
to_categorical = tf.keras.utils.to_categorical
import cv2

# --- Constants ---
IMG_SIZE = 64
DATA_DIR = "data"
MODEL_DIR = "saved_models"
OUTPUT_CSV = os.path.join(MODEL_DIR, "model_metrics.csv")

# --- Load Dataset ---
print("📦 Loading and processing data...")
X, y = [], []
labels = sorted(os.listdir(DATA_DIR))
label_map = {label: idx for idx, label in enumerate(labels)}

for label in labels:
    folder = os.path.join(DATA_DIR, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label_map[label])

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Flattened input for non-CNN models
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Load PCA if available
pca = None
pca_path = os.path.join(MODEL_DIR, "pca.joblib")
if os.path.exists(pca_path):
    pca = joblib.load(pca_path)

# --- Define models ---
MODELS = {
    "Logistic Regression": "log_reg.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
    "Linear SVM (PCA)": "linear_svc_pca.joblib",
    "CNN": "cnn_model.keras",
}

# --- Evaluate ---
results = []

for model_name, filename in MODELS.items():
    print(f"🔍 Evaluating {model_name}...")
    model_path = os.path.join(MODEL_DIR, filename)

    try:
        if model_name == "CNN":
            model = load_model(model_path)
            y_test_cat = to_categorical(y_test, num_classes=len(labels))
            loss, acc = model.evaluate(X_test, y_test_cat, verbose=0)
            results.append((model_name, round(acc, 4)))
            print(f"✅ {model_name} Accuracy: {acc:.4f}")

        else:
            model = joblib.load(model_path)

            # Apply PCA if required
            if model_name in ["Naive Bayes", "SVM", "Linear SVM (PCA)"]:
                if pca is None:
                    raise FileNotFoundError("❌ PCA transformation required but not found.")
                X_eval = pca.transform(X_test_flat)
            else:
                X_eval = X_test_flat

            preds = model.predict(X_eval)
            acc = accuracy_score(y_test, preds)
            results.append((model_name, round(acc, 4)))
            print(f"✅ {model_name} Accuracy: {acc:.4f}")

    except Exception as e:
        print(f"❌ Failed to evaluate {model_name}: {e}")

# --- Save Results ---
df = pd.DataFrame(results, columns=["Model", "Accuracy"])
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n📁 Accuracy results saved to: {OUTPUT_CSV}")
