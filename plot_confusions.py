import os
import joblib
import numpy as np
import cv2
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model # type: ignore

# --- Constants ---
IMG_SIZE = 64
DATA_DIR = "data"
MODEL_DIR = "saved_models"
CONFUSION_DIR = os.path.join(MODEL_DIR, "confusion_matrices")
os.makedirs(CONFUSION_DIR, exist_ok=True)

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

# Flatten for classical models
X_flat = X.reshape(X.shape[0], -1)

# Load PCA if needed
pca = None
pca_path = os.path.join(MODEL_DIR, "pca.joblib")
if os.path.exists(pca_path):
    pca = joblib.load(pca_path)

# --- Models ---
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

# --- Save confusion matrix ---
def save_conf_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    out_path = os.path.join(CONFUSION_DIR, f"{model_name.replace(' ', '_')}_confusion.png")
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved: {out_path}")

# --- Generate Confusion Matrices ---
for model_name, filename in MODELS.items():
    print(f"📊 Plotting for: {model_name}")
    try:
        path = os.path.join(MODEL_DIR, filename)
        if model_name == "CNN":
            model = load_model(path)
            preds = model.predict(X)
            y_pred = np.argmax(preds, axis=1)

        else:
            model = joblib.load(path)
            input_data = X_flat
            if model_name in ["Naive Bayes", "SVM", "Linear SVM (PCA)"]:
                if pca is None:
                    raise ValueError("❌ PCA required but not found.")
                input_data = pca.transform(X_flat)

            y_pred = model.predict(input_data)

        save_conf_matrix(y, y_pred, model_name)

    except Exception as e:
        print(f"❌ {model_name} failed: {e}")
