import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.metrics import f1_score
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import cv2

# Constants
IMG_SIZE = 64
DATA_DIR = "data"
MODEL_DIR = "saved_models"

# Model filenames
MODELS = {
    "Logistic Regression": "log_reg.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
    "CNN": "cnn_model.keras",
}

# Load data
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Load PCA (if needed)
pca = None
pca_path = os.path.join(MODEL_DIR, "pca.joblib")
if os.path.exists(pca_path):
    pca = joblib.load(pca_path)

# F1 scores will be stored here
f1_results = {}

# Evaluate each model
for model_name, filename in MODELS.items():
    print(f"🔍 Evaluating F1 for {model_name}...")

    try:
        model_path = os.path.join(MODEL_DIR, filename)

        if model_name == "CNN":
            model = load_model(model_path)
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)

        else:
            model = joblib.load(model_path)
            X_test_flat = X_test.reshape(len(X_test), -1)

            if model_name in ["Linear SVM (PCA)", "Naive Bayes"] and pca:
                X_test_flat = pca.transform(X_test_flat)

            y_pred = model.predict(X_test_flat)

        score = f1_score(y_test, y_pred, average="weighted")
        f1_results[model_name] = round(score, 4)
        print(f"✅ {model_name} F1 Score: {score:.4f}")

    except Exception as e:
        print(f"❌ Failed to evaluate {model_name}: {e}")

# Save F1 scores to CSV
f1_df = pd.DataFrame(list(f1_results.items()), columns=["Model", "F1 Score"])
f1_df.to_csv(os.path.join(MODEL_DIR, "model_f1_scores.csv"), index=False)
print(f"\n📁 F1 scores saved to: {MODEL_DIR}/model_f1_scores.csv")

# Plot histogram
plt.figure(figsize=(10, 6))
sns.barplot(data=f1_df, x="Model", y="F1 Score", palette="viridis")
plt.title("F1 Scores of Different Models")
plt.ylabel("F1 Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "f1_score_plot.png"))
plt.show()
