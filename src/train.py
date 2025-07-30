import os, joblib, cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

IMG_SIZE = 64
data_path = "data"

# Load images and labels
X, y = [], []
labels = sorted(os.listdir(data_path))
for label in labels:
    folder = os.path.join(data_path, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label)

X = np.array(X).reshape(len(X), -1) / 255.0
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply PCA
print("🔍 Applying PCA for SVM and Naive Bayes...")
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Save PCA
os.makedirs("saved_models", exist_ok=True)
joblib.dump(pca, "saved_models/pca.joblib")

# Models to train
models = {
    "svm": LinearSVC(max_iter=10000),
    "naive_bayes": GaussianNB()
    # "knn": KNeighborsClassifier(),
    # "log_reg": LogisticRegression(max_iter=1000),
    # "decision_tree": DecisionTreeClassifier(),
    # "random_forest": RandomForestClassifier()
}

# Train and save models
for name, model in models.items():
    print(f"🔁 Training {name}...")
    model.fit(X_train_pca, y_train)
    acc = model.score(X_test_pca, y_test)
    print(f"✅ {name} Accuracy: {acc:.4f}")
    joblib.dump(model, f"saved_models/{name}.joblib")

# Save Label Encoder
joblib.dump(le, "saved_models/label_encoder.joblib")
print("🎉 Training complete. Models and PCA saved!")
