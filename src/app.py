import streamlit as st
import os, joblib, cv2, numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
from src.utils import IMG_SIZE

# --- Caching model loaders ---
@st.cache_resource
def load_cnn_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_joblib_model(path):
    return joblib.load(path)

# --- Paths ---
MODEL_DIR = "saved_models"
MODELS = {
    "Logistic Regression": "log_reg.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Decision Tree": "decision_tree.joblib",
    "Random Forest": "random_forest.joblib",
    "SVM": "svm.joblib",
    "CNN": "cnn_model.keras",
}

# Load label encoder
label_encoder = load_joblib_model(os.path.join(MODEL_DIR, "label_encoder.joblib"))

# Load PCA (for specific models)
pca = None
pca_path = os.path.join(MODEL_DIR, "pca.joblib")
if os.path.exists(pca_path):
    pca = load_joblib_model(pca_path)

# --- Explanation Map (used for all models) ---
explanation_map = {
    "M": "⚠️ May be confused with 'N' or 'T' due to similar thumb positions.",
    "N": "⚠️ Can be mistaken for 'M', especially if fingers are tightly closed.",
    "T": "⚠️ Often confused with 'M' or 'N' because of overlapping structure.",
    "D": "⚠️ May resemble 'I' or 'L' depending on angle.",
    "W": "⚠️ Could be misclassified as 'V' if fingers are not fully extended.",
    # Add more if needed
}

# --- Streamlit UI ---
st.set_page_config(page_title="ISL Sign Recognition", layout="centered")
st.title("🤟 ISL Sign Language Predictor")

# Model selection
selected_model_name = st.selectbox("📊 Choose a model", list(MODELS.keys()))
model_path = os.path.join(MODEL_DIR, MODELS[selected_model_name])

# Image upload
uploaded_file = st.file_uploader("📷 Upload a hand sign image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    st.image(image, caption="⬆️ Uploaded Image", width=150)

    # Preprocess
    img_array = np.array(image)
    input_img = (
        img_array.reshape(1, IMG_SIZE, IMG_SIZE, 1) / 255.0
        if selected_model_name == "CNN"
        else img_array.flatten().reshape(1, -1)
    )

    # Apply PCA if needed
    if selected_model_name in ["Linear SVM (PCA)", "Naive Bayes"] and pca:
        input_img = pca.transform(input_img)

    # --- Prediction ---
    try:
        if selected_model_name == "CNN":
            model = load_cnn_model(model_path)
            predictions = model.predict(input_img)[0]
            top5_indices = np.argsort(predictions)[::-1][:5]
            top5_labels = label_encoder.inverse_transform(top5_indices)
            top5_conf = predictions[top5_indices]

        else:
            model = load_joblib_model(model_path)

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_img)[0]
                top5_indices = np.argsort(proba)[::-1][:5]
                top5_labels = label_encoder.inverse_transform(top5_indices)
                top5_conf = proba[top5_indices]
            else:
                predicted_index = model.predict(input_img)[0]
                class_name = label_encoder.inverse_transform([predicted_index])[0]
                st.success(f"✅ Predicted Sign: **{class_name.upper()}**")

                # Show smart explanation
                if class_name.upper() in explanation_map:
                    st.warning(explanation_map[class_name.upper()])

                st.info("ℹ️ This model does not support confidence scores.")
                st.stop()  # Skip rest (no proba chart)

        # Common block for models that return top-5 predictions
        class_name = top5_labels[0]
        st.success(f"✅ Predicted Sign: **{class_name.upper()}**")

        # Show smart explanation
        if class_name.upper() in explanation_map:
            st.warning(explanation_map[class_name.upper()])

        # Plot top-5
        conf_df = pd.DataFrame({
            'Label': top5_labels,
            'Confidence': top5_conf
        })
        st.write("📈 Top 5 Predictions")
        st.bar_chart(conf_df.set_index("Label"))

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")

# === STEP 4: MODEL PERFORMANCE COMPARISON TABLE ===
st.markdown("---")
st.subheader("📊 Model Comparison Summary")

try:
    df_metrics = pd.read_csv(os.path.join(MODEL_DIR, "model_metrics.csv"))
    st.dataframe(df_metrics.set_index("Model"))
except Exception as e:
    st.warning("⚠️ Could not load model comparison table. Please ensure 'model_metrics.csv' exists in the saved_models folder.")