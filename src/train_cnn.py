import tensorflow as tf

Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
to_categorical = tf.keras.utils.to_categorical
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

from sklearn.model_selection import train_test_split
import numpy as np
import os
import cv2

# Constants
IMG_SIZE = 64
data_path = "data"

# Load images and labels
X, y = [], []
labels = sorted(os.listdir(data_path))
label_map = {label: idx for idx, label in enumerate(labels)}

print("📷 Loading and processing images...")

for label in labels:
    folder = os.path.join(data_path, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label_map[label])

# Preprocess data
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = to_categorical(np.array(y), num_classes=len(labels))

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Image augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# Build improved CNN model
cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(len(labels), activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with validation and early stopping
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("🧠 Training CNN model...")
# Manually split validation set from training set
from sklearn.model_selection import train_test_split
X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)

# Apply data augmentation only to training data
train_generator = datagen.flow(X_train_new, y_train_new, batch_size=32)

# Train using generator + separate validation data
history = cnn.fit(
    train_generator,
    epochs=30,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# Evaluate on test set
test_loss, test_acc = cnn.evaluate(X_test, y_test)
print(f"📊 Final Test Accuracy: {test_acc:.4f}")

# Save the model
cnn.save("saved_models/cnn_model.h5")
print("✅ CNN training complete and model saved!")
