import cv2
import joblib
import numpy as np
from utils import IMG_SIZE

model = joblib.load("saved_models/knn.joblib")

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    resized = cv2.resize(blurred, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalized = resized / 255.0
    return normalized.flatten().reshape(1, -1)


label_map = {
    "1": "1 (One)", "2": "2 (Two)", "3": "3 (Three)", "4": "4 (Four)",
    "5": "5 (Five)", "6": "6 (Six)", "7": "7 (Seven)", "8": "8 (Eight)", "9": "9 (Nine)"
}
label_map.update({chr(i): chr(i) for i in range(65, 91)})

cap = cv2.VideoCapture(0)
print("🎥 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    roi = frame[100:300, 100:300]
    input_data = preprocess(roi)
    prediction = model.predict(input_data)[0]

    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)
    cv2.putText(frame, f"Predicted: {label_map.get(prediction, prediction)}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Sign Language Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
