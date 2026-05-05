from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from label_image import predict_with_confidence

app = Flask(__name__)

# ✅ File upload settings
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

# ✅ Load Haar cascade
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

@app.route("/", methods=["GET"])
def home():
    return "Emotion Detection Server Running ✅"

@app.route("/predict", methods=["POST"])
def predict():
    # ✅ ONLY check for image file
    if "image" not in request.files:
        return jsonify({"error": "No image file received"}), 415

    file = request.files["image"]

    image_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    if len(faces) == 0:
        return jsonify({
            "emotion": "No face detected",
            "confidence": 0.0
        })

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face = img[y:y+h, x:x+w]

    face_path = "/tmp/face.jpg"
    cv2.imwrite(face_path, face)

    emotion, confidence = predict_with_confidence(face_path)

    return jsonify({
        "emotion": emotion,
        "confidence": round(float(confidence), 3)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)