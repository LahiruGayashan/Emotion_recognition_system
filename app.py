from flask import Flask, request, jsonify
from label_image import predict_with_confidence
import requests
import uuid
import os
import cv2

app = Flask(__name__)

UPLOAD_DIR = "api_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

@app.route("/", methods=["GET"])
def home():
    return "Emotion Recognition API is running ✅"

# ✅ Emotion prediction from IMAGE URL WITH FACE DETECTION
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()

    if not data or "image_url" not in data:
        return jsonify({"error": "image_url is required"}), 400

    image_url = data["image_url"]
    image_path = None
    face_path = None

    try:
        # ✅ Download image
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(image_url, headers=headers, timeout=10)

        if r.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400

        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(UPLOAD_DIR, filename)

        with open(image_path, "wb") as f:
            f.write(r.content)

        # ✅ Read image using OpenCV
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ✅ Detect faces
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

        # ✅ Take largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = img[y:y+h, x:x+w]

        face_path = os.path.join(UPLOAD_DIR, f"face_{filename}")
        cv2.imwrite(face_path, face)

        # ✅ Emotion prediction on CROPPED FACE
        emotion, confidence = predict_with_confidence(face_path)

        return jsonify({
            "status": "success",
            "emotion": emotion,
            "confidence": round(float(confidence), 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
        if face_path and os.path.exists(face_path):
            os.remove(face_path)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)