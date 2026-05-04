from flask import Flask, request, jsonify
from label_image import predict_with_confidence
import requests
import uuid
import os

app = Flask(__name__)

UPLOAD_DIR = "api_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Health check (Railway needs this)
@app.route("/", methods=["GET"])
def home():
    return "Emotion Recognition API is running ✅"

# ✅ Emotion prediction from IMAGE URL
@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()

    if not data or "image_url" not in data:
        return jsonify({"error": "image_url is required"}), 400

    image_url = data["image_url"]

    try:
        # ✅ Browser-like header to avoid 403
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        r = requests.get(image_url, headers=headers, timeout=10)

        if r.status_code != 200:
            return jsonify({"error": "Failed to download image"}), 400

        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(UPLOAD_DIR, filename)

        with open(image_path, "wb") as f:
            f.write(r.content)

        emotion, confidence = predict_with_confidence(image_path)

        return jsonify({
            "status": "success",
            "emotion": emotion,
            "confidence": round(confidence, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

# ✅ Railway PORT handling
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
