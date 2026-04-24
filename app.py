from flask import Flask, request, jsonify
import requests
import uuid
import os

# Import your prediction function
from label_image import predict

app = Flask(__name__)

UPLOAD_DIR = "api_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict_api():
    data = request.get_json()

    if not data or "image_url" not in data:
        return jsonify({"error": "image_url is required"}), 400

    image_url = data["image_url"]

    # Download image from URL
    r = requests.get(image_url, timeout=10)
    if r.status_code != 200:
        return jsonify({"error": "Failed to download image"}), 400

    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join(UPLOAD_DIR, filename)

    with open(image_path, "wb") as f:
        f.write(r.content)

    try:
        emotion = predict(image_path)
        return jsonify({
            "status": "success",
            "emotion": emotion
        })
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)