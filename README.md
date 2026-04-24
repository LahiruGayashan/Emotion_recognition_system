🎭 Emotion Recognition System
Webcam · Image · API (Postman) · Mobile · Voice Output

📌 Project Overview
This project is a real‑time Emotion Recognition System that detects human facial emotions using Deep Learning (TensorFlow + MobileNet).
✅ Supported Input Methods

💻 Laptop / PC Webcam
🖼 Single Image File
🌐 REST API (Postman)
📱 Mobile Phone Camera (IP Webcam)
🔊 Voice Output (Text‑to‑Speech)

✅ Output

Emotion label (Angry, Happy, Sad, Fear, Disgust, Surprise, Neutral)
Real‑time display
Voice announcement of detected emotion

🧠 Technologies Used

Python 3.10
TensorFlow (TF1 compatibility mode)
OpenCV
Haar Cascade (Face Detection)
MobileNet (Transfer Learning)
Flask (API)
pyttsx3 (Offline Text‑to‑Speech)
Postman (API Testing)

 Project Folder Structure
 Emotion_recognition_system/
│
├── Images/
│   ├── angry/
│   ├── happy/
│   ├── sad/
│   ├── fear/
│   ├── disgust/
│   ├── surprise/
│   └── neutral/
│
├── retrain.py
├── Face_crop.py
├── recognition_webcam.py
├── label_image.py
├── app.py
├── android_recognition.py
│
├── retrained_graph.pb
├── retrained_labels.txt
│
├── haarcascade_frontalface_alt.xml
├── haarcascade_frontalface_default.xml
│
└── README.md

1️⃣ Install Python Packages

pip install tensorflow opencv-python flask requests pyttsx3 numpy

2️⃣ Verify Installations

python --version  
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import cv2; print(cv2.__version__)"


Dataset Preparation
Add images to correct emotion folders:
Images/angry/
Images/happy/
Images/sad/
...

Crop faces (mandatory):

cd Images/angry
python Face_crop.py
cd ..
(Repeat for all emotion folders)


/////////////////////////////////////////////////////////////////////
 Train the Model

 ✅ Recommended Training Command

python retrain.py \
--image_dir=Images \
--output_graph=retrained_graph.pb \
--output_labels=retrained_labels.txt \
--architecture=mobilenet_1.0_224 \
--how_many_training_steps 15000

✅ After training, these files are created:

retrained_graph.pb
retrained_labels.txt

//////////////////////////////////////////////////////////
Test with a Single Image
python label_image.py --image test.jpg
✅ Output Example:
Predicted Emotion: angry
///////////////////////////////////////////////////////////

Webcam Emotion Recognition (With Voice)
python recognition_webcam.py


 ✅ Features

Live face detection
Emotion display
Voice announcement
Emotion spoken only when changed OR every N second


🔊 Voice Output (Text‑to‑Speech)

Fully offline
Uses pyttsx3
Example voice:


"You look happy"

✅ Logic

Emotion‑change based OR
Time‑based (every 5 seconds)

/////////////////////////////////////////////////

 REST API (Postman)
 python app.py   // run this

Server runs at:
http://127.0.0.1:5000

✅ Postman – Image URL JSON
Method: POST --http://127.0.0.1:5000/predict

Body (raw JSON):
{
  "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/37/Portrait_Placeholder.png"
}

✅ Response:

{
  "status": "success",
  "emotion": "angry"
}

📱 Mobile Phone Camera Input
✅ Method 1: IP Webcam (Android)

Install IP Webcam app
Start server → get URL
http://192.168.x.x:8080/video

Edit android_recognition.py:
cv2.VideoCapture("http://192.168.x.x:8080/video")

Run:python android_recognition.py

