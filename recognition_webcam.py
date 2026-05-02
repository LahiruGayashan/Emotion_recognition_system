# Coded by:- Lahiru Gayashan

import pyttsx3
import cv2
import label_image
import os
import time
import urllib.request
import imghdr

# ================= VOICE ENGINE =================
engine = pyttsx3.init('sapi5')
engine.setProperty('rate', 135)
engine.setProperty('volume', 1.0)

def speak_emotion(emotion):
    print("Speaking:", emotion)
    engine.say(f"You look {emotion}")
    engine.runAndWait()

# ================= SETTINGS =================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
size = 4
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# ================= INPUT SELECTION =================
print("\nSelect Camera Source")
print("A - Laptop / USB Webcam")
print("B - Web URL Image (jpg/png)")
print("C - Mobile IP Webcam")

choice = input("Enter option (A / B / C): ").strip().upper()

# ====================================================
# OPTION B : WEB URL IMAGE (✅ FIXED)
# ====================================================
if choice == "B":
    image_url = input("Enter IMAGE URL (jpg/png): ")

    try:
        print("Downloading image...")

        req = urllib.request.Request(
            image_url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urllib.request.urlopen(req) as response:
            image_data = response.read()

        with open("web_image.jpg", "wb") as f:
            f.write(image_data)

        img_type = imghdr.what("web_image.jpg")
        if img_type not in ["jpeg", "png"]:
            print("❌ Not a valid JPG or PNG image")
            exit()

        frame = cv2.imread("web_image.jpg")
        if frame is None:
            print("❌ Image could not be read")
            exit()

        emotion, confidence = label_image.predict_with_confidence("web_image.jpg")
        emotion = emotion.title()

        print(f"Detected Emotion: {emotion} ({confidence:.2f})")

        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 0),
            3
        )

        cv2.imshow("Emotion Recognition - Web Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()

    except Exception as e:
        print("❌ Failed to process image:", e)
        exit()

# ====================================================
# OPTION A & C : VIDEO SOURCES
# ====================================================
if choice == "A":
    print("✅ Using USB Webcam")
    webcam = cv2.VideoCapture(0)

elif choice == "C":
    ip = input("Enter Mobile IP Webcam URL: ")
    webcam = cv2.VideoCapture(ip, cv2.CAP_FFMPEG)

else:
    print("❌ Invalid option")
    exit()

# ================= SAFETY CHECK =================
if not webcam.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ Camera connected successfully")

# ================= MAIN LOOP =================
last_spoken_time = 0
SPEAK_INTERVAL = 7

while True:
    ret, frame = webcam.read()
    if not ret or frame is None:
        continue

    frame = cv2.flip(frame, 1)

    mini = cv2.resize(
        frame,
        (frame.shape[1] // size, frame.shape[0] // size)
    )

    faces = classifier.detectMultiScale(mini)

    for f in faces:
        (x, y, w, h) = [v * size for v in f]

        padding = 20
        face = frame[
            max(0, y - padding):y + h + padding,
            max(0, x - padding):x + w + padding
        ]

        cv2.imwrite("test.jpg", face)

        emotion, confidence = label_image.predict_with_confidence("test.jpg")
        emotion = emotion.title()

        if confidence > 0.6 and time.time() - last_spoken_time > SPEAK_INTERVAL:
            speak_emotion(emotion)
            last_spoken_time = time.time()

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.2f})",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Recognition System", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()
