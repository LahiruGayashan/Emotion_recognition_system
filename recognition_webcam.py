# Coded by:- Lahiru Gayashan

import pyttsx3
import cv2
import label_image
import os
import time

engine = pyttsx3.init('sapi5')
engine.setProperty('rate', 135)
engine.setProperty('volume', 1.0)

def speak_emotion(emotion):
    print("Speaking:", emotion)
    engine.say(f"You look {emotion}")
    engine.runAndWait()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

size = 4
classifier = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# ✅ Mobile Phone Camera (IP Webcam)
webcam = cv2.VideoCapture(
    "http://192.168.8.100:8080/video",
    cv2.CAP_FFMPEG
)

if not webcam.isOpened():
    print("❌ IP Webcam stream open වෙන්නේ නැහැ")
    exit()
else:
    print("✅ IP Webcam stream connected")

last_spoken_time = 0
SPEAK_INTERVAL = 7  # seconds

while True:
    rval, im = webcam.read()

    # ✅ Safety check
    if not rval or im is None:
        print("Camera frame not received, skipping frame...")
        time.sleep(0.1)
        continue

    im = cv2.flip(im, 1)

    mini = cv2.resize(
        im,
        (int(im.shape[1] / size), int(im.shape[0] / size))
    )

    faces = classifier.detectMultiScale(mini)

    for f in faces:
        (x, y, w, h) = [v * size for v in f]

        sub_face = im[y:y + h, x:x + w]
        cv2.imwrite("test.jpg", sub_face)

        text = label_image.predict("test.jpg")
        text = text.title()

        current_time = time.time()
        if current_time - last_spoken_time > SPEAK_INTERVAL:
            speak_emotion(text)
            last_spoken_time = current_time

        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 255, 0)

        cv2.rectangle(im, (x, y), (x + w, y + h), color, 3)
        cv2.putText(im, text, (x, y - 10), font, 0.9, color, 2)

    cv2.imshow('Emotion Recognition (Mobile Camera + Voice)', im)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()