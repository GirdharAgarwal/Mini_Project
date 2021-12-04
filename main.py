from flask import Flask, render_template, Response
import pyautogui
import cv2
from camera import VideoCamera
from deepface import DeepFace
import player as p
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/findEmotion/')
def findEmotion():
    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("test")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)
    
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            emotion="no-emotion"
            break
        elif k%256 == 13:
            # ENTER pressed
            cam.release()
            predictions=DeepFace.analyze(frame)
            cv2.destroyAllWindows()
            emotion=predictions['dominant_emotion']
            break
    p.MusicPlayer(emotion)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000', debug='True')