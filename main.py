from flask import Flask, render_template
import cv2
import pyautogui
from camera import VideoCamera
from deepface import DeepFace
import player as p

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/findEmotion/')
def findEmotion():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("test", cv2.WINDOW_FULLSCREEN )
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
            emotion="neutral"
            break
        elif k%256 == 13:
            # ENTER pressed
            cam.release()
            predictions=DeepFace.analyze(frame,actions = ["emotion"])
            cv2.destroyAllWindows()
#            cv2.destroyAllWindows()
            emotion=predictions['dominant_emotion']
            break
    p.MusicPlayer(emotion)
    return emotion

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000', debug='True')