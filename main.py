from flask import Flask, render_template, Response
import pyautogui
from camera import VideoCamera
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/findEmotion')
def findEmotion():
    cam = cv2.VideoCapture(0)
    
    cv2.namedWindow("test")
    
    img_counter = 0
    
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
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
    
    print("tanisha")
    cam.release()
    
    print("girdhar")
    img_counter -=1
    img = cv2.imread("opencv_frame_"+str(img_counter)+".png")
    predictions=DeepFace.analyze(img)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)) ##RGB
    emotion=predictions['dominant_emotion']
    cv2.destroyAllWindows()
    return emotion

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000', debug='True')