import cv2
import numpy as np
import dlib
from imutils import face_utils
import streamlit as st

import time

from pygame import mixer



st.title("Driver Drowsiness Detection")
st.write("Whether it's due to medication, a sleep disorder or a poor night's rest, new research points to the risks and potential dangers of drowsy driving.Missing one to two hours of the recommended seven hours of sleep a night nearly doubles the risk of a car accident.Sleepiness can come without warning, so drivers should prioritize getting enough sleep and avoid driving when they are fatigued.")

st.write("The best drivers are aware that they must be beware.")
st.write("Click on Run to Open WEBCAM")

st.sidebar.header("About")
st.sidebar.write("This Application uses computer vision techniques to detect driver drowsiness.It detects the drowsiness of drivers by assessing the drowsy eyes and produces custom sound alarms based on the drowsy condition.")

mixer.init()
sound = mixer.Sound('alarm.wav')




run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0
status = ""
color = (0,0,0)

Ear_Values = []
Mar_Value = []
Time_value = []

def compute(ptA,ptB):
    dist = np.linalg.norm(ptA-ptB)
    return dist

def blinked(a,b,c,d,e,f):
    up = compute(b,d)+compute(c,e)
    down = compute(a,f)
    ratio = up/(2.0*down)

    if(ratio>0.25):
        return 2,ratio
    elif(ratio>0.18 and ratio<=0.25):
        return 1,ratio
    else:
        return 0,ratio

def yawn(a,b,c,d,e,f,g,h):
    den = compute(a,e)
    num = compute(b,h)+compute(c,g)+compute(d,f)
    ratio=num/(3.0*den)

    Mar_Value.append(ratio)

    if(ratio>=0.40):
        return 2
    else:
        return 0

start_time = time.time()


while run:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    

    faces = detector(frame)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        #cv2.rectangle(face_frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        landmarks = predictor(frame,face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink,ratioL = blinked(landmarks[36],landmarks[37],landmarks[38],landmarks[41],landmarks[40],landmarks[39])

        right_blink,ratioR = blinked(landmarks[42],landmarks[43],landmarks[44],landmarks[47],landmarks[46],landmarks[45])

        yawned = yawn(landmarks[60],landmarks[61],landmarks[62],landmarks[63],landmarks[64],landmarks[65],landmarks[66],landmarks[67])


        ratioF=(ratioL+ratioR)/2
        Ear_Values.append(ratioF)
        temp_time = time.time()
        Time_value.append(temp_time-start_time)


        if(left_blink==0 or right_blink==0):
            sleep+=1
            drowsy = 0
            active=0
            if(sleep>6):
                status="SLEEPING!!"
                color = (255,0,0)
                try:
                    sound.play()
                except:
                    pass
        
        elif(left_blink==1 or right_blink==1):
            sleep=0
            active=0
            drowsy+=1
            if(drowsy>6):
                status = "Drowsy !!"
                color = (0,0,255)
                try:
                    mixer.music.load('warning_sleep.mp3')
                except:
                    pass

        elif(yawned==2):
            sleep=0
            active=0
            drowsy+=1
            if(drowsy>6):
                status="Drowsy Yawned!!"
                color = (0,0,255)
                try:
                    mixer.music.load('warning_yawn.mp3')
                    mixer.music.play(1,0.0)
                except:
                    pass
        
        else:
            drowsy=0
            sleep=0
            active+=1
            if(active>6):
                status="Active"
                color = (0,255,0)
        
        cv2.putText(frame,status,(100,100),cv2.FONT_HERSHEY_SIMPLEX,1.2,color,3)

        # for n in range (0,68):
        #     (x,y) = landmarks[n]
        #     cv2.circle(frame,(x,y),1,(255,255,255),-1)

        
        
        cv2.imshow("Frame",frame)
        FRAME_WINDOW.image(frame)
        #cv2.imshow("Result of detector",face_frame)
        key = cv2.waitKey(1)
        if key==27:
            break
        

else:
    st.write('Stopped')


st.subheader(' ------------------------Created By :  Priyanshu harsh ---------------------- :')

# import cv2
# import streamlit as st

# st.title("Webcam Live Feed")
# run = st.checkbox('Run')
# FRAME_WINDOW = st.image([])
# camera = cv2.VideoCapture(0)

# while run:
#     _, frame = camera.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     FRAME_WINDOW.image(frame)
# else:
#     st.write('Stopped')