import cv2
import streamlit as st
import numpy as np
from madblocks import *

statusFlag=0
thresholdFire=20000

def app():
    global statusFlag,thresholdFire
    st.title("Fire Detector")
    run = st.checkbox('Run Camera')
    FRAME_WINDOW = st.image([])
    #FRAME_WINDOW1=st.text_area()
    a='Fire Not Detected'
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        dummy=frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        FRAME_WINDOW.image(frame)
        blur = cv2.GaussianBlur(frame, (21, 21), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
 
        lower = [18, 50, 50]
        upper = [35, 255, 255]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(frame, hsv, mask=mask)
        no_red = cv2.countNonZero(mask)
        #cv2.imshow("output", output)
        #print("output:", no_red)
        if int(no_red) > thresholdFire and statusFlag==0:
            st.write('Fire detected')
            statusFlag=1
            cv2.imwrite('fire.png',dummy)
            send_email_with_attachment(username="saimadhav.alert@gmail.com",password="m@keskilled",receiver="parvathanenimadhu@gmail.com",subject="Fire Alert",body="This alert is generated automatically",imagefile="fire.png")
        if int(no_red)<thresholdFire and statusFlag==1:
            statusFlag=0
            st.write('Fire Not Detected')     
    else:
        st.write('Stopped')