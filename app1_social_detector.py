import cv2
import streamlit as st
import numpy as np
import imutils
import time
import cv2
import os
import math

from itertools import chain 
from constants import *
from madblocks import *

LABELS = open(YOLOV3_LABELS_PATH).read().strip().split('\n')

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

print('Loading YOLO from disk...')

neural_net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG_PATH, YOLOV3_WEIGHTS_PATH)
layer_names = neural_net.getLayerNames()
layer_names = [layer_names[i-1] for i in neural_net.getUnconnectedOutLayers()]

def app():
    st.title("Social Distance Detector")
    run = st.checkbox('Run Camera')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    writer = None
    (W, H) = (None, None)

    while run:
        _, frame = camera.read()
        dummy=frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if W is None or H is None:
            H, W = (frame.shape[0], frame.shape[1])

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        neural_net.setInput(blob)

        start_time = time.time()
        layer_outputs = neural_net.forward(layer_names)
        end_time = time.time()
    
        boxes = []
        confidences = []
        classIDs = []
        lines = []
        box_centers = []

        for output in layer_outputs:
            for detection in output:
            
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
            
                if confidence > 0.5 and classID == 0:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype('int')
                
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                
                    box_centers = [centerX, centerY]

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
        if len(idxs) > 0:
            unsafe = []
            count = 0
        
            for i in idxs.flatten():
            
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                centeriX = boxes[i][0] + (boxes[i][2] // 2)
                centeriY = boxes[i][1] + (boxes[i][3] // 2)

                color = [int(c) for c in COLORS[classIDs[i]]]
                text = '{}: {:.4f}'.format(LABELS[classIDs[i]], confidences[i])

                idxs_copy = list(idxs.flatten())
                idxs_copy.remove(i)

                for j in np.array(idxs_copy):
                    centerjX = boxes[j][0] + (boxes[j][2] // 2)
                    centerjY = boxes[j][1] + (boxes[j][3] // 2)

                    distance = math.sqrt(math.pow(centerjX - centeriX, 2) + math.pow(centerjY - centeriY, 2))
                    print(distance)
                    email_flag=0
                    if distance <= SAFE_DISTANCE:
                        cv2.imwrite('demo.png',dummy)
                        send_email_with_attachment(username="saimadhav.alert@gmail.com",password="m@keskilled",receiver="parvathanenimadhu@gmail.com",subject="Social Distance Alert",body="This alert is generated automatically",imagefile="demo.png")
                        cv2.line(frame, (boxes[i][0] + (boxes[i][2] // 2), boxes[i][1]  + (boxes[i][3] // 2)), (boxes[j][0] + (boxes[j][2] // 2), boxes[j][1] + (boxes[j][3] // 2)), (0, 0, 255), 2)
                        unsafe.append([centerjX, centerjY])
                        unsafe.append([centeriX, centeriY])

                if centeriX in chain(*unsafe) and centeriY in chain(*unsafe):
                    count += 1
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (50, 50), (450, 90), (0, 0, 0), -1)
                cv2.putText(frame, 'No. of people unsafe: {}'.format(count), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)


        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30,(frame.shape[1], frame.shape[0]), True)
        
        FRAME_WINDOW.image(frame)
        writer.write(frame)
    else:
        st.write('Stopped')
        if writer is not None:
            writer.release()