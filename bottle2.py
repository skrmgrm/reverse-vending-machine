import serial
import time

from tkinter import *
import numpy as np
import cv2
import serial
import time
from PIL import Image, ImageTk


# Arduino
# arduino = serial.Serial('COM3', 9600)
# time.sleep(2)


def click():
    thres = 0.45
    nms_threshold = 0.2

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # width
    cap.set(4, 480)  # height
    cap.set(10, 150)  # brightness

    not_bottle = 'not bottle'

    classFile = 'coco.names'

    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    # print(classNames)

    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weigthsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weigthsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)

        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        if all(indices):
            print("no bottle")
            # arduino.write(b'0')
            return "0"

        else:

            for i in indices:
                i = i[0]
                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]

                if classNames[classIds[i][0]-1] == 'bottle':

                    cv2.rectangle(img, (x, y), (x+w, h+y),
                                  color=(0, 225, 0), thickness=3)
                    cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confs[0]*100, 2))+"%", (box[0]+10, box[1]+60),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(h)+"-"+str(w), (box[0]+10, box[1]+90),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

                    # arduino.write(b"2")
                    dimension = str(h)+"-"+str(w)
                    # arduino.write(dimension.encode())
                    return dimension

                elif classNames[classIds[i][0]-1] != 'bottle':

                    classNames[classIds[i][0]-1] = not_bottle
                    # print(classNames[classIds[i][0]-1])
                    cv2.rectangle(img, (x, y), (x+w, h+y),
                                  color=(0, 0, 255), thickness=3)
                    cv2.putText(img, classNames[classIds[i][0]-1], (box[0]+10, box[1]+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 225), 2)
                    # arduino.write(b"1")
                    return "1"

        cv2.imshow("press 'q' to stop ", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


startMarker = '<'
endMarker = '>'
dataStarted = False
dataBuf = ""
messageComplete = False


def setupSerial(baudRate, serialPortName):

    global serialPort

    serialPort = serial.Serial(
        port=serialPortName, baudrate=baudRate, timeout=0, rtscts=True)
    print("Serial port " + serialPortName +
          " opened Baudrate " + str(baudRate))

    waitForArduino()


def sendToArduino(stringToSend):
    global startMarker, endMarker, serialPort

    stringWithMarkers = startMarker
    stringWithMarkers += click()
    stringWithMarkers += endMarker

    serialPort.write(stringWithMarkers.encode('utf-8'))


def waitForArduino():
    print("Waiting for Arduino to reset")

    msg = ""

    while msg.find("Arduino is ready") == -1:
        msg = recvLikeArduino()
        if not msg == "XXX":
            print(msg)


def recvLikeArduino():
    global startMarker, endMarker, serialPort, dataStarted, dataBuf, messageComplete

    if serialPort.inWaiting() > 0 and messageComplete == False:
        x = serialPort.read().decode('utf-8')

        if dataStarted == True:
            if x != endMarker:
                dataBuf = dataBuf + x
            else:
                dataStarted = False
                messageComplete = True
        elif x == startMarker:
            dataBuf = ''
            dataStarted = True

    if messageComplete == True:
        messageComplete = False
        return dataBuf
    else:
        return 'XXX'


setupSerial(115200, "com3")
count = 0
prevTime = time.time()
while True:

    arduinoReply = recvLikeArduino()
    if not arduinoReply == 'XXX':
        print("Time %s Reply %s" % (time.time(), arduinoReply))

    if time.time() - prevTime > 1.0:
        sendToArduino("This is the output " + str(count))
        prevTime = time.time()
        count += 1
        # break
