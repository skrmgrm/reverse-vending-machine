######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import RPi.GPIO as GPIO
import pigpio
import drivers

LED_PIN = 18
LED_PIN2 = 12
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.setup(LED_PIN2, GPIO.OUT)
BUTTON_PIN = 27
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)

display = drivers.Lcd()

# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True
        
        
    ################################################ HX711 ################################################################
EMULATE_HX711=False
referenceUnit = -7030.0

if not EMULATE_HX711:
    import RPi.GPIO as GPIO
    from hx711 import HX711
else:
    from emulated_hx711 import HX711

hx = HX711(4, 17)
hx.set_reading_format("MSB", "MSB")

hx.set_reference_unit(referenceUnit)

hx.reset()

hx.tare()

print("Tare done! Add weight now...")

################################################ ENTRANCE SERVO #####################################################################
#entrance_servo = Servo(24)
servo = 24
#coin = 9
coin2=9
#entrance_servo.value = -0.8
#time.sleep(2)
pwm = pigpio.pi() 
pwm.set_mode(servo, pigpio.OUTPUT)

coin = pigpio.pi() 
coin.set_mode(coin2, pigpio.OUTPUT)
 
pwm.set_PWM_frequency( servo, 50 )
coin.set_PWM_frequency( coin2, 50 )
 


################################################ SORTER AND COIN SERVO #####################################################################
#GPIO.setmode(GPIO.BCM)
GPIO.setup(10, GPIO.OUT)
#GPIO.setup(9, GPIO.OUT)
#GPIO.setup(24, GPIO.OUT)
#GPIO.setup(10, GPIO.OUT)

sorter=GPIO.PWM(10, 50)
#entrance=GPIO.PWM(24, 50)
#coin=GPIO.PWM(9, 50)

sorter.start(0)
#entrance.start(0)
#coin.start(0)



################################################ FUNCTIONS ################################################################################

# HX711
def weight_sensor():
    val = hx.get_weight(5)
    absolute=abs(round(val*10)/10)
    #print(absolute, val)
    hx.power_down()
    hx.power_up()
    return absolute

# ENTRANCE SERVO
def open_servo():
    entrance_servo.value = 0.5
    time.sleep(1)
    
def close_servo():
    entrance_servo.value = -1
    time.sleep(1)

# SORTER SERVO
def sorterr(angle):
    duty = angle / 18 + 2
    GPIO.output(10, True)
    sorter.ChangeDutyCycle(duty)
    time.sleep(1)
    GPIO.output(10, False)
    sorter.ChangeDutyCycle(0)

def coinn(angle):
    duty = angle / 18 + 2
    GPIO.output(9, True)
    coin.ChangeDutyCycle(duty)
    time.sleep(1)
    #GPIO.output(9, False)
    #coin.ChangeDutyCycle(0)

def entrancee(angle):
    pwm.set_servo_pulsewidth( servo, angle ) ;
    time.sleep(1)

def coins(angle):
    coin.set_servo_pulsewidth( coin2, angle ) ;
    time.sleep(1)
    
display.lcd_display_string("Welcome", 1)
entrancee(1100)
sorterr(75)
#coinn(80)
count = 5
coin_count = 5
time.sleep(1)
display.lcd_clear() 

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()



#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    display.lcd_display_string("Insert Bottle", 1)
    display.lcd_display_string("Bottle Count:"+str(count), 2)
    time.sleep(1)
    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            #cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
            weight = weight_sensor()
            print(weight)
            if object_name == 'bottle' and weight>0.0 and weight<3.0:
                GPIO.output(LED_PIN, GPIO.HIGH)
                GPIO.output(LED_PIN2, GPIO.LOW)
                entrancee(1900)
                entrancee(1200)
                #cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                #cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                #cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                #display.lcd_clear()
                #display.lcd_display_string("Bottle",1)
                
                sorterr(20)
                sorterr(75)
                GPIO.output(LED_PIN, GPIO.LOW)
                
                count+=1
                display.lcd_display_string(              str(count), 2)

                
            elif object_name != 'bottle' and weight>0.0 or weight>3.0:
                GPIO.output(LED_PIN2, GPIO.HIGH)
                GPIO.output(LED_PIN, GPIO.LOW)
                entrancee(1900)
                entrancee(1200)
                #cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0, 0, 255), 2)
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                #cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                #cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                #display.lcd_clear()
                #display.lcd_display_string("NOT Bottle",1)
                
                sorterr(150)
                sorterr(75)
                GPIO.output(LED_PIN2, GPIO.LOW)
                
           

    # Draw framerate in corner of frame
    #cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Plastic Bottle Detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
    elif GPIO.input(BUTTON_PIN) and count == 0:
        display.lcd_clear()
        display.lcd_display_string("Insert Bottle", 1)
        display.lcd_display_string("First", 2)
        time.sleep(2)
        display.lcd_clear()
        
    elif GPIO.input(BUTTON_PIN) and count!=0:
       # print(count)
       display.lcd_clear()
       display.lcd_display_string("Thank You", 1)
       display.lcd_display_string("For Recycling", 2)
       for i in range(0,count):
            print(i)
            coins(500)
            coins(1500)
       count = 0
       time.sleep(2)
       display.lcd_clear()

        
# Clean up
cv2.destroyAllWindows()
videostream.stop()


