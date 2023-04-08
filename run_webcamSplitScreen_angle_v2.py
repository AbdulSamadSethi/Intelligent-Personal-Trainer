import argparse
import logging
import time
import cv2
import re
import csv
import pandas as pd
import numpy as np
from contextlib import redirect_stdout
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
import math
from dtaidistance import dtw
import PySimpleGUI as sg
import imutils

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def getXY(data):
    x = []
    y = []
    for i in range(len(data)):
        x.append(float(data[i].split(' ')[0]))
        y.append(float(data[i].split(' ')[1]))

    return x, y

def angle3pt(ax,ay, bx,by, cx,cy):
    # Counterclockwise angle in degrees by turning from a to c around b Returns a float between 0.0 and 360.0
    ang = math.degrees(math.atan2(cy-by, cx-bx) - math.atan2(ay-by, ax-bx))
    #ang = ang + 360 if ang < 0 else ang

    return abs(ang)

###---------------------------------------------------------------------------------------------
sg.theme('SystemDefault')
Exercises = ['Shoulder Press', 'Squat', 'Lateral Raises']

# define the window layout
layout = [[sg.Image(filename='', key='image'), sg.Image(filename='', key='image2')],
          [sg.ProgressBar(1000, orientation='h', size=(118, 30), key='progressbar')], 
          [sg.Exit('Exit')]]

# create the window and show it without the plot
window = sg.Window('Demo Application - OpenCV Integration', layout, location=(0, 400), size = (1300, 575))
progress_bar = window['progressbar']

# ---===--- Event LOOP Read and display frames, operate the GUI --- #
cap1 = cv2.VideoCapture('abdultest.mp4')
recording = False
###-----------------------------------------------------------------------------------------FROM SPLITSCREEN

parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
parser.add_argument('--camera', type=int, default=0)

parser.add_argument('--resize', type=str, default='0x0',
                    help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                    help='if provided, resize heatmaps before they are post-processed. default=1.0')

parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
parser.add_argument('--show-process', type=bool, default=False,
                    help='for debug purpose, if enabled, speed for inference is dropped.')

parser.add_argument('--tensorrt', type=str, default="False",
                    help='for tensorrt process.')

parser.add_argument('--output_json', type=str, default="False",
                    help='for json output')
args = parser.parse_args()

logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
w, h = model_wh(args.resize)
if w > 0 and h > 0:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
else:
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
logger.debug('cam read+')
cam = cv2.VideoCapture(args.camera)
ret_val, image = cam.read()
logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))


#clear output file
fields = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 
                'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'Background']

with open('out.csv', 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)             
        csvwriter.writerow(fields)   

df_expert = pd.read_csv('workout_coor/out_shoulderpress_expert.csv')

rshoul_eX, rshoul_eY = getXY(df_expert['RShoulder']) 
relbow_eX, relbow_eY = getXY(df_expert['RElbow'])
rwrist_eX, rwrist_eY = getXY(df_expert['RWrist'])

lshoul_eX, lshoul_eY = getXY(df_expert['LShoulder']) 
lelbow_eX, lelbow_eY = getXY(df_expert['LElbow'])
lwrist_eX, lwrist_eY = getXY(df_expert['LWrist'])

rightArm_eangles = []
leftArm_eangles = []

for i in range(len(rshoul_eY)):
    rightArm_eangles.append(angle3pt(rshoul_eX[i],rshoul_eY[i], relbow_eX[i], relbow_eY[i], rwrist_eX[i], rwrist_eY[i]))

for i in range(len(lshoul_eY)):
    leftArm_eangles.append(angle3pt(lshoul_eX[i],lshoul_eY[i], lelbow_eX[i], lelbow_eY[i], lwrist_eX[i], lwrist_eY[i]))

right_dist = [0]
left_dist = [0]

right_rate = [0]
left_rate = [0]
 

while True:
    event, values = window.read(timeout=20)        

    if event == 'Exit' or event == sg.WIN_CLOSED:
        window.close()
        break

    recording = True

    if recording:
        ret_val, image = cam.read()
        ret, frame1 = cap1.read()
        
        frame1 = imutils.resize(frame1, width=640)

        
        if frame1 is None:
            cap1 = cv2.VideoCapture('abdultest.mp4')
            ret, frame1 = cap1.read()
            frame1 = imutils.resize(frame1, width=640)
        else:
            frame1 = imutils.resize(frame1, width=640)
        

        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        #fields = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
        
        with open('out.csv', 'a') as csvfile: 
            # creating a csv dict writer object 
            csvwriter = csv.writer(csvfile)            
            # writing data rows 
            try:
                body_list = [[str(humans[0].body_parts[0]), str(humans[0].body_parts[1]), str(humans[0].body_parts[2]), 
                    str(humans[0].body_parts[3]), str(humans[0].body_parts[4]), str(humans[0].body_parts[5]), 
                    str(humans[0].body_parts[6]), str(humans[0].body_parts[7]), str(humans[0].body_parts[8]), 
                    str(humans[0].body_parts[9]), str(humans[0].body_parts[10]), str(humans[0].body_parts[11]), 
                    str(humans[0].body_parts[12]), str(humans[0].body_parts[13]), str(humans[0].body_parts[14]), 
                    str(humans[0].body_parts[15]), str(humans[0].body_parts[16]),str(humans[0].body_parts[17])]] 
                
                #check if joints for shoulder press are in frame
                #if all (k in str(humans[0].body_parts.keys()) for k in ('2','3','4','5','6','7')):
                if len(humans[0].body_parts.keys())-1 == 17:
                    csvwriter.writerows(body_list)

                    df_user = pd.read_csv('out.csv')

                    rshoul_uX, rshoul_uY = getXY(df_user['RShoulder']) 
                    relbow_uX, relbow_uY = getXY(df_user['RElbow'])
                    rwrist_uX, rwrist_uY = getXY(df_user['RWrist'])

                    lshoul_uX, lshoul_uY = getXY(df_user['LShoulder']) 
                    lelbow_uX, lelbow_uY = getXY(df_user['LElbow'])
                    lwrist_uX, lwrist_uY = getXY(df_user['LWrist'])

                    rightArm_uangles = []
                    leftArm_uangles = []

                    for i in range(len(rshoul_uY)):
                        rightArm_uangles.append(angle3pt(rshoul_uX[i],rshoul_uY[i], relbow_uX[i], relbow_uY[i], rwrist_uX[i], rwrist_uY[i]))

                    for i in range(len(lshoul_uY)):
                        leftArm_uangles.append(angle3pt(lshoul_uX[i],lshoul_uY[i], lelbow_uX[i], lelbow_uY[i], lwrist_uX[i], lwrist_uY[i]))

                    #keep track of line/frame
                    line = len(leftArm_uangles)
                    print(line)

                    #calculate DTW
                    right_dist.append(dtw.distance(rightArm_uangles[:line], rightArm_eangles[:line]))
                    left_dist.append(dtw.distance(leftArm_uangles[:line], leftArm_eangles[:line]))

                    #calculate rate of change of DTW
                    right_rate.append(right_dist[-1] - right_dist[-2])
                    left_rate.append(left_dist[-1] - left_dist[-2])

                    print((right_dist[-1]+left_dist[-1]))
                    print(right_rate[-1]+left_rate[-1])

                    if ((right_dist[-1]+left_dist[-1]) < 400):
                        print("Amazing!!!")
                    elif((right_dist[-1]+left_dist[-1]) > 400 and (right_dist[-1]+left_dist[-1]) < 1200):
                        print("Average")
                    elif((right_dist[-1]+left_dist[-1]) > 1200):
                        print("BAD!!!")

                    progress_bar.UpdateBar(right_dist[-1]+left_dist[-1])
                    '''
                    with open('out_rateR.txt', 'a') as f1: 
                        with redirect_stdout(f1):
                            print(right_rate[-1], end = "\n")
                    '''

                else:
                    raise Exception("Joints out of frame...")
            except:
                print("Need to get necessary joints in frame")
        
        '''
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        '''
        
        imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)

        imgbytes = cv2.imencode('.png', frame1)[1].tobytes()  # ditto
        window['image2'].update(data=imgbytes)
        

        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
