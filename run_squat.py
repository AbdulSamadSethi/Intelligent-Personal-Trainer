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

def endWin(num_sc,wr_sc):

    sc = str(num_sc)
    center = [[sg.Text('Congratulations On Completeing Your Workout!')],
                [sg.Text('You did: '), sg.Text(wr_sc)],
                [sg.Text('Score: '), sg.Text(sc)],
                [sg.Exit('Exit')]]

    layout = [[sg.Column(center, element_justification = 'center', vertical_alignment = 'center', justification = 'center')]]

    window = sg.Window('WORKOUT COMPLETED', layout, finalize = True)

    while True:
        event, values = window.read()

        if event == 'Exit' or event == sg.WIN_CLOSED:
            window.close()
            break

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

def change_bar_color(progbar: sg.ProgressBar, color: str):
    from tkinter import ttk
    s = ttk.Style()
    s.configure(progbar.TKProgressBar.style_name, background=color)

###---------------------------------------------------------------------------------------------
sg.theme('SystemDefault')
Exercises = ['Shoulder Press', 'Squat', 'Lateral Raises']

but_col = [[sg.Button('Exit',size=(20,3))]]
wor_sc_col = [[sg.Text('Please Get Into Frame', key='score', text_color = 'Black', font = 'Helvetica 14')]]
wor_num_col = [[sg.Text('-------', key='num_score',text_color = 'Black', font = 'Helvetica 14')]] 

# define the window layout
layout = [[sg.Image(filename='', key='image'), sg.Image(filename='', key='image2')],
          [sg.ProgressBar(100, orientation='h', size=(118, 30),bar_color = ('red','white'), key='progressbar')], 
          [sg.Column(wor_sc_col, element_justification = 'center', vertical_alignment = 'center', justification = 'center')],
          [sg.Column(wor_num_col, element_justification = 'center', vertical_alignment = 'center', justification = 'center')],
          [sg.Column(but_col, element_justification = 'center', vertical_alignment = 'center', justification = 'center')]]

# create the window and show it without the plot
window = sg.Window('Workout', layout, size = (1300, 650))
progress_bar = window['progressbar']

# ---===--- Event LOOP Read and display frames, operate the GUI --- #
cap1 = cv2.VideoCapture('workout_files/new-squats-expert-fast2.mp4')
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

df_expert = pd.read_csv('workout_coor/new_squats_expert.csv')

rankle_ex, rankle_ey = getXY(df_expert['RAnkle']) 
rknee_ex, rknee_ey = getXY(df_expert['RKnee'])
rhip_ex, rhip_ey = getXY(df_expert['RHip'])

lankle_ex, lankle_ey = getXY(df_expert['LAnkle']) 
lknee_ex, lknee_ey = getXY(df_expert['LKnee'])
lhip_ex, lhip_ey = getXY(df_expert['LHip'])

rightLeg_eangles = []
leftLeg_eangles = []

for i in range(len(rankle_ey)):
    rightLeg_eangles.append(angle3pt(rankle_ex[i],rankle_ey[i], rknee_ex[i], rknee_ey[i], rhip_ex[i], rhip_ey[i]))

for i in range(len(lankle_ey)):
    leftLeg_eangles.append(angle3pt(lankle_ex[i],lankle_ey[i], lknee_ex[i], lknee_ey[i], lhip_ex[i], lhip_ey[i]))

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
        
        
        if frame1 is None:
            cap1 = cv2.VideoCapture('workout_files/new-squats-expert-fast2.mp4')
            ret, frame1 = cap1.read()
            frame1 = imutils.resize(frame1, width= 640)
        else:
            frame1 = imutils.resize(frame1, width= 640)
        

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

                    rankle_ux, rankle_uy = getXY(df_user['RAnkle']) 
                    rknee_ux, rknee_uy = getXY(df_user['RKnee'])
                    rhip_ux, rhip_uy = getXY(df_user['RHip'])

                    lankle_ux, lankle_uy = getXY(df_user['LAnkle']) 
                    lknee_ux, lknee_uy = getXY(df_user['LKnee'])
                    lhip_ux, lhip_uy = getXY(df_user['LHip'])

                    rightLeg_uangles = []
                    leftLeg_uangles = []

                    for i in range(len(rankle_uy)):
                        rightLeg_uangles.append(angle3pt(rankle_ux[i],rankle_uy[i], rknee_ux[i], rknee_uy[i], rhip_ux[i], rhip_uy[i]))

                    for i in range(len(lankle_uy)):
                        leftLeg_uangles.append(angle3pt(lankle_ux[i],lankle_uy[i], lknee_ux[i], lknee_uy[i], lhip_ux[i], lhip_uy[i]))



                    #keep track of line/frame
                    line = len(leftLeg_uangles)
                    print(line)

                    #calculate DTW
                    if line % 2 == 0:
                        if line >= 100:
                            right_dist.append(dtw.distance(rightLeg_uangles[line-50:line], rightLeg_eangles[line-50:line]))
                            left_dist.append(dtw.distance(leftLeg_uangles[line-50:line], leftLeg_eangles[line-50:line]))
                        else:
                            right_dist.append(dtw.distance(rightLeg_uangles[:line], rightLeg_eangles[:line]))
                            left_dist.append(dtw.distance(leftLeg_uangles[:line], leftLeg_eangles[:line]))

                        #calculate rate of change of DTW
                        right_rate.append(right_dist[-1] - right_dist[-2])
                        left_rate.append(left_dist[-1] - left_dist[-2])

                        #print((right_dist[-1]+left_dist[-1]))
                        #print(right_rate[-1]+left_rate[-1])

                        if ((right_dist[-1]+left_dist[-1]) < 150):
                            curr_sr = 'Amazing!!!'
                            out_color = 'Green'
                            change_bar_color(window['progressbar'],'green')
                        
                        elif((right_dist[-1]+left_dist[-1]) > 150 and (right_dist[-1]+left_dist[-1]) < 500):
                            curr_sr = 'Good'
                            out_color = 'orange'
                            change_bar_color(window['progressbar'],'orange')

                        elif((right_dist[-1]+left_dist[-1]) > 500):
                            curr_sr = 'BAD!!'
                            out_color = 'Red'
                            change_bar_color(window['progressbar'],'red')

                        print(str(int(right_dist[-1]+left_dist[-1])))
                        print(right_dist[-1]+left_dist[-1])

                        us_scr = (right_dist[-1]+left_dist[-1])*0.1818
                        new_index = abs(max(0, min(us_scr, 100)) - 100)

                        num_sr = str(int(new_index))
                        window['score'].update(curr_sr,text_color = out_color)
                        window['num_score'].update(num_sr,text_color = out_color)
                        progress_bar.UpdateBar(new_index)
                        if line == 200:
                            window.close()
                            endWin(num_sr,curr_sr) 

                else:
                    raise Exception("Joints out of frame...")
            except:
                print("Need to get necessary joints in frame")
        
        imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

        imgbytes = cv2.imencode('.png', frame1)[1].tobytes()  # ditto
        window['image2'].update(data=imgbytes)
        

        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
