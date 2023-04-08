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

###--------------------------------------------------------------------------------------------------------------------------------------------------
sg.theme('SystemDefault')

center_vid = [[sg.Image(filename = '', key='image')],
                [sg.Text('Please Step Into Frame!', key = 'text', text_color = 'Black', font = 'Helvetica 14')]]

layout = [[sg.Column(center_vid, element_justification = 'center', vertical_alignment = 'center', justification = 'center')],
            [sg.Exit('Exit')]]

window = sg.Window('Test Configuration', layout)
recording = False
###--------------------------------------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
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
     
    while True:
        event, values = window.read(timeout = 100)

        if event == 'Exit' or event == sg.WIN_CLOSED:
            window.close()
            break

        recording = True

        if recording:
            ret_val, image = cam.read()


        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        #logger.debug('postprocess+')
        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        #fields = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
        
        try:
            body_list = [[str(humans[0].body_parts[0]), str(humans[0].body_parts[1]), str(humans[0].body_parts[2]), 
                str(humans[0].body_parts[3]), str(humans[0].body_parts[4]), str(humans[0].body_parts[5]), 
                str(humans[0].body_parts[6]), str(humans[0].body_parts[7]), str(humans[0].body_parts[8]), 
                str(humans[0].body_parts[9]), str(humans[0].body_parts[10]), str(humans[0].body_parts[11]), 
                str(humans[0].body_parts[12]), str(humans[0].body_parts[13]), str(humans[0].body_parts[14]), 
                str(humans[0].body_parts[15]), str(humans[0].body_parts[16]),str(humans[0].body_parts[17])]] 
                
            if len(humans[0].body_parts.keys())-1 == 17: #wait for all joints to be in frame
                current_pos = 'GOOD!'
                out_color = 'Green'
            else:
                raise Exception("Joints out of frame...")
        except:
            current_pos = 'Please Get Into Frame!'
            out_color  = 'Red'
        
        window['text'].update(current_pos, text_color = out_color)
        
        '''
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        '''

        imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)

        fps_time = time.time()
        if cv2.waitKey(1) == 27:
            break
        #logger.debug('finished+')

    cv2.destroyAllWindows()
