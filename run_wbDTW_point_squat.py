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
from operator import add 

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

    #-----------------------------------------------------------------------------------------------------------------------#
    #clear output file
    fields = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist', 'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee', 
                'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'REye', 'LEye', 'REar', 'LEar', 'Background']

    with open('out.csv', 'w') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile)             
            csvwriter.writerow(fields)   

    df_expert = pd.read_csv('workout_coor/out_squats_expert.csv')

    rhip_eX, rhip_eY = getXY(df_expert['RHip']) 
    rknee_eX, rknee_eY = getXY(df_expert['RKnee'])
    rankle_eX, rankle_eY = getXY(df_expert['RAnkle'])

    lhip_eX, lhip_eY = getXY(df_expert['LHip']) 
    lknee_eX, lknee_eY = getXY(df_expert['LKnee'])
    lankle_eX, lankle_eY = getXY(df_expert['LAnkle'])

    #X postion for both arms expert
    exR_list = [sum(i) for i in zip(rhip_eX, rknee_eX, rankle_eX)]
    exL_list = [sum(i) for i in zip(lhip_eX, lelbow_eX, lwrist_eX)]

    #Y postion for both arms expert
    eyR_list = [sum(i) for i in zip(rhip_eY, rknee_eY, rankle_eY)]
    eyL_list = [sum(i) for i in zip(lhip_eY, lknee_eY, lankle_eY)]

    exyR = [sum(i) for i in zip(exR_list, eyR_list)]
    exyL = [sum(i) for i in zip(exL_list, eyL_list)]

    exy = np.asarray(list(map(add, exyR, exyL)) , dtype = np.float32)
    
    right_dist = [0]
    left_dist = [0]
    dist = [0]

    right_rate = [0]
    left_rate = [0]

    #-----------------------------------------------------------------------------------------------------------------------#
    while True:
        ret_val, image = cam.read()

        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)

        image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        #print(len(humans[0].body_parts.keys()))

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

                    rhip_uX, rhip_uY = getXY(df_user['RHip']) 
                    rknee_uX, rknee_uY = getXY(df_user['RKnee'])
                    rankle_uX, rankle_uY = getXY(df_user['RAnkle'])

                    lshoul_uX, lshoul_uY = getXY(df_user['LHip']) 
                    lelbow_uX, lelbow_uY = getXY(df_user['LKnee'])
                    lwrist_uX, lwrist_uY = getXY(df_user['LAnkle'])

                    uxR_list = [sum(i) for i in zip(rshoul_uX, relbow_uX, rwrist_uX)]
                    uxL_list = [sum(i) for i in zip(lshoul_uX, lelbow_uX, lwrist_uX)]

                    uyR_list = [sum(i) for i in zip(rshoul_uY, relbow_uY, rwrist_uY)]
                    uyL_list = [sum(i) for i in zip(lshoul_uY, lelbow_uY, lwrist_uY)]

                    #keep track of line/frame
                    line = len(leftArm_uX)
                    print(line)

                    uxyR = [sum(i) for i in zip(uxR_list, uyR_list)]
                    uxyL = [sum(i) for i in zip(uxL_list, uyLR_list)]

                    uxy = np.asarray(list(map(add, uxyR, uxyL)) , dtype = np.float32)

                    dist.append(dtw.distance(uxy[:line], exy[:line]))
                    
                    print(dist[-1])

                else:
                    raise Exception("Joints out of frame...")
            except:
                print("Need to get necessary joints in frame")
        
        cv2.putText(image,
                    "FPS: %f" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
