import pandas as pd
import math
import numpy as np

from dtaidistance import dtw
from dtaidistance import dtw_visualisation as dtwvis
import random

df_expert = pd.read_csv('workout_coor/out_shoulderpress_expert.csv')
df_beginner = pd.read_csv('workout_coor/out_shoulderpress_beginner_v2.csv')
df_bad = pd.read_csv('workout_coor/out_shoulderpress_bad.csv')

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

rshoul_eX, rshoul_eY = getXY(df_expert['RShoulder']) 
relbow_eX, relbow_eY = getXY(df_expert['RElbow'])
rwrist_eX, rwrist_eY = getXY(df_expert['RWrist'])

rshoul_bX, rshoul_bY = getXY(df_beginner['RShoulder']) 
relbow_bX, relbow_bY = getXY(df_beginner['RElbow'])
rwrist_bX, rwrist_bY = getXY(df_beginner['RWrist'])

rshoul_bdX, rshoul_bdY = getXY(df_bad['RShoulder']) 
relbow_bdX, relbow_bdY = getXY(df_bad['RElbow'])
rwrist_bdX, rwrist_bdY = getXY(df_bad['RWrist'])

lshoul_eX, lshoul_eY = getXY(df_expert['LShoulder']) 
lelbow_eX, lelbow_eY = getXY(df_expert['LElbow'])
lwrist_eX, lwrist_eY = getXY(df_expert['LWrist'])

lshoul_bX, lshoul_bY = getXY(df_beginner['LShoulder']) 
lelbow_bX, lelbow_bY = getXY(df_beginner['LElbow'])
lwrist_bX, lwrist_bY = getXY(df_beginner['LWrist'])

lshoul_bdX, lshoul_bdY = getXY(df_bad['LShoulder']) 
lelbow_bdX, lelbow_bdY = getXY(df_bad['LElbow'])
lwrist_bdX, lwrist_bdY = getXY(df_bad['LWrist'])

rightArm_eangles = []
rightArm_bangles = []
rightArm_bdangles = []

leftArm_eangles = []
leftArm_bangles = []
leftArm_bdangles = []

for i in range(len(rshoul_eY)):
	rightArm_eangles.append(angle3pt(rshoul_eX[i],rshoul_eY[i], relbow_eX[i], relbow_eY[i], rwrist_eX[i], rwrist_eY[i]))

for i in range(len(rshoul_bY)):
	rightArm_bangles.append(angle3pt(rshoul_bX[i],rshoul_bY[i], relbow_bX[i], relbow_bY[i], rwrist_bX[i], rwrist_bY[i]))

for i in range(len(rshoul_bdY)):
	rightArm_bdangles.append(angle3pt(rshoul_bdX[i],rshoul_bdY[i], relbow_bdX[i], relbow_bdY[i], rwrist_bdX[i], rwrist_bdY[i]))

for i in range(len(lshoul_eY)):
	leftArm_eangles.append(angle3pt(lshoul_eX[i],lshoul_eY[i], lelbow_eX[i], lelbow_eY[i], lwrist_eX[i], lwrist_eY[i]))

for i in range(len(lshoul_bY)):
	leftArm_bangles.append(angle3pt(lshoul_bX[i],lshoul_bY[i], lelbow_bX[i], lelbow_bY[i], lwrist_bX[i], lwrist_bY[i]))

for i in range(len(lshoul_bdY)):
	leftArm_bdangles.append(angle3pt(lshoul_bdX[i],lshoul_bdY[i], lelbow_bdX[i], lelbow_bdY[i], lwrist_bdX[i], lwrist_bdY[i]))

leftArm_b = np.asarray(leftArm_bangles, dtype = np.float32)
leftArm_e = np.asarray(leftArm_eangles, dtype = np.float32)

#distance = dtw.distance(leftArm_eangles[:250], leftArm_bangles[:250])
#print(distance)

d, path = dtw.warping_paths(leftArm_b, leftArm_e)
best_path = dtw.best_path(paths)
dtwvis.plot_warping(leftArm_b, leftArm_e, best_path, filename = "test11.png")
#print(paths)
#dtwvis.plot_warpingpaths(leftArm_e, leftArm_b, paths, path = best_path, filename = "Graphs/bestpath_window100+psi5.png") 
