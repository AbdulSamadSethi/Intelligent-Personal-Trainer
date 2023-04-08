import PySimpleGUI as sg
import cv2 as cv
import numpy as np
import imutils

"""
Demo program that displays a webcam using OpenCV
"""
def main():

    sg.theme('SystemDefault')
    Exercises = ['Shoulder Press', 'Squat', 'Lateral Raises']

    # define the window layout
    layout = [[sg.Text('Intelligent Personal Trainer', size=(40, 1), justification='center', font='Helvetica 20')],
              [sg.Listbox(list(Exercises), size=(20,4), pad=((250, 0), 3), enable_events=True, key='_LIST_')],
              [sg.Image(filename='', key='image'), sg.Image(filename='', key='image2')],
              [sg.Button('START', size=(10, 1),pad=((270, 0), 3), font='Helvetica 14'),sg.Exit('Exit', size=(10, 1),pad=((270, 0), 5), font='Helvetica 14')]]

    layout2 = [[sg.Text('Shoulder Press', size = (40, 1), justification='center', font='Helvetica 20')]]

    # create the window and show it without the plot
    window = sg.Window('Demo Application - OpenCV Integration', layout, location=(600, 400))

    window2 = sg.Window('Shoulder Press', layout2, location(600,400))

    # ---===--- Event LOOP Read and display frames, operate the GUI --- #
    cap = cv.VideoCapture(0)
    cap1 = cv.VideoCapture('workouts_files/shoulder_press_expert.mp4')

    recording = False

    frame_rate = 10

    while True:
        event, values = window.read(timeout=20)

        if event == 'START' and values['_LIST_']:
            recording = True

        if event == 'Exit':
            window.close()
            break

        if recording:
            ret, frame = cap.read()
            ret, frame1 = cap1.read()
            
            frame1 = imutils.resize(frame1, width=640)
            #frame1 = cv.resize(frame1,(640, 400));

            imgbytes = cv.imencode('.png', frame)[1].tobytes()  # ditto
            window['image'].update(data=imgbytes)

            imgbytes = cv.imencode('.png', frame1)[1].tobytes()  # ditto
            window['image2'].update(data=imgbytes)
            
main()