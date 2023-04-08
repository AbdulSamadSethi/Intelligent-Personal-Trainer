import PySimpleGUI as sg
import cv2 
import imutils
import os

def shoulder_press():
    
    sp_info = ("Shoulder Press:\n"
                "Position your feet shoulder-width apart and stand holding two dumbbells\n"
                "at shoulder height with an overhand grip. Press the weights up above your\n" 
                "head until your arms are fully extended. Return slowly to the start position.\n"
                "\n"
                "Expert tip: Avoid the urge to arch your back and tilting your hids forward\n"
                "\n"
                "Press start when you are ready to commence the workout.\n")
    
    g1 = r'workout_gifs\shoulder_press.gif'
    gifs = [g1]

    layout = [[sg.Image(filename = '', background_color = 'white', key = 'image')],
                [sg.Text(sp_info)], 
                [sg.Button('Start'),sg.Exit('Exit')]]
    window = sg.Window('Shoulder Press Workout', layout, finalize = True)
    
    cap1 = cv2.VideoCapture(g1)

    while True:
        event, values = window.read(timeout=100)
        ret, frame1 = cap1.read()

        if frame1 is None:
            cap1 = cv2.VideoCapture(g1)
            ret, frame1 = cap1.read()
            frame1 = imutils.resize(frame1, width= 610)
        else:
            frame1 = imutils.resize(frame1, width= 610)

        if event == 'Exit' or event == sg.WIN_CLOSED:
            window.close()
            break
        
        if event == 'Start':
            os.system('python run_shoulderpress.py --model=mobilenet_thin --resize=432x368 --camera=0')
            window.close()
        
        imgbytes = cv2.imencode('.png', frame1)[1].tobytes()
        window['image'].update(data=imgbytes)

def squat():

    squat_info = ("Squat:\n"
                "\n"
                "Stand with feet hip-width apart, holding dumbbells at shoulders, with abs tight. Send hips back and bend knees to lower\n" 
                "until your thighs are at least parallel to the ground, ideally lower. Push back up to the starting position.\n"
                "\n"
                "Expert tip: If you are just starting out you can ditch the weights and clasp your hands in front of your chest for balance\n"
                "\n"
                "Expert tip: To check if your form is correct, when decending on the movement look down and see if your knee goes over your toes,\n"
                "if so adjust by shifting your body weight to your heals to drive the movement through your glutes and less with your knees!\n"
                "\n"
                "Press start when you are ready to commence the workout.\n")

    g1 = r'workout_gifs\squat.gif'
    gifs = [g1]
    
    center_gif = [[sg.Image(filename = '', background_color = 'white', key = 'image')]]


    layout = [[sg.Column(center_gif, element_justification = 'center', vertical_alignment = 'center', justification = 'center')],
                [sg.Text(squat_info)], 
                [sg.Button('Start'),sg.Exit('Exit')]]
    
    window = sg.Window('Squat Workout', layout, finalize = True)
    cap1 = cv2.VideoCapture(g1)

    while True:
        event, values = window.read(timeout = 100)
        ret, frame1 = cap1.read()

        if frame1 is None:
            cap1 = cv2.VideoCapture(g1)
            ret, frame1 = cap1.read()
            frame1 = imutils.resize(frame1, width= 610)
        else:
            frame1 = imutils.resize(frame1, width= 610)

        if event == 'Exit' or event == sg.WIN_CLOSED:
            window.close()
            break

        if event == 'Start':
            os.system('python run_squat.py --model=mobilenet_thin --resize=432x368 --camera=0')
            window.close()
        
        imgbytes = cv2.imencode('.png', frame1)[1].tobytes()
        window['image'].update(data=imgbytes)


def lateral_raise():

    lr_info = ("Lateral Raise:\n"
                "Stand or sit with a dumbbell in each hand at your sides. Keep your back straight, brace your core, and then slowly lift the\n" 
                "weights out to the side until your arms are parallel with the floor, with the elbow slightly bent.\n"
                "\n"
                "Expert tip: When using weights make sure to use very light weights as it does not take much to work this muscle\n"	
                "\n"
                "Press start when you are ready to commence the workout.\n")
    
    g1 = r'workout_gifs\side_raise.gif'
    gif = [g1]

    center_gif = [[sg.Image(filename = '', background_color = 'white', key = 'image')]]

    layout = [[sg.Column(center_gif, element_justification = 'center', vertical_alignment = 'center', justification = 'center')],
                [sg.Text(lr_info)], 
                [sg.Button('Start'),sg.Exit('Exit')]]

    window = sg.Window('Lateral Raise Workout', layout, finalize = True)
    cap1 = cv2.VideoCapture(g1)


    while True:
        event, values = window.read(timeout = 100)
        ret, frame1 = cap1.read()

        if frame1 is None:
            cap1 = cv2.VideoCapture(g1)
            ret, frame1 = cap1.read()
            frame1 = imutils.resize(frame1, width= 610)
        else:
            frame1 = imutils.resize(frame1, width= 610)
        
        if event == 'Exit' or event == sg.WIN_CLOSED:
            window.close()
            break

        if event == 'Start':
            os.system('python run_lateralraise.py --model=mobilenet_thin --resize=432x368 --camera=0')
            window.close()
        
        imgbytes = cv2.imencode('.png', frame1)[1].tobytes()
        window['image'].update(data=imgbytes)
    

def workout_menu():

    exercises = ['Shoulder Press', 'Lateral Raise', 'Squat']

    items_in_center = [[sg.Text('Workout Library', size = (40,1), justification = 'center', font='Helvetica 20')],
                        [sg.Listbox(exercises, size= (20,4), key = '_LIST_',pad = (0,(10,0)) ,enable_events=True)]] 

    layout = [[sg.Column(items_in_center, element_justification = 'center', vertical_alignment = 'center', justification = 'center')],
                [sg.Button('Select', pad = ((125,0),(10,0)), size = (10,2)), sg.Button('Main Menu', pad = ((65,0),(10,0)), size = (10,2))]]
    
    window = sg.Window('Workout Menu', layout, size = (500,200), resizable = True, finalize = True)

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED:
            break

        if event == 'Main Menu':
            window.close()
            main_menu()

        if event == 'Select' :
            selection = values['_LIST_']

            if not selection:
                sg.popup_error('Please select a workout to continue')
            elif selection[0] == 'Shoulder Press':
                shoulder_press()

            elif selection[0] == 'Lateral Raise':
                lateral_raise()
                
            elif selection[0] == 'Squat':
                squat()
                
def test_setup():
    os.system('python run_webcamtest.py --model=mobilenet_thin --resize=432x368 --camera=0')


def instructions():
    how_to = ("HOW IT WORKS:\n"
    " \n"
	"The AI Personal Trainer uses machine learning algorithims, implemented with tensorFlow and OpenCV, to track your body and compare your movements during workouts with our expert videos and give you REAL TIME FEEDBACK on your progress. Our goal is to provide everyone the opertunity to stay and maintain physical fitness during these unprecedented times due to the COVID-19 epidemic.\n"
    " \n"
    " \n"
	"HOW TO SET IT UP:\n"
    " \n"
	"For the best results please place your webcam/laptop in an area in which you can easily view your screen while having your whole body in frame for the AI Trainer to read. To check if your possible set up can work effectively please use our Test Setup' feature in the Home screen before you attempt a workout. We've seen that the AI Trainer works best in areas with ample lighting and room for you to move around in.\n")

    sg.popup(how_to, title = 'Instructions')

def support():
    sg.popup('For any product concerns or technical support please email us at: support@aitrainer.com', title = 'Support')


def main_menu():

    center_col = [[sg.Text('Welcome to the AI Personal Trainer', size=(40, 1), font='Helvetica 20', justification = 'center')],
                    [sg.Button('Select Workout', size = (20,2))], 
                    [sg.Button('Test Setup',  size = (20,2))], 
                    [sg.Button('Instructions', size = (20, 2)),],
                    [sg.Button('Support',  size = (20,2))],
                    [sg.Exit('Exit',  size = (20,2))]]

    layout = [[sg.Column(center_col, element_justification = 'center', vertical_alignment = 'center', justification = 'center')]]
    
    window = sg.Window('Main Menu', layout, size = (500,300), resizable = True, finalize = True)

    while True:
        event, values = window.read()

        if event == 'Exit' or event == sg.WIN_CLOSED:
            window.close()
            break

        if event == 'Select Workout':
            window.close()
            workout_menu()
            
        if event == 'Instructions':
            instructions()

        if event == 'Support':
            support()
        
        if event == 'Test Setup':
            test_setup()



if __name__ == "__main__":
    main_menu()
