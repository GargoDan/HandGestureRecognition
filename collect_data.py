import cv2
import numpy as np
import os
import mediapipe as mp
from itertools import product

from utils.utils import *

mp_hands = mp.solutions.hands # Hands model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles

# Set parametrs

# Path for exported data, numpy arrays
DATA_FOLDER = 'Data_Daniil'
DATA_PATH = os.path.join(DATA_FOLDER) # make sence?


dict_yn = {'n':False, 'y':True}
default_settings = True
print('To stop video recording press q, \n')

print('default settings:')
# Actions that we try to detect
actions = np.array(['fire', 'water', 'earth', 'air'])
print('actions:', *actions)
# Number of videos
no_sequences = 30
print(f'Number of videos to record: {no_sequences}')
# break per videos in ms
time_to_break = 1000
print(f'Break per videos in ms: {time_to_break}')
# Videos are going to be 30 frames in length
frames_in_video = 30
print(f'Frames in video: {frames_in_video}, \n')


inp = input('Want to use the default settings? (y/n):')
if inp in dict_yn:
    default_settings = dict_yn[inp]
else:
    print('wrong key, using default settings')


if not default_settings:
    actions_dict = {1:'fire', 2:'water', 3:'earth', 4:'air'}
    print(f'Type numbers of actions: {actions_dict}')
    print('For example for fire and air type: 1, 4')
    actions = np.array([actions_dict[int(x)] for x in input().split(',')])
    no_sequences = int(input('Number of videos to record, default=30: '))
    time_to_break = int(input('Break per videos in ms, default=1000: '))
    frames_in_video = int(input('Frames in video, default=30: '))
 

# creating folders
def create_folders(actions, no_sequences):
    '''
    Сreate folders. Needs to be rewritten.
    '''
    folders_to_fill = []
    # create folders if first time
    # переписать, если первый раз создавать не все директории, то лишнее просто удаляться
    if DATA_FOLDER not in os.listdir():
        for action, num in list(product(actions, np.arange(no_sequences))):
            folders_to_fill.append((action, num))
            os.makedirs(os.path.join(DATA_FOLDER, action, str(num))) 
    else:
        for action in actions: 
            # удаляет все пустые директории
            folders_in_action = list(os.walk(os.path.join(DATA_FOLDER, action)))[1:]

            max_folder_num = 0
            for folder in folders_in_action:
                if not folder[2]:
                    os.rmdir(folder[0])
                else:
                    # если директория не пустая, то проверяем не максимальный ли у нее номер
                    # должен быть способ написать это проще
                    folder_num = int(os.path.split(folder[0])[-1])
                    if max_folder_num < folder_num:
                        max_folder_num = folder_num

            for num in np.arange(max_folder_num + 1, max_folder_num + 1 + no_sequences):
                folders_to_fill.append((action, num))
                os.makedirs(os.path.join(DATA_FOLDER, action, str(num)))

    return folders_to_fill

folders_to_fill = create_folders(actions, no_sequences)
print('folders created')


try:
    cap = cv2.VideoCapture(0)
    STOP_FLAG = False
    # Set mediapipe model 
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        # NEW LOOP
        for action, sequence in folders_to_fill:
            # Loop through video length aka sequence length
            folder_for_lh = np.array([]) # check the right place for it
            folder_for_rh = np.array([])
            
            out = cv2.VideoWriter(os.path.join(DATA_PATH, action, str(sequence), 'video.mp4'),
                                cv2.VideoWriter_fourcc(*'mp4v'), frames_in_video, (640,480)) # for saving video
        
            for frame_num in range(frames_in_video):

                ret, frame = cap.read()
                frame = cv2.flip(frame, 1) # flip the frame to detect hands cprrectly
                # write video
                if ret:
                    out.write(frame)
                    
                # Make detections
                image, results = mediapipe_detection(frame, hands)
                
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Collecting frames for =={action.upper()}== Video Number {sequence}', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(time_to_break)
                else: 
                    cv2.putText(image, f'Collecting frames for =={action.upper()}== Video Number {sequence}', (15,12), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                # Draw landmarks and get keypoints for it
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                    
                    left_fea, right_fea = extract_keypoints(results)
                    folder_for_lh = np.append(folder_for_lh, left_fea)
                    folder_for_rh = np.append(folder_for_rh, right_fea)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("break")
                    STOP_FLAG = True
                    break
            
            # saving keypoints
            lh_and_rh = np.concatenate([folder_for_lh, folder_for_rh])
            np.save(os.path.join(DATA_PATH, action, str(sequence), 'kp'), lh_and_rh)
            # saving the video
            out.release() 

            if STOP_FLAG: break
        out.release() 
        cap.release()
        cv2.destroyAllWindows()

finally:
    cv2.destroyAllWindows()