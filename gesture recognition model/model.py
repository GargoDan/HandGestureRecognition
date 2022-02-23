import numpy as np
import cv2
import mediapipe as mp
import time

from catboost import CatBoostClassifier


colors = [(245,117,16), (117,245,16), (16,117,245), (255, 185, 0), (255, 100, 0)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def get_scores(results, left_hand_detection, right_hand_detection):
    '''
    fingers_for_action = ['air', 'water', 'earth', 'fire']
    '''
    scores = np.zeros(4)
    fingers_for_action = [[4, 8, 12], [4, 8, 20], [5, 9, 13, 17], [7, 6, 10, 11]]


    for i, action in enumerate(fingers_for_action):
        if left_hand_detection and right_hand_detection:
            hand0 = results.multi_hand_landmarks[0]
            hand1 = results.multi_hand_landmarks[1]
            for kp in action:
                scores[i] += np.abs(hand1.landmark[kp].x - hand0.landmark[kp].x)
                scores[i] += np.abs(hand1.landmark[kp].y - hand0.landmark[kp].y)
        else:
            hand0 = results.multi_hand_landmarks[0]
            for kp in action:
                scores[i] += np.abs(hand0.landmark[kp].x)
                scores[i] += np.abs(hand0.landmark[kp].y)

    # scores = scores ** 2
    return scores


# function for extracting keypoints
def extract_keypoints(results):
    left_hand_detection, right_hand_detection = False, False
    for cl in results.multi_handedness:
        if cl.classification[0].index == 0:
            left_hand_detection = True
        else:
            right_hand_detection = True

    if left_hand_detection and right_hand_detection:
        lh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_world_landmarks[0].landmark]).flatten()
        rh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_world_landmarks[1].landmark]).flatten()
    elif left_hand_detection:
        lh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_world_landmarks[0].landmark]).flatten()
        rh = np.zeros(21*3)
    elif right_hand_detection:
        lh = np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_world_landmarks[0].landmark]).flatten()

    scores = get_scores(results, left_hand_detection, right_hand_detection)

    kps = np.concatenate([lh.reshape((21*3, -1)).T, rh.reshape((21*3, -1)).T], axis=1)
    kps = np.append(kps, scores)
    return kps

mp_hands = mp.solutions.hands # Hands model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles

actions = np.array(['fire', 'water', 'earth', 'air', 'trash'])
map_actions = {label:num for num, label in enumerate(actions)}
inv_map_actions = {v: k for k, v in map_actions.items()}


model = CatBoostClassifier()

model.load_model('gesture recognition model\\model\\catboost_m_v5_nf_cl.cbm')
classes = model.classes_
actions = model.classes_
map_actions = {label:num for num, label in enumerate(actions)}
inv_map_actions = {v: k for k, v in map_actions.items()}


# 1. New detection variables
str_action = []
predictions = []
threshold = 0.7
prev_frame_time = 0
new_frame_time = 0

try:
    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1) # flip the frame to detect hands cprrectly

            new_frame_time = time.time()

            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))

            # Make detections
            image, results = mediapipe_detection(frame, hands)
            
            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

                # 2. Prediction logic
                # only if hand detected
                keypoints = extract_keypoints(results)

                res = model.predict_proba(keypoints)
                # print(keypoints[-1])
                predictions.append(np.argmax(res))

                #3. Viz logic
            if len(predictions) > 0:
                image = prob_viz(res, actions, image, colors)
                if np.unique(predictions[-5:])[0] == np.argmax(res): # если последние 10 предикты совпадают
                    if res[np.argmax(res)] > threshold:
                        str_action = inv_map_actions[np.argmax(res)]
                
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(str_action), (3,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, fps, (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
finally:
    cv2.destroyAllWindows()