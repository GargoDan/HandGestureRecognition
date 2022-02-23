import cv2
import numpy as np


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