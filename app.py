import streamlit as st
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

UPPER_EYE_LEFT = [246, 161, 160, 159, 158, 157, 173, 133]
UPPER_EYE_RIGHT = [7, 33, 161, 160, 159, 158, 157, 173]

def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.face_landmarks.landmark]
    if draw :
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Function to check if eyes are looking up
def is_eyes_looking_up(landmarks, upper_eye_indices):
    upper_eye_points = [landmarks[idx] for idx in upper_eye_indices]
    average_y = sum(point[1] for point in upper_eye_points) / len(upper_eye_points)
    return average_y < landmarks[LEFT_EYE[0]][1] and average_y < landmarks[RIGHT_EYE[0]][1]

# Function to process video and perform actions
def process_video():
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        image_widget = st.image([], channels="BGR", use_column_width=True)
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            
            # 1. Draw face landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                    )
            
            # 2. Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                    )

            # 3. Left Hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                    )

            # 4. Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )

            if results.right_hand_landmarks and results.left_hand_landmarks and results.pose_landmarks and results.face_landmarks:
                right_hand_landmarks = results.right_hand_landmarks.landmark
                left_hand_landmarks = results.left_hand_landmarks.landmark
                pose_landmarks = results.pose_landmarks.landmark
                face_landmarks = results.face_landmarks.landmark

                mesh_coords = landmarksDetection(frame, results, False)

                right_hand_tip_x = int(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                left_hand_tip_x = int(left_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
                right_hand_tip_y = int(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                left_hand_tip_y = int(left_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
                right_hand_dip_y = int(right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y * frame.shape[0])
                left_hand_dip_y = int(left_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_DIP].y * frame.shape[0])
                right_hand_thumb = int(right_hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP].y * frame.shape[0])
                left_hand_thumb = int(left_hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP].y * frame.shape[0])
                right_shoulder = pose_landmarks[12]
                left_shoulder = pose_landmarks[11]
                mouth_left = pose_landmarks[9]
                mouth_right = pose_landmarks[10]

                mouth_left_y = int(mouth_left.y * frame.shape[0])
                mouth_right_y = int(mouth_right.y * frame.shape[0])
                mouth_left_x = int(mouth_left.x * frame.shape[1])
                mouth_right_x = int(mouth_right.x * frame.shape[1])
                right_shoulder_y = int(right_shoulder.y * frame.shape[0])
                left_shoulder_y = int(left_shoulder.y * frame.shape[0])

                val_hand_tips = left_hand_tip_x - right_hand_tip_x
                val_hand_thumbs = left_hand_thumb - right_hand_thumb

                thumb_tip = right_hand_landmarks[mp_holistic.HandLandmark.THUMB_TIP]
                index_finger_tip = right_hand_landmarks[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
                right_cheek = face_landmarks[116]
                right_ear = face_landmarks[147]
                chin = face_landmarks[152]
                left_cheek = face_landmarks[323]

                thumb_tip_y = int(thumb_tip.y * frame.shape[0])
                thumb_tip_x = int(thumb_tip.x * frame.shape[1])
                index_finger_tip_y = int(index_finger_tip.y * frame.shape[0])
                index_finger_tip_x = int(index_finger_tip.x * frame.shape[1])
                cheek_y = int(right_cheek.y * frame.shape[0])
                right_ear_x = int(right_ear.x * frame.shape[1])
                chin_y = int(chin.y * frame.shape[0])
                left_cheek_x = int(left_cheek.x * frame.shape[1])

                # Display message if thumb or index finger is on the cheek
                if ((chin_y > thumb_tip_y) and (right_ear_x < thumb_tip_x) and (thumb_tip_y > cheek_y) and (thumb_tip_x < left_cheek_x)) or ((chin_y > index_finger_tip_y) and (right_ear_x < index_finger_tip_x) and (index_finger_tip_y > cheek_y) and (index_finger_tip_x < left_cheek_x)):
                    if(left_hand_tip_y > left_shoulder_y):
                        cv2.putText(image, "Positive evaluation low risk situation", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif((right_hand_tip_y < right_shoulder_y and left_hand_tip_y < left_shoulder_y) and (right_hand_tip_y > mouth_right_y and left_hand_tip_y > mouth_left_y) and (right_hand_dip_y > right_hand_tip_y and left_hand_dip_y > left_hand_tip_y) and (left_hand_tip_x < mouth_left_x and right_hand_tip_x > mouth_right_x)):
                    if(val_hand_tips < 20 and val_hand_thumbs < 20):
                        cv2.putText(image, "Wants her knowledge to be recognized now", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif is_eyes_looking_up(mesh_coords, UPPER_EYE_LEFT + UPPER_EYE_RIGHT):
                    cv2.putText(image, "Eyes Looking Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
            image_widget.image(image, channels="BGR")


# Streamlit app layout
st.title("Mediapipe and Streamlit Demo")
st.write("Press the button to start processing the video.")

if st.button("Stop", key="stop_button"):
    st.stop()

process_video()
