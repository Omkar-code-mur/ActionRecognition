import cv2
import os
import numpy as np
import streamlit  as st
import sys
sys.path.insert(1, 'D:/Sign/new')
from model1 import *


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

action = 0
# Actions that we try to detect
action = st.text_input("Enter the word you want to train")
stop_button_clicked = st.button("Stop")

restart_button_clicked = st.button("Restart")

if action != "":
    # Thirty videos worth of data
    no_sequences = 60

    # Videos are going to be 30 frames in length
    sequence_length = 30

    # Folder start
    start_folder = 2

    
    print(restart_button_clicked,"value")

    for sequence in range(30,no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()
# Set mediapipe model 
def start():
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # NEW LOOP
        # Loop through sequences aka videos
        for sequence in range(30,no_sequences):
            if stop_button_clicked:
                break
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
    #                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame,channels="RGB")
                    cv2.waitKey(2000)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame,channels="RGB")
                
                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
                        
        cap.release()
        cv2.destroyAllWindows()
if action != "":
    start()
    stop_button_clicked = True


if restart_button_clicked:
    start()