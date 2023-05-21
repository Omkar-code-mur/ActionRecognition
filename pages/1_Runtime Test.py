
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'D:/Streamlit')
sys.path.insert(1, 'D:/Streamlit/models')

import base64
from model1 import *
import cv2
import numpy as np
import os
import time
import mediapipe as mp
import streamlit as st
from gtts import gTTS

def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        
        sound = st.empty()
        sound.markdown(
            md,
            unsafe_allow_html=True,
        )
        time.sleep(2)  # wait for 2 seconds to finish the playing of the audio
        sound.empty()
  
# This module is imported so that we can 
# play the converted audio

  

  
# Language in which you want to convert
language = 'en'
  
# Passing the text and language to the engine, 
# here we have marked slow=False. Which tells 
# the module that the converted audio should 
# have a high speed


cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()


# Set mediapipe model
start_button_pressed = st.button("Start Camera")
stop_button_pressed = st.button("Stop Camera")
# 1. New detection variables

start_button_pressed = True

if start_button_pressed:
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.8
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened() and not stop_button_pressed:

            # Read feed
            ret, frame = cap.read()

            if not ret:
                st.write("the video capture has ended.")
            # # Make detections
            image, results = mediapipe_detection(frame, holistic)
            
            # image = frame
            #
            # Draw landmarks
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                # print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:

                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                mytext = actions[np.argmax(res)]
                                myobj = gTTS(text=mytext, lang=language, slow=False)
  
                                # Saving the converted audio in a mp3 file named
                                # welcome 
                                myobj.save("audio.mp3")
                                
                                # Playing the converted file
                                autoplay_audio("audio.mp3")
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                # Viz probabilities
                #image = prob_viz(res, actions, image, colors)

            cv2.rectangle(image, (0, 0), (1640, 40), (1245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            frame = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame,channels="RGB")

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()



