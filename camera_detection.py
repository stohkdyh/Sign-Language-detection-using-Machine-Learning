import os
import tensorflow as tf
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import mediapipe as mp
# from sklearn.model_selection import train_test_split

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

cap = None
camera_open = False

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=4), mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=4), mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=4), mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=2))

def close_camera():
    global camera_open, cap
    if camera_open:
        cv2.destroyAllWindows()
        cap.release()
        camera_open = False

def open_camera():
    global camera_open, cap
    if not camera_open:  # Jika kamera belum dibuka
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        camera_open = True
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                # Read feed
                ret, frame = cap.read()
                if not ret:
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                mirror_cam = cv2.flip(image, 1)

                # Show to screen
                cv2.imshow('OpenCV Feed', mirror_cam)

                # Break gracefully
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    close_camera()
                    break
    else:
        print("Kamera sudah dibuka. Tutup kamera sebelum membukanya lagi.")