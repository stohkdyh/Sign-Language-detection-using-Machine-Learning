import customtkinter as ctk
import mysql.connector
import cv2
import numpy as np
import mediapipe as mp

def connect_to_db():
    conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="db_signlanguage"
    )
    return conn

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

cap = None
camera_open = False

num_sequences = 30
sequence_length = 30

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

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return pose, face, lh, rh
    # return np.concatenate([pose, face, lh, rh])

def close_camera():
    global camera_open, cap
    if camera_open:
        cv2.destroyAllWindows()
        cap.release()
        camera_open = False


def add_motion():
    dialog = ctk.CTkInputDialog(text="Masukkan Bahasa Isyarat yang ingin anda tambahkan", title="Tambah Isyarat")
    action = dialog.get_input()
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO actions (action_name) VALUES (%s)", (action,))
    conn.commit()
    
    # Ambil action_id
    cursor.execute("SELECT action_id FROM actions WHERE action_name=%s", (action,))
    action_id = cursor.fetchone()[0]  # Ambil hasil pertama

    global camera_open, cap
    if not camera_open:  # Jika kamera belum dibuka
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        camera_open = True
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for sequence in range(1, num_sequences + 1):
                cursor.execute("INSERT INTO sequences (sequence_num, action_id) VALUES (%s, %s)", (sequence, action_id))
                conn.commit()
                
                # Ambil sequence_id
                cursor.execute("SELECT sequence_id FROM sequences WHERE sequence_num=%s", (sequence,))
                sequence_id = cursor.fetchone()[0]  # Ambil hasil pertama

                # Loop through video length aka sequence length
                for frame_num in range(1, sequence_length + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    mirror_cam = cv2.flip(image, 1)

                    if frame_num == 1: 
                        cv2.putText(mirror_cam, 'STARTING COLLECTION', (120, 200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        cv2.putText(mirror_cam, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', mirror_cam)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15, 12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', mirror_cam)

                    # NEW Export keypoints
                    pose, face, lh, rh = extract_keypoints(results)

                    # Simpan keypoints ke tabel keyframes
                    cursor.execute("INSERT INTO keyframes (keyframe_num, sequence_id) VALUES (%s, %s)", (frame_num, sequence_id,))
                    conn.commit()
                    cursor.execute("SELECT keyframe_id FROM keyframes WHERE keyframe_num=%s", (frame_num,))
                    keyframe_id = cursor.fetchone()[0]  # Ambil ID keyframe yang baru saja dimasukkan

                    # Simpan pose landmarks
                    for i in range(len(pose) // 4):
                        cursor.execute("INSERT INTO pose_landmarks (x, y, z, visibility, keyframe_id) VALUES (%s, %s, %s, %s, %s)",
                                       (pose[i*4], pose[i*4+1], pose[i*4+2], pose[i*4+3], keyframe_id))

                    # Simpan face landmarks
                    for i in range(len(face) // 3):
                        cursor.execute("INSERT INTO face_landmarks (x, y, z, keyframe_id) VALUES (%s, %s, %s, %s)",
                                       (face[i*3], face[i*3+1], face[i*3+2], keyframe_id))

                    # Simpan left hand landmarks
                    for i in range(len(lh) // 3):
                        cursor.execute("INSERT INTO lh_landmarks (x, y, z, keyframe_id) VALUES (%s, %s, %s, %s)",
                                       (lh[i*3], lh[i*3+1], lh[i*3+2], keyframe_id))

                    # Simpan right hand landmarks
                    for i in range(len(rh) // 3):
                        cursor.execute("INSERT INTO rh_landmarks (x, y, z, keyframe_id ) VALUES (%s, %s, %s, %s)",
                                       (rh[i*3], rh[i*3+1], rh[i*3+2], keyframe_id))

                    # Show to screen
                    cv2.imshow('OpenCV Feed', mirror_cam)

                    # Break gracefully
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        close_camera()
                        break
    else:
        print("Kamera sudah dibuka. Tutup kamera sebelum membukanya lagi.")

    conn.commit()
    cursor.close()
    conn.close()

    cursor.close()
    conn.close()