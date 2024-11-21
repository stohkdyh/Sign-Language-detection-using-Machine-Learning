import mysql.connector
import cv2
import numpy as np
import mediapipe as mp

# Mediapipe utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Kamera
cap = None
camera_open = False

# Fungsi database
def connect_to_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",  # Ubah sesuai konfigurasi MySQL Anda
        database="db_signlanguage"
    )

# Fungsi ekstraksi landmarks dari database
def fetch_landmarks_from_db(sequence_id):
    conn = connect_to_db()
    cursor = conn.cursor()

    # Fetch pose landmarks
    cursor.execute("""
        SELECT x, y, z, visibility 
        FROM pose_landmarks 
        WHERE keyframe_id IN (SELECT keyframe_id FROM keyframes WHERE sequence_id = %s)
        ORDER BY keyframe_id ASC
    """, (sequence_id,))
    pose_landmarks = cursor.fetchall()

    # Fetch face landmarks
    cursor.execute("""
        SELECT x, y, z 
        FROM face_landmarks 
        WHERE keyframe_id IN (SELECT keyframe_id FROM keyframes WHERE sequence_id = %s)
        ORDER BY keyframe_id ASC
    """, (sequence_id,))
    face_landmarks = cursor.fetchall()

    conn.close()

    # Ubah menjadi array numpy
    pose_landmarks_array = np.array(pose_landmarks).reshape(-1, 4) if pose_landmarks else np.zeros((33, 4))
    face_landmarks_array = np.array(face_landmarks).reshape(-1, 3) if face_landmarks else np.zeros((468, 3))

    return pose_landmarks_array, face_landmarks_array

# Fungsi deteksi MediaPipe
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Fungsi menggambar landmarks
def draw_styled_landmarks(image, results):
    # Gambar pose landmarks
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Gambar face landmarks
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    # Gambar hand landmarks
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

# Fungsi ekstraksi keypoints dari hasil MediaPipe
def extract_keypoints(results):
    # Pose landmarks
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    # Face landmarks
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    return pose, face

# Fungsi untuk membandingkan landmarks
def compare_landmarks(camera_pose, db_pose, camera_face, db_face):
    # Hitung Euclidean distance untuk pose
    pose_distance = np.linalg.norm(camera_pose - db_pose) if len(camera_pose) == len(db_pose) else float('inf')
    # Hitung Euclidean distance untuk face
    face_distance = np.linalg.norm(camera_face - db_face) if len(camera_face) == len(db_face) else float('inf')
    # Total distance
    return pose_distance + face_distance

# Tutup kamera
def close_camera():
    global camera_open, cap
    if camera_open:
        cv2.destroyAllWindows()
        cap.release()
        camera_open = False

# Buka kamera dan bandingkan landmarks
def open_camera_and_compare(sequence_id):
    global camera_open, cap
    if not camera_open:
        cap = cv2.VideoCapture(1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        camera_open = True

        # Ambil data dari database
        db_pose_landmarks, db_face_landmarks = fetch_landmarks_from_db(sequence_id)

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Deteksi dengan MediaPipe
                image, results = mediapipe_detection(frame, holistic)

                # Ekstraksi landmarks dari kamera
                camera_pose, camera_face = extract_keypoints(results)

                # Bandingkan landmarks
                distance = compare_landmarks(camera_pose, db_pose_landmarks.flatten(), camera_face, db_face_landmarks.flatten())
                match_status = "Match" if distance < 20 else "Not Match"  # Threshold 20 untuk kecocokan

                # Gambar landmarks dan tampilkan hasil
                draw_styled_landmarks(image, results)
                mirror_cam = cv2.flip(image, 1)
                cv2.putText(mirror_cam, f"Status: {match_status} (Distance: {distance:.2f})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if match_status == "Match" else (0, 0, 255), 2)
                cv2.imshow('OpenCV Feed', mirror_cam)

                # Keluar jika menekan tombol 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    close_camera()
                    break
    else:
        print("Kamera sudah dibuka. Tutup kamera sebelum membukanya lagi.")
