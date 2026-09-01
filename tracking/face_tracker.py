# tracking/face_tracker.py
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from utils.camera import Camera
MODEL_PATH = "assets/face_landmarker.task"
def create_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)
def draw_landmarks(frame, face_landmarks):
    h, w, _ = frame.shape
    for landmark in face_landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    return frame
def run_face_tracking():
    landmarker = create_landmarker()
    camera = Camera()
    print("Face tracking started. Press 'q' to quit.")
    while True:
        frame = camera.get_frame()
        if frame is None:
            print("Error: Could not read frame.")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        if result.face_landmarks:
            frame = draw_landmarks(frame, result.face_landmarks[0])
            cv2.putText(frame, "Face Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Face Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("AI Head Mouse - Face Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    run_face_tracking()