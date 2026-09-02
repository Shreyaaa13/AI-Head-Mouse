# tracking/face_tracker.py

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import pyautogui

from utils.camera import Camera
from mouse.cursor_controller import CursorController

MODEL_PATH = "assets/face_landmarker.task"
NOSE_TIP_INDEX = 1

BLINK_THRESHOLD = 0.5       # blendshape score above this = eye considered closed
BLINK_COOLDOWN_SEC = 0.8    # minimum time between registered clicks


def create_landmarker():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


KEY_LANDMARK_INDICES = [1, 33, 133, 362, 263, 61, 291]  # nose tip + eye corners + mouth corners

def draw_landmarks(frame, face_landmarks):
    h, w, _ = frame.shape
    for idx in KEY_LANDMARK_INDICES:
        landmark = face_landmarks[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    return frame


def get_blink_scores(blendshapes):
    """Returns (left_blink_score, right_blink_score) from blendshape list."""
    left_score = 0.0
    right_score = 0.0
    for category in blendshapes:
        if category.category_name == "eyeBlinkLeft":
            left_score = category.score
        elif category.category_name == "eyeBlinkRight":
            right_score = category.score
    return left_score, right_score


def run_face_tracking():
    landmarker = create_landmarker()
    camera = Camera()
    cursor = CursorController()

    last_click_time = 0.0

    print("Face tracking + cursor control started. Press 'q' to quit.")
    print("Blink both eyes together to click.")
    print("Move your mouse to a screen corner at any time to force-stop (failsafe).")

    try:
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
                landmarks = result.face_landmarks[0]
                frame = draw_landmarks(frame, landmarks)

                nose = landmarks[NOSE_TIP_INDEX]
                cursor.move_to(nose.x, nose.y)

                cv2.putText(frame, "Face Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # --- Blink detection ---
                if result.face_blendshapes:
                    blendshapes = result.face_blendshapes[0]
                    left_score, right_score = get_blink_scores(blendshapes)

                    cv2.putText(frame, f"L:{left_score:.2f} R:{right_score:.2f}",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    now = time.time()
                    both_closed = left_score > BLINK_THRESHOLD and right_score > BLINK_THRESHOLD
                    cooldown_ok = (now - last_click_time) > BLINK_COOLDOWN_SEC

                    if both_closed and cooldown_ok:
                        pyautogui.click()
                        last_click_time = now
                        cv2.putText(frame, "CLICK!", (20, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        print("Blink click registered.")

            else:
                cv2.putText(frame, "No Face Detected", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("AI Head Mouse - Face Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except pyautogui.FailSafeException:
        print("\nFailsafe triggered — program stopped safely.")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_face_tracking()