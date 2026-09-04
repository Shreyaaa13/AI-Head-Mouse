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
KEY_LANDMARK_INDICES = [1, 33, 133, 362, 263, 61, 291]  # nose tip + eye corners + mouth corners

BLINK_THRESHOLD = 0.5       # blendshape score above this = eye considered closed
BLINK_COOLDOWN_SEC = 0.8    # minimum time between registered clicks
HOLD_STOP_SEC = 2.0         # how long both eyes must stay closed to force-stop


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


def draw_landmarks(frame, face_landmarks):
    h, w, _ = frame.shape
    for idx in KEY_LANDMARK_INDICES:
        landmark = face_landmarks[idx]
        x, y = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    return frame


def get_blink_scores(blendshapes):
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
    eyes_closed_since = None  # timestamp when continuous closure started

    print("Face tracking + cursor control started.")
    print("Blink both eyes briefly to click.")
    print("Hold both eyes closed for 2 seconds to force-stop.")
    print("(Backup) Press 'q' or move mouse to a screen corner to stop.")

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

                if result.face_blendshapes:
                    blendshapes = result.face_blendshapes[0]
                    left_score, right_score = get_blink_scores(blendshapes)

                    cv2.putText(frame, f"L:{left_score:.2f} R:{right_score:.2f}",
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    now = time.time()
                    both_closed = left_score > BLINK_THRESHOLD and right_score > BLINK_THRESHOLD

                    if both_closed:
                        if eyes_closed_since is None:
                            eyes_closed_since = now

                        closed_duration = now - eyes_closed_since

                        # Show a progress hint once it's been closed a little while
                        if closed_duration > 0.3:
                            cv2.putText(frame, f"Holding: {closed_duration:.1f}s / {HOLD_STOP_SEC}s",
                                        (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                        if closed_duration >= HOLD_STOP_SEC:
                            print("\nHeld blink detected — stopping safely.")
                            break

                        # Quick click only fires on the way to becoming a hold;
                        # cooldown prevents repeat clicks while eyes stay closed
                        cooldown_ok = (now - last_click_time) > BLINK_COOLDOWN_SEC
                        if cooldown_ok and eyes_closed_since == now:
                            pass  # first frame of closure; wait to see if it's a click or a hold

                    else:
                        # Eyes just opened — if it was a short closure, treat as a click
                        if eyes_closed_since is not None:
                            closed_duration = now - eyes_closed_since
                            cooldown_ok = (now - last_click_time) > BLINK_COOLDOWN_SEC
                            if closed_duration < HOLD_STOP_SEC and cooldown_ok:
                                pyautogui.click()
                                last_click_time = now
                                cv2.putText(frame, "CLICK!", (20, 200),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                print("Blink click registered.")
                        eyes_closed_since = None

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