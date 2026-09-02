# calibration/calibrate.py

import cv2
import json
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

from utils.camera import Camera

MODEL_PATH = "assets/face_landmarker.task"
NOSE_TIP_INDEX = 1
CALIBRATION_FILE = "calibration/calibration_data.json"

# Order matters: center first (baseline), then extremes
STEPS = [
    ("CENTER", "Look at the CENTER of your screen"),
    ("LEFT", "Turn your head slightly LEFT (toward screen's left edge)"),
    ("RIGHT", "Turn your head slightly RIGHT (toward screen's right edge)"),
    ("UP", "Tilt your head slightly UP (toward screen's top edge)"),
    ("DOWN", "Tilt your head slightly DOWN (toward screen's bottom edge)"),
]


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


def run_calibration():
    landmarker = create_landmarker()
    camera = Camera()

    captured = {}
    step_index = 0
    latest_nose = None

    print("Calibration started.")
    print("For each prompt: look in that direction, then press SPACE to capture.")
    print("Press 'q' at any time to cancel.\n")

    while step_index < len(STEPS):
        frame = camera.get_frame()
        if frame is None:
            print("Error: Could not read frame.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        label, instruction = STEPS[step_index]

        if result.face_landmarks:
            nose = result.face_landmarks[0][NOSE_TIP_INDEX]
            latest_nose = (nose.x, nose.y)
            cv2.putText(frame, f"Step: {label}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, instruction, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, "Press SPACE to capture", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            latest_nose = None
            cv2.putText(frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Calibration cancelled.")
            camera.release()
            cv2.destroyAllWindows()
            return

        if key == ord(' ') and latest_nose is not None:
            captured[label] = latest_nose
            print(f"Captured {label}: x={latest_nose[0]:.3f}, y={latest_nose[1]:.3f}")
            step_index += 1

    camera.release()
    cv2.destroyAllWindows()

    if len(captured) < len(STEPS):
        print("Calibration incomplete, not saving.")
        return

    # Build min/max ranges from captured points
    calibration_data = {
        "x_min": min(captured["LEFT"][0], captured["CENTER"][0]),
        "x_max": max(captured["RIGHT"][0], captured["CENTER"][0]),
        "y_min": min(captured["UP"][1], captured["CENTER"][1]),
        "y_max": max(captured["DOWN"][1], captured["CENTER"][1]),
    }

    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calibration_data, f, indent=2)

    print(f"\nCalibration complete! Saved to {CALIBRATION_FILE}")
    print(calibration_data)


if __name__ == "__main__":
    run_calibration()