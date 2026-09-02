# mouse/cursor_controller.py

from collections import deque
import json
import os
import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CALIBRATION_FILE = "calibration/calibration_data.json"


def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE, "r") as f:
            return json.load(f)
    return None


class CursorController:
    def __init__(self, smoothing_window=5):
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.smoothing_window = smoothing_window
        self.recent_x = deque(maxlen=smoothing_window)
        self.recent_y = deque(maxlen=smoothing_window)

        self.calibration = load_calibration()
        if self.calibration:
            print("Loaded calibration data:", self.calibration)
        else:
            print("No calibration data found, using raw full-frame range.")

    def _remap(self, value, in_min, in_max):
        if in_max - in_min == 0:
            return 0.5
        remapped = (value - in_min) / (in_max - in_min)
        return max(0.0, min(1.0, remapped))

    def move_to(self, norm_x, norm_y):
        if self.calibration:
            norm_x = self._remap(norm_x, self.calibration["x_min"], self.calibration["x_max"])
            norm_y = self._remap(norm_y, self.calibration["y_min"], self.calibration["y_max"])
        else:
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))

        self.recent_x.append(norm_x)
        self.recent_y.append(norm_y)

        smooth_x = sum(self.recent_x) / len(self.recent_x)
        smooth_y = sum(self.recent_y) / len(self.recent_y)

        screen_x = int(smooth_x * self.screen_width)
        screen_y = int(smooth_y * self.screen_height)

        try:
            pyautogui.moveTo(screen_x, screen_y)
        except pyautogui.FailSafeException:
            print("Failsafe triggered: mouse moved to screen corner. Stopping.")
            raise