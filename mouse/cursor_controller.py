# mouse/cursor_controller.py

from collections import deque
import pyautogui

pyautogui.FAILSAFE = True  # move mouse to top-left corner to abort
pyautogui.PAUSE = 0  # no artificial delay between pyautogui calls

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


class CursorController:
    def __init__(self, smoothing_window=5):
        """
        smoothing_window: how many recent positions to average over.
        Higher = smoother but more lag. Lower = snappier but more jitter.
        """
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        self.smoothing_window = smoothing_window
        self.recent_x = deque(maxlen=smoothing_window)
        self.recent_y = deque(maxlen=smoothing_window)

    def move_to(self, norm_x, norm_y):
        """
        norm_x, norm_y: floats in range [0.0, 1.0], representing
        the nose tip's position within the webcam frame.
        """
        # Clamp to [0.0, 1.0] just in case
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        # Add to rolling history
        self.recent_x.append(norm_x)
        self.recent_y.append(norm_y)

        # Average over recent frames for smoothing
        smooth_x = sum(self.recent_x) / len(self.recent_x)
        smooth_y = sum(self.recent_y) / len(self.recent_y)

        screen_x = int(smooth_x * self.screen_width)
        screen_y = int(smooth_y * self.screen_height)

        try:
            pyautogui.moveTo(screen_x, screen_y)
        except pyautogui.FailSafeException:
            print("Failsafe triggered: mouse moved to screen corner. Stopping.")
            raise