# mouse/cursor_controller.py

import pyautogui

pyautogui.FAILSAFE = True  # move mouse to top-left corner to abort
pyautogui.PAUSE = 0  # no artificial delay between pyautogui calls

SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()


class CursorController:
    def __init__(self):
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT

    def move_to(self, norm_x, norm_y):
        """
        norm_x, norm_y: floats in range [0.0, 1.0], representing
        the nose tip's position within the webcam frame.
        """
        # Clamp to [0.0, 1.0] just in case
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        screen_x = int(norm_x * self.screen_width)
        screen_y = int(norm_y * self.screen_height)

        try:
            pyautogui.moveTo(screen_x, screen_y)
        except pyautogui.FailSafeException:
            print("Failsafe triggered: mouse moved to screen corner. Stopping.")
            raise