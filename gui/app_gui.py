# gui/app_gui.py

import tkinter as tk
from tkinter import messagebox
import threading
import os

from calibration.calibrate import run_calibration, CALIBRATION_FILE
from tracking.face_tracker import run_face_tracking


class HeadMouseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Head Mouse")
        self.root.geometry("400x250")
        self.root.resizable(False, False)

        self.tracking_thread = None

        title_label = tk.Label(root, text="AI Head Mouse", font=("Segoe UI", 16, "bold"))
        title_label.pack(pady=(20, 10))

        self.status_label = tk.Label(root, text="Status: Idle", font=("Segoe UI", 11), fg="gray")
        self.status_label.pack(pady=(0, 20))

        self.calibrate_btn = tk.Button(
            root, text="Calibrate", font=("Segoe UI", 11),
            width=20, command=self.on_calibrate
        )
        self.calibrate_btn.pack(pady=5)

        self.start_btn = tk.Button(
            root, text="Start Tracking", font=("Segoe UI", 11),
            width=20, command=self.on_start_tracking
        )
        self.start_btn.pack(pady=5)

        info_label = tk.Label(
            root,
            text="Blink both eyes to click.\nHold both eyes closed 2s to stop.",
            font=("Segoe UI", 9), fg="gray", justify="center"
        )
        info_label.pack(pady=(20, 0))

    def set_status(self, text, color="black"):
        self.status_label.config(text=f"Status: {text}", fg=color)
        self.root.update_idletasks()

    def on_calibrate(self):
        self.set_status("Calibrating... follow webcam window", "orange")
        self.calibrate_btn.config(state="disabled")
        self.start_btn.config(state="disabled")

        def task():
            run_calibration()
            self.set_status("Idle", "gray")
            self.calibrate_btn.config(state="normal")
            self.start_btn.config(state="normal")

        threading.Thread(target=task, daemon=True).start()

    def on_start_tracking(self):
        if not os.path.exists(CALIBRATION_FILE):
            messagebox.showwarning(
                "No Calibration",
                "No calibration data found. Please run Calibrate first for best accuracy."
            )

        self.set_status("Tracking active — blink to click, hold 2s to stop", "green")
        self.calibrate_btn.config(state="disabled")
        self.start_btn.config(state="disabled")

        def task():
            run_face_tracking()
            self.set_status("Idle", "gray")
            self.calibrate_btn.config(state="normal")
            self.start_btn.config(state="normal")

        self.tracking_thread = threading.Thread(target=task, daemon=True)
        self.tracking_thread.start()


def main():
    root = tk.Tk()
    app = HeadMouseGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()