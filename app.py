import cv2
from utils.camera import Camera

camera = Camera()

while True:

    frame = camera.get_frame()

    if frame is None:
        break

    cv2.imshow("AI Head Mouse", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()