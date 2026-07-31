import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.resize(frame,(640,480))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_frame],[0],None, [256],[0,256])

    canvas = np.ones((200,256), dtype=np.uint8) * 255
    cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)
    

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()