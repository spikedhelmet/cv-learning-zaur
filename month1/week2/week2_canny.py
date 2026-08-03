import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.resize (frame, (640,480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gaussian = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(gaussian, 50, 150)

    stack = np.hstack((gray, gaussian, canny))
    cv2.imshow("",stack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()