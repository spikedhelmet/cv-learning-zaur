import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        continue

    roi_box = cv2.selectROI("Select the ROI", frame, fromCenter=False, showCrosshair=True)
    x, y, w, h = roi_box

    # Extract the selection
    roi = frame[y:y+h, x:x+w]
    print(roi)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    frame[y:y+h, x:x+w] = edges_bgr

    cv2.imshow("frame", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('r'):
        break

cap.release()
cv2.destroyAllWindows()