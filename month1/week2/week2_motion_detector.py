import cv2
import numpy as np

cap = cv2.VideoCapture(0)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=False
)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.resize(frame, (640, 480))
    fg_mask = bg_subtractor.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    diluted = cv2.dilate(opened, kernel, iterations=2)

    contours, _ = cv2.findContours(diluted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0),2)
        cv2.putText(frame, "Motion", (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),2)

    diluted_bgr = cv2.cvtColor(diluted, cv2.COLOR_GRAY2BGR)

    show_stack = np.hstack(( diluted_bgr, frame))
    cv2.imshow("", show_stack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()