import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cv2.namedWindow("Canny Tuner")
cv2.createTrackbar("Low", "Canny Tuner", 50, 255, lambda x: None)
cv2.createTrackbar("High", "Canny Tuner", 150, 255, lambda x: None)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=400,
    varThreshold=16,
    detectShadows=False
)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Foreground with background subtracted
    fg_mask = bg_subtractor.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    diluted = cv2.dilate(opened, kernel, iterations=2)

    # Canny edge detector
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    low = cv2.getTrackbarPos("Low", "Canny Tuner")
    high = cv2.getTrackbarPos("High", "Canny Tuner")
    edges = cv2.Canny(blurred, low, high)

    # Combined
    result = cv2.bitwise_and(diluted, edges)

    # Metge edges for easier boundary detection
    merged_edges = cv2.dilate(result, kernel, iterations=2)
    # Get teh contours of the bitwise
    contours, _ = cv2.findContours(merged_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # if cv2.contourArea(contour) < 1000:
        # if w * h < 500:
        if cv2.arcLength(contour, closed=False) < 100:
            continue
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0),2)
        cv2.putText(frame, "Motion", (x,y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    show_stack = np.hstack(( result_bgr, frame))
    cv2.imshow("Canny Tuner", show_stack)
    # cv2.imshow("Canny Tuner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()