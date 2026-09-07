import cv2
import numpy as np

cap = cv2.VideoCapture(0) # webcam pc

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.resize(frame, (320, 320))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Erosion
    kernel = np.ones((5,5), np.uint8)
    eroded = cv2.erode(otsu, kernel, iterations=1)
    dilated = cv2.dilate(otsu, kernel, iterations=1)
    opened = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)

    # Show the feed
    stacked_frames = np.hstack((otsu, eroded, dilated, opened, closed))

    cv2.imshow("All", stacked_frames)

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close everything
cap.release()
cv2.destroyAllWindows()