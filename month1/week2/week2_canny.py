import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cv2.namedWindow("Canny Tuner")
def on_low_slider_change(new_value):
    print(f"The low slider moved to {new_value}")

cv2.createTrackbar("Low", "Canny Tuner", 50, 255, on_low_slider_change)
cv2.createTrackbar("High", "Canny Tuner", 150, 255, lambda x: None)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.resize (frame, (640,480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    # canny = cv2.Canny(blurred, 50, 150)

    low = cv2.getTrackbarPos("Low", "Canny Tuner")
    high = cv2.getTrackbarPos("High", "Canny Tuner")
    edges = cv2.Canny(blurred, low, high)

    stack = np.hstack((gray, blurred, edges))
    cv2.imshow("Canny Tuner",stack)
    # cv2.imshow("", blurred)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()