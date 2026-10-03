import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.resize(frame,(256, 200))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0,256])

    # Canvas for histogram visualization
    canvas = np.ones((200, 256), dtype=np.uint8) * 255
    # Next, normalize the histogram so its values fit within the 200-pixel height of your canvas
    cv2.normalize(hist, hist, 0, 200, cv2.NORM_MINMAX)

    # Draw a vertical line per bin
    for i in range(256):
        cv2.line(canvas, (i, 200), (i, 200 - int(hist[i])), 0, 1)
    
    eqalized = cv2.equalizeHist(gray_frame)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(gray_frame)

    stacked_frames = np.hstack((gray_frame, canvas, eqalized, cl1))
    cv2.imshow("all",stacked_frames)

    # Quit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()