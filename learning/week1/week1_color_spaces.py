import cv2
import numpy as np

cap = cv2.VideoCapture("http://192.168.0.120:8080/video")

while True:
    success, frame = cap.read()
    if not success:
        continue
    
    frame = cv2.resize(frame,(640,360))
    
    # Convert BGR to HSV
    # hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Simple binary threshold: pixels > 127 become white (255), rest become black (0)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # Otsu's method: OpenCV auto picks best treshold value
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Adaptive treshold: threshold varies across the image based on local regions
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)


    # Define a color range to isolate. This example targets blue objects.
    # HSV ranges in OpenCV: H(0-179), S(0-255), V(0-255)
    # lower_blue = np.array([100,50,50])
    # upper_blue = np.array([130,255,255])

    # Create a binary mask: white where blue exists, black everywhere else
    # mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    # Fake 3 dimension mask in order to combine with others
    # mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Apply the mask to see only blue parts
    # result = cv2.bitwise_and(frame,frame, mask=mask)

    # Show all 3 views
    # cv2.imshow("original", frame)
    # cv2.imshow("mask", mask)
    # cv2.imshow("filtered", result)

    # Combine all three views in one
    stacked_frames = np.hstack((gray, binary, otsu, adaptive))

    cv2.imshow("Original | Mask | Filtered", stacked_frames)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()