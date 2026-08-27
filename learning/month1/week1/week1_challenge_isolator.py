import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    success,frame = cap.read()

    if not success:
        continue

    frame = cv2.resize(frame,(640,640))


    # HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_tennis_ball = np.array([29, 120, 40])
    upper_tennis_ball = np.array([64, 255, 255])
    # need to isolate orange comb
    mask = cv2.inRange(hsv_frame, lower_tennis_ball, upper_tennis_ball)
    
    kernel = np.ones((7,7), np.uint8)

    # eroded = cv2.erode(otsu, None, iterations=2)
    # dilated = cv2.dilate(eroded, None, iterations=2)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)

        if cv2.contourArea(cont) > 500:
            cv2.rectangle(frame, (x,y), (x+w, y+h),(0, 255, 0), 2)

    # img = cv2.drawContours(frame, contours, -1, (0,255,0), 2)

    cv2.imshow("",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Close everything
cap.release()
cv2.destroyAllWindows()