import cv2

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (640, 640))
    cropped_frame = resized_frame[0:200, 200:400]

    cv2.imshow("Cam feed", cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()