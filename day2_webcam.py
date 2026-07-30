import cv2

cap = cv2.VideoCapture("http://192.168.0.120:8080/video")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('drone_feed.mp4', fourcc, 20.0, (640,640), False)

while True:
    success, frame = cap.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (640, 640))
    # cropped_frame = resized_frame[0:200, 200:400]
    out.write(resized_frame)

    cv2.imshow("Cam feed", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
out.release()