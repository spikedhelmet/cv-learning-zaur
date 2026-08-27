import cv2
import numpy as np

cap = cv2.VideoCapture(0)

bg_sub_mog = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=16,
    detectShadows=True
)

bg_sub_knn = cv2.createBackgroundSubtractorKNN(
    history=500,
    dist2Threshold=400.0,
    detectShadows=True
)

while True:
    success, frame = cap.read()

    if not success:
        continue

    frame = cv2.resize(frame, (640,480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    fg_mask_mog = bg_sub_mog.apply(frame)
    fg_mask_knn = bg_sub_knn.apply(frame)

    # Stack
    show_stack = np.hstack((gray,fg_mask_mog, fg_mask_knn))

    cv2.imshow("",show_stack)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()