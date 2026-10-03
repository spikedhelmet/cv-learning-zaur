import cv2
import numpy as np

cap = cv2.VideoCapture(0)

feature_params = dict(
    maxCorners=100,       # Maximum number of corners to track
    qualityLevel=0.3,     # Minimal accepted quality of image corners (0.0 - 1.0)
    minDistance=7,        # Minimum distance between returned corners
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),     # Size of the search window at each pyramid level
    maxLevel=2,           # Pyramid levels (allows tracking larger movements)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

success, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
mask = np.zeros_like(first_frame)

while True:
    success, frame = cap.read()  
    if not success:
        continue
    
    # frame = cv2.resize(frame, (640, 480))
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)
    
    # st.ravel() converts shape (N, 1) -> (N,)
    # st.ravel() == 1 creates an array like: [True, True, False, True...]
    # valid_mask = st.ravel() == 1

    if p0 is not None:
        good_old = p0[st == 1]
    if p1 is not None:
        good_new = p1[st == 1]

    for new_point, old_point in zip(good_new, good_old):
        # Extract (x, y) coordinates as integers (OpenCV requires int for pixels)
        curr_x, curr_y = new_point.ravel().astype(int)
        prev_x, prev_y = old_point.ravel().astype(int)

        # Now draw your lines/circles:
        cv2.line(mask, (prev_x, prev_y), (curr_x, curr_y), (0, 255, 0), 2)
        cv2.circle(frame, (curr_x, curr_y), 5, (0, 0, 255), -1)

    output = cv2.add(frame,mask) # combines the mask with the frame

    cv2.imshow("Optical Flow!", output)

    # 
    prev_gray = current_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if len(p0) < 10:
       p0 = cv2.goodFeaturesToTrack(current_gray, mask=None, **feature_params)
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()