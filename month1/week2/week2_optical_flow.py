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
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
    
    # st.ravel() converts shape (N, 1) -> (N,)
    # st.ravel() == 1 creates an array like: [True, True, False, True...]
    valid_mask = st.ravel() == 1

    good_old = p0[valid_mask]
    good_new = p1[valid_mask]
    print(good_old)

    for new_point, old_point in zip(good_new, good_old):
        # Extract (x, y) coordinates as integers (OpenCV requires int for pixels)
        curr_x, curr_y = new_point.ravel().astype(int)
        prev_x, prev_y = old_point.ravel().astype(int)

        # Now draw your lines/circles:
        cv2.line(mask, (prev_x, prev_y), (curr_x, curr_y), (0, 255, 0), 2)
        cv2.circle(frame, (curr_x, curr_y), 5, (0, 0, 255), -1)





    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()