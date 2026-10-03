import cv2
import numpy as np

cap = cv2.VideoCapture(0)

success, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

feature_params = dict(
    maxCorners=100,       # Maximum number of corners to track
    qualityLevel=0.3,     # Minimal accepted quality of image corners (0.0 - 1.0)
    minDistance=7,        # Minimum distance between returned corners
    blockSize=7
)

# We record the original position here
p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
mask = np.zeros_like(first_frame) # We make a mask to draw lines on

lk_params = dict(
    winSize=(15, 15),     # Size of the search window at each pyramid level
    maxLevel=2,           # Pyramid levels (allows tracking larger movements)
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)


# Source - https://stackoverflow.com/a/54616857
# Posted by TeddybearCrisis, modified by community. See post 'Timeline' for change history
# Retrieved 2026-08-06, License - CC BY-SA 4.0

def __draw_label(img, text, pos, bg_color):
   font_face = cv2.FONT_HERSHEY_SIMPLEX
   scale = 1.2
   color = (0, 0, 0)
   thickness = cv2.FILLED
   margin = 2
   txt_size = cv2.getTextSize(text, font_face, scale, thickness)

   end_x = pos[0] + txt_size[0][0] + margin
   end_y = pos[1] - txt_size[0][1] - margin

   cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
   cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

# __draw_label(mask, 'Hello ZORT', (20,20), (255,0,0))

while True:
    success, frame = cap.read()  
    if not success:
        continue
    
    # frame = cv2.resize(frame, (640, 480))
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # The point it moves to
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)
    
    # st.ravel() converts shape (N, 1) -> (N,)
    # st.ravel() == 1 creates an array like: [True, True, False, True...]
    # valid_mask = st.ravel() == 1

    if p0 is not None:
        good_old = p0[st == 1]
    if p1 is not None:
        good_new = p1[st == 1]

    # Average Displacement
    diff = good_new - good_old
    mean_dx = np.mean(diff[:, 0])
    mean_dy = np.mean(diff[:, 1])

    magnitude = np.sqrt(mean_dx**2 + mean_dy**2)
    # avg_displacement = np.mean(magnitude)

    if magnitude > 3.0:
        if abs(mean_dx) > abs(mean_dy):
            direction = 'RIGHT' if mean_dx > 0 else 'LEFT'

        else:
            direction = 'DOWN' if mean_dy > 0 else 'UP'
        
        __draw_label(mask, f'Moving {direction}', (20,20), (255,255,255))

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
       mask = np.zeros_like(frame) # Wipe the clear glass sheet clean!

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()