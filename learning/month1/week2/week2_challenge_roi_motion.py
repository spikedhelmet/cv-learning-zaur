import cv2
import numpy as np

cap = cv2.VideoCapture(0)

success, first_frame = cap.read()
roi_box = cv2.selectROI("ROI",first_frame, fromCenter=False, showCrosshair=True)
x, y, w, h = roi_box

bg_sub = cv2.createBackgroundSubtractorMOG2(  
    history=500,
    varThreshold=16,
    detectShadows=True
)

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

while True:
    success, frame = cap.read()
    if not success:
        continue

    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    fg_mask = bg_sub.apply(gray)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    diluted = cv2.dilate(opened, kernel, iterations=2)

    # Canny edge detector
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Combined
    result = cv2.bitwise_and(diluted, edges)

    # Merge edges for easier boundary detection
    merged_edges = cv2.dilate(result, kernel, iterations=2)

    fg_bgr = cv2.cvtColor(merged_edges, cv2.COLOR_GRAY2BGR)

    # frame[y:y+h, x:x+w] = fg_bgr

    
    contours, _ = cv2.findContours(merged_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False

    for contour in contours:
        # if cv2.contourArea(contour) < 1000:
        # if w * h < 500:
        if cv2.arcLength(contour, closed=False) < 100:
            motion_detected = False
            continue
        motion_detected = True
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255),2)
        __draw_label(roi, "ALERT: Motion in Zone!", (20,20), (255,255,255))

    box_color = (0, 0, 255) if motion_detected else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x+w, y+h), box_color,2)

    # combined = cv2.add(frame, mask)
    cv2.imshow("ROI", frame)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('r'):
        roi_box = cv2.selectROI("ROI", frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi_box # need to unpack the box and update the coords

cap.release()
cv2.destroyAllWindows()