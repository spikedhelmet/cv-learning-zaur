import cv2
import numpy as np

# 1. Create a blank dark canvas (500x500 pixels, 3 BGR color channels)
# In Computer Vision, an image is just a 3D NumPy array!
image = np.zeros((500, 500, 3), dtype=np.uint8)

# 2. Draw a target bounding box (representing a detected drone)
# cv2.rectangle(image, top_left_pt, bottom_right_pt, color_bgr, thickness)
cv2.rectangle(image, (150, 150), (350, 350), (0, 255, 0), 2)

# 3. Add text label above the bounding box
cv2.putText(
    image, 
    "Target Drone (98%)", 
    (150, 135), 
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.6, 
    (0, 255, 0), 
    2
)

# 4. Print the raw array dimensions: (Height, Width, Color Channels)
print("Image Array Shape:", image.shape)
print("Top-left pixel RGB/BGR values:", image[150, 150])

# 5. Display the frame in a window (Press any key to close)
cv2.imshow("Defense CV - Day 1 Canvas", image)
cv2.waitKey(0)
cv2.destroyAllWindows()