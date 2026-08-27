import cv2
import numpy as np

# 1. Load an actual image from disk
# In TypeScript/Node, you might use 'fs.readFileSync' and a JPEG decoder.
# In OpenCV, cv2.imread automatically reads the bytes and decodes it into a 3D NumPy array!
image = cv2.imread("test_image.jpg")

# Always check if the image loaded successfully (null check!)
if image is None:
    print("Error: Could not load image. Make sure 'test_image.jpg' is in the directory.")
    exit()

# 2. Get the dimensions of the image
# shape returns a tuple: (Height, Width, Channels)
h, w, c = image.shape
print(f"Loaded Image - Height: {h}px, Width: {w}px, Channels: {c}")

# 3. Draw a bounding box around a hypothetical target
# Let's say our drone (or bus in this case) is detected at these coordinates
top_left = (50, 200)
bottom_right = (250, 400)

# Draw a red rectangle (Remember: B-G-R color space! Red is (0, 0, 255))
cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 3)

# Add a label above the bounding box
cv2.putText(
    image, 
    "Target Detected", 
    (50, 185), # Slightly above the top_left y-coordinate
    cv2.FONT_HERSHEY_SIMPLEX, 
    0.8, 
    (0, 0, 255), 
    2
)

# 4. Display the image
cv2.imshow("Defense CV - Day 2 (Real Image)", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
