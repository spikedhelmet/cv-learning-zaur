Ran command: `python day2_webcam.py`
Edited day2_webcam.py
Viewed day2_webcam.py:1-16
Ran command: `python day2_webcam.py`
Edited ai_context.md

Perfectly executed! You sliced exactly 200 rows and 200 columns to get a perfect `200x200` square. By doing this, you've essentially just learned the exact preprocessing pipeline (Grayscale -> Resize -> Crop) that gets fed into neural networks.

I've updated `ai_context.md` with our progress!

### The Final Month 1 Challenge: Saving Video

To wrap up the Month 1 fundamentals, we need to know how to save these manipulated frames to disk. In defense systems, you don't just detect drones—you record the incident for playback and auditing.

Your challenge is to modify `day2_webcam.py` to **record your resized 640x640 grayscale feed to a video file**.

Here are the 4 steps to do it:

**1. Create a Codec (Before the loop):**
In TypeScript/Web, the browser handles video codecs for you (like H.264). In OpenCV, you have to explicitly define it using a "FourCC" code. Add this line:

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # This tells OpenCV to use the MP4 codec
```

**2. Initialize the VideoWriter (Before the loop):**
Create the writer object.

```python
# The syntax: cv2.VideoWriter(filename, codec, fps, (width, height), isColor)
out = cv2.VideoWriter('drone_feed_record.mp4', fourcc, 20.0, (640, 640), False)
```

_⚠️ **Major Gotcha:** Remember how NumPy matrices are `(Height, Width)`? The `VideoWriter` expects `(Width, Height)`. In this case, it's 640x640 so it doesn't matter, but keep this in mind for the future! We also pass `False` at the end because we are writing Grayscale (1-channel), not color._

**3. Write the Frame (Inside the loop):**
Right after you create `resized_frame` (the 640x640 gray one), save it to the file by calling:

```python
out.write(resized_frame)
```

_(You can delete your cropping logic for this, or just change the VideoWriter dimensions to `200, 200` and write the `cropped_frame` instead!)_

**4. Cleanup (After the loop):**
Just like `cap.release()`, you must close the file writer so it saves properly:

```python
out.release()
```

Give it a shot, run the script for a few seconds, press 'q' to quit, and then check your folder for the new MP4 file! Let me know when you get it working!
