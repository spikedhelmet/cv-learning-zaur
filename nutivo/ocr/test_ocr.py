import cv2
import easyocr
from pathlib import Path

source = "nutivo/cropped_images"
directory = Path(source)
reader = easyocr.Reader(['en', 'az', 'tr'])
end_result = []

for file in directory.iterdir():
    if file.is_file():
        img = cv2.imread(file)
        ocr_results = reader.readtext(img)

        for (bbox, text, confidence) in ocr_results:
            # if confidence > 0.5:
            #     continue
            # print(f"Text: {text:30s} | Confidence: {confidence:.2f}")
            summary = f"File: {file} --> {text:30s} | {confidence:.2f}\n"
            end_result.append(summary)

            with open("nutivo/output.txt", "a", encoding="utf-8") as f:
                f.write(summary)
        
            # top_left = tuple(int(v) for v in bbox[0])
            # bottom_right = tuple(int(v) for v in bbox[2])
            # cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            # cv2.putText(img, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
# print(end_result)
# with open("output.txt", "a") as file:
#     file.write(end_result)