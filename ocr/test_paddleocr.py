from paddleocr import PaddleOCR
from pathlib import Path

# Initialize PaddleOCR
# use_textline_orientation=True helps with tilted text. 
# lang='en' is highly optimized. Most brands (Lays, Doritos, ICE, EFES) use Latin characters.
ocr = PaddleOCR(use_textline_orientation=True, lang='en')

directory = Path("nutivo/cropped_images")
output_file = "nutivo/paddle_output.txt"

# Clear previous output if it exists
with open(output_file, "w", encoding="utf-8") as f:
    f.write("")

print("Starting PaddleOCR...")

for file in directory.iterdir():
    if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        # PaddleOCR takes the file path string directly
        result = ocr.predict(str(file))
        
        # PaddleOCR returns [None] if no text is found
        if not result or not result[0]:
            continue
            
        for line in result[0]:
            # Each line looks like: [[[x1,y1], [x2,y2], [x3,y3], [x4,y4]], ('text', confidence)]
            bbox, (text, confidence) = line
            
            summary = f"File: {file.name:30s} --> {text} | {confidence:.2f}\n"
            
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(summary)

print(f"Finished! Results written to {output_file}")
