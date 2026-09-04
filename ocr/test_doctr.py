from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from pathlib import Path
import warnings

# Suppress PyTorch warnings for clean output
warnings.filterwarnings("ignore")

# Initialize docTR with a pretrained PyTorch model
# This uses a ResNet-based detection model and a CRNN-based recognition model
print("Loading docTR model...")
model = ocr_predictor(pretrained=True)

directory = Path("nutivo/cropped_images")
output_file = "nutivo/doctr_output.txt"

# Clear previous output
with open(output_file, "w", encoding="utf-8") as f:
    f.write("")

print(f"Running docTR on images in {directory}...")

for file in directory.iterdir():
    if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        try:
            # Load the image
            doc = DocumentFile.from_images([str(file)])
            
            # Run inference
            result = model(doc)
            
            # Extract text
            text = ""
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            text += word.value + " "
                            
            text = text.strip()
            
            if text:
                summary = f"File: {file.name:30s} --> {text}\n"
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(summary)
        except Exception as e:
            print(f"Failed on {file.name}: {e}")

print(f"Finished! Results written to {output_file}")
