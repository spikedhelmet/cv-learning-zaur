import cv2
import os
import sys
from PIL import Image
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv(".env")

# Load both models once at startup
yolo_model = YOLO("nutivo/sku-shelf.pt")
clip_model = SentenceTransformer("clip-ViT-B-32")

# Connect to Qdrant
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

if not client.collection_exists(collection_name="products"):
    sys.exit()

    

def detect_products(image_path, confidence_threshold=0.6, min_size=20):
    results = yolo_model.predict(image_path, batch=8) 
    detected_product_list = []

    for shelf_index, result in enumerate(results, start=1):
        img = result.orig_img
        img_height, img_width = img.shape[:2]

        boxes_xyxy = result.boxes.xyxy
        conf_numbers = result.boxes.conf.cpu().numpy()
        # good_results = list(filter(lambda x: x > 0.6, conf_numbers))
        
        for box, conf in zip(boxes_xyxy, conf_numbers):
            if conf < confidence_threshold:
                continue

            x1, y1, x2, y2 = box.tolist()
            modded_x1 = max(0, int(x1))
            modded_x2 = min(img_width, int(x2))
            modded_y1 = max(0, int(y1))
            modded_y2 = min(img_height, int(y2))

            if (modded_x2 - modded_x1) < 20 or (modded_y2 - modded_y1) < 20:
                continue

            cropped_img = img[modded_y1:modded_y2, modded_x1:modded_x2]
            # We do this cos yolo uses bgr but pil needs rgb
            crop_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)
            detection = {
                "crop": pil_img,
                "bbox": [modded_x1, modded_y1, modded_x2, modded_y2],
                "detection_conf": conf
            }
            detected_product_list.append(detection)

    print("Detected Products:",detected_product_list)
    return detected_product_list

def identify_product(crop_pil, similiraty_threshold=0.85):
    """
    Encodes a crop with CLIP and queries Qdrant for the best match.
    
    Returns:
        dict with "product_name", "confidence", and "status"
    """
    # 1. Encode the crop with CLIP
    # 2. Query Qdrant
    # 3. Check if the top score is above the threshold
    # 4. Return the result as a dictionary