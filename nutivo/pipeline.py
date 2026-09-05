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
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yolo_model = YOLO(os.path.join(BASE_DIR, "sku-shelf.pt"))
clip_model = SentenceTransformer("clip-ViT-B-32")

# Connect to Qdrant
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

    
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

def identify_product(crop, similiraty_threshold=0.85):
    query_emb = clip_model.encode(crop).tolist()

    results = client.query_points(
        collection_name="products",
        query=query_emb,
        limit=3,  # return top 3 matches
    )
    best_match = results.points[0]

    if best_match.payload:
        if best_match.score < similiraty_threshold:
            return {
                "product_name": "unknown",
                "confidence": best_match.score,
                "status": "unknown"
            }
        else:            
            return {
                "product_name": best_match.payload['product_name'],
                "confidence": best_match.score,
                "status": "matched"
            }

    return {
        "product_name": "unknown",
        "confidence": best_match.score,
        "status": "unknown_no_payload"
    }
        

def scan_shelf(img_path):
    detections = detect_products(img_path)
    cv2_img = cv2.imread(img_path)
    results = []

    for det in detections:
        bbox = det['bbox']
        x, y, x2, y2 = bbox

        match = identify_product(det['crop'])
        product_name = match['product_name']
        # match_conf = match['confidence']
        match["bbox"] = det["bbox"]
        match["detection_conf"] = float(det["detection_conf"])
        results.append(match)
        
        det_color = (0, 0, 255) if product_name=="unknown" else (0, 255, 0)
        cv2.putText(cv2_img, str(product_name), (int(x),int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, det_color, 3)
        cv2.rectangle(cv2_img, (int(x), int(y)), (int(x2), int(y2)), det_color, 2)
    
    # cv2.imwrite("annotated_output.jpg", cv2_img)
    _ , img_bytes = cv2.imencode('.jpg',cv2_img)
    return results, img_bytes.tobytes()


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1] if len(sys.argv) > 1 else "nutivo/source_images/shelf_01.jpg"
    
    print(f"Scanning: {image_path}")
    results = scan_shelf(image_path)
    
    print(f"\nFound {len(results)} products:\n")
    for r in results:
        if r["status"] == "matched":
            print(f"  {r['product_name']:25s} | Confidence: {r['confidence']:.2f}")
        else:
            print(f"  {'UNKNOWN':25s} | Best score: {r['confidence']:.2f}")