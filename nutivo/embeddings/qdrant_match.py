import sys
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from qdrant_client import QdrantClient
from pathlib import Path
import random
from dotenv import load_dotenv
import os


model = SentenceTransformer("clip-ViT-B-32")
load_dotenv(".env")
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    # cloud_inference=True
)

if not client.collection_exists(collection_name="products"):
    sys.exit()

ref_folder  = Path("nutivo/reference_db/")
crop_folder  = Path("nutivo/cropped_images/")
# reference_db  = {}

# for file in ref_folder.iterdir():
#     if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
#         product_name = file.stem 
#         img = Image.open(file)
#         emb = model.encode(img)
#         reference_db[product_name] = emb

all_crops = [f for f in crop_folder.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
random_crops = random.sample(all_crops, min(10, len(all_crops)))

for crop in random_crops:
        product_name = crop.stem 
        query_img = Image.open(crop)
        query_emb = model.encode(query_img).tolist()

        results = client.query_points(
            collection_name="products",
            query=query_emb,
            limit=3,  # return top 3 matches
        )


        best_match = results.points[0]

        if best_match.payload:
            if best_match.score < 0.85:
                print(f"Crop {product_name:25s} | UNKNOWN PRODUCT (Highest score was {best_match.score:.2f})")
            else:
                print(f"Crop {product_name} | Match: {best_match.payload['product_name']} | Score: {best_match.score:.2f}")



# print(reference_db)