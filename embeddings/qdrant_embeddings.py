from sentence_transformers import SentenceTransformer, util
from PIL import Image
from pathlib import Path
import random

model = SentenceTransformer("clip-ViT-B-32")

ref_folder  = Path("nutivo/reference_db/")
crop_folder  = Path("nutivo/cropped_images/")
reference_db  = {}

for file in ref_folder.iterdir():
    if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        product_name = file.stem 
        img = Image.open(file)
        emb = model.encode(img)
        reference_db[product_name] = emb


all_crops = [f for f in crop_folder.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
random_crops = random.sample(all_crops, min(10, len(all_crops)))

for crop in random_crops:
        product_name = crop.stem 
        img = Image.open(crop)
        emb = model.encode(img)

        best_score = 0
        best_match = None

        for key, ref in reference_db.items():
            score_match = util.cos_sim(emb, ref).item()
            if score_match > best_score:
                best_score = score_match
                best_match = key
        print(f"Crop {product_name} matched with {best_match} | Score: {best_score}")


# print(reference_db)