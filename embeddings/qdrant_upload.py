from qdrant_client.conversions.common_types import PointStruct
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
from pathlib import Path
from sentence_transformers import SentenceTransformer
from PIL import Image
import os

load_dotenv(".env")

model = SentenceTransformer("clip-ViT-B-32")
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    # cloud_inference=True
)
ref_folder = Path("nutivo/reference_db")
client.collection_exists(
    collection_name="products"
)
print("Collection 'products' created successfully!")

points = []
for idx, file in enumerate(ref_folder.iterdir()):
    if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg','.png']:
        img = Image.open(file)
        embedding = model.encode(img).tolist()

        point = PointStruct(
            id=idx,
            vector=embedding,
            payload={
                "product_name":file.stem,
                "filename": file.name,
                # You can add more fields here later:
                # "health_score": 7,
                # "category": "drinks",
            }
        )
        points.append(point)
        print(f"Encoded: {file.stem}")

client.upsert(collection_name="products",points=points)
print(f"Uploaded {len(points)} products to Qdrant!")