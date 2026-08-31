from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv(".env")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    cloud_inference=True
)

# Create a collection for our product embeddings
# CLIP ViT-B-32 outputs 512-dimensional vectors
client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

print("Collection 'products' created successfully!")