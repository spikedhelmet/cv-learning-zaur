import uuid
from fastapi import Form
from qdrant_client.conversions.common_types import PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from qdrant_client import QdrantClient
import sys
import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

load_dotenv(".env")


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import scan_shelf

app = FastAPI(title="Nutivo Shelf Scanner API")

# Allow frontend (running on different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend's domain
    allow_methods=["*"],
    allow_headers=["*"],
)

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

model = SentenceTransformer("clip-ViT-B-32")

@app.post("/scan")
async def scan_endpoint(file: UploadFile = File(...)):
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_file.write(await file.read())
    temp_file.close()

    results, img_bytes = scan_shelf(temp_file.name)
    # 3. Clean up the temp file
    os.unlink(temp_file.name)
    
    # 4. Return the results as JSON
    return results

@app.get("/products")
async def list_products():
    products = []
    records, next_page_offset = client.scroll(
        collection_name="products",
        limit=100,
        with_payload=True,
        with_vectors=False)
    
    for record in records:
        payload = record.payload
        product_name = payload.get("product_name")
        filename = payload.get("filename")
        product = {
            "product_name":product_name,
            "filename":filename
        }
        products.append(product)
        
        print(f"Product: {product_name} | Image File: {filename}")
    
    return products

@app.post("/scan/annotated")
async def scan_annotated_endpoint(file: UploadFile = File(...)):
    temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    temp_file.write(await file.read())
    temp_file.close()

    results, img_bytes = scan_shelf(temp_file.name)
    # Clean up the temp file
    os.unlink(temp_file.name)
    
    # Return the results as an image
    return Response(content=img_bytes, media_type="image/jpeg")

@app.post("/products/add")
async def add_products(file: UploadFile = File(...), product_name: str = Form(...)):
    points = []
    img = Image.open(file.file)
    embedding = model.encode(img).tolist()

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding,
        payload={
            "product_name": product_name,
            "filename": file.filename,
            }
    )
    points.append(point)
    client.upsert(collection_name="products",points=points)
    print(f"Uploaded {len(points)} products to Qdrant!")
