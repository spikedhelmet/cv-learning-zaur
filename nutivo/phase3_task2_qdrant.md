# Phase 3.2 — Qdrant Vector Database

---

## Goal

Replace the in-memory Python dictionary with a proper vector database (Qdrant Cloud). By the end of this task, you will be able to upload product embeddings with metadata to Qdrant, and query it to find the closest matching product for any given crop.

---

## Concept: Why a Vector Database?

In your current `embeddings.py`, you loop through every single reference embedding and calculate cosine similarity one by one. This is called a **linear scan** — it works for 10 products, but if you had 10,000 products, it would be painfully slow.

A vector database like Qdrant uses a data structure called **HNSW (Hierarchical Navigable Small World)** — a graph where similar vectors are connected by edges. When you search, instead of checking all 10,000 vectors, the algorithm "walks" through the graph toward the most similar region, checking only ~50-100 vectors total. This is called **Approximate Nearest Neighbor (ANN)** search.

Beyond speed, Qdrant also gives you:
- **Persistence**: Your embeddings survive restarts (no re-encoding every run).
- **Metadata filtering**: Search only within a category (e.g., "show me only drinks").
- **Scalability**: Works the same whether you have 100 or 100 million vectors.

---

## Technical Mechanics: Qdrant Concepts

| Concept      | What it means                                                                 |
|--------------|-------------------------------------------------------------------------------|
| **Collection** | A named container for vectors (like a table in SQL). You'll create one called `products`. |
| **Point**      | A single entry: a vector (the embedding) + a payload (metadata like name, health score). |
| **Payload**    | Key-value metadata attached to each point. Searchable and filterable.           |
| **Distance**   | The similarity metric. We'll use `Cosine` (same as what you used with `util.cos_sim`). |

---

## Step-by-Step Task

### 1. Install Dependencies

```bash
pip install qdrant-client python-dotenv
```

### 2. Create a `.env` File

Create a file called `nutivo/.env` with your Qdrant Cloud credentials:

```
QDRANT_URL=https://your-cluster-url.aws.cloud.qdrant.io
QDRANT_API_KEY=your-api-key-here
```

Never commit this file to git. If you have a `.gitignore`, add `.env` to it.

### 3. Connect to Qdrant and Create a Collection

Create a new script `nutivo/embeddings/qdrant_upload.py`:

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from dotenv import load_dotenv
import os

load_dotenv("nutivo/.env")

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

# Create a collection for our product embeddings
# CLIP ViT-B-32 outputs 512-dimensional vectors
client.recreate_collection(
    collection_name="products",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

print("Collection 'products' created successfully!")
```

Run this. If it prints the success message, your connection to Qdrant Cloud works.

### 4. Upload Reference Embeddings

Now combine CLIP + Qdrant. Load your reference images, generate embeddings, and upload them as **points** with metadata:

```python
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv("nutivo/.env")

model = SentenceTransformer("clip-ViT-B-32")
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

ref_folder = Path("nutivo/reference_db/")
points = []

for idx, file in enumerate(ref_folder.iterdir()):
    if file.is_file() and file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        img = Image.open(file)
        embedding = model.encode(img).tolist()  # Qdrant expects a plain Python list, not numpy

        point = PointStruct(
            id=idx,                          # unique integer ID
            vector=embedding,                # the 512-dim vector
            payload={                        # metadata
                "product_name": file.stem,
                "filename": file.name,
                # You can add more fields here later:
                # "health_score": 7,
                # "category": "drinks",
            }
        )
        points.append(point)
        print(f"Encoded: {file.stem}")

# Upload all points in one batch
client.upsert(collection_name="products", points=points)
print(f"Uploaded {len(points)} products to Qdrant!")
```

### 5. Query Qdrant with a Crop

Now, instead of your manual `for ref_name, ref_emb in reference_db.items()` loop, you let Qdrant do the search:

```python
# Load a query image (a crop from your shelf)
query_img = Image.open("nutivo/cropped_images/shelf_03_crop_007.jpg")
query_emb = model.encode(query_img).tolist()

# Search Qdrant for the closest match
results = client.query_points(
    collection_name="products",
    query=query_emb,
    limit=3,  # return top 3 matches
)

for result in results.points:
    print(f"Match: {result.payload['product_name']} | Score: {result.score:.4f}")
```

Notice: you no longer loop through every reference. Qdrant returns the top matches instantly.

---

## Checkpoint Questions

1. Why does Qdrant need `.tolist()` on the numpy array but `util.cos_sim` didn't?
2. What happens if you upload two points with the same `id`? (Hint: look up what `upsert` means.)
3. If you switch from `clip-ViT-B-32` (512-dim) to `clip-ViT-L-14` (768-dim), what do you need to change in Qdrant?

---

## Challenge (No Guidance)

**Build the Full Qdrant Matcher**

Write a single script `nutivo/embeddings/qdrant_match.py` that:
1. Connects to Qdrant Cloud.
2. Takes 10 random crops from `nutivo/cropped_images/`.
3. For each crop, queries Qdrant and prints the top match + score.
4. If the top score is below 0.85, prints "UNKNOWN PRODUCT" instead.

Compare the output to your old dictionary-based `embeddings.py`. The results should be identical, but Qdrant is doing the search for you.

---

## Supplemental Reading

**For interviews:**
- **"What is HNSW?"** — "Hierarchical Navigable Small World is a graph-based index for approximate nearest neighbor search. It builds a multi-layer graph where each layer has fewer nodes. Search starts at the top (sparse) layer for coarse navigation and drills down to the bottom (dense) layer for precision. It achieves sub-linear search time: O(log N) instead of O(N)."
- **"Why Cosine over Euclidean for embeddings?"** — "Embedding models are trained with contrastive losses that optimize the angle between vectors, not their magnitude. Two vectors can have different magnitudes but point in the same direction, meaning they represent the same concept. Cosine similarity ignores magnitude and measures only direction."

**For production context:**
- Qdrant supports **filtering during search**: e.g., "find the nearest product, but only among drinks." This is done via payload filters and doesn't degrade search speed significantly because Qdrant applies filters during the graph traversal, not after.
- For very large collections (millions of vectors), you'd enable **quantization** (compressing 32-bit floats to 8-bit integers) to reduce memory usage by 4x with minimal accuracy loss.
