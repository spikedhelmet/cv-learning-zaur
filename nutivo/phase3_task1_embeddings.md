# Phase 3.1 — Embeddings and Cosine Similarity

---

## Goal

Pivot from text recognition (OCR) to visual fingerprinting (Embeddings). By the end of this task, you will understand how to convert an image into an embedding vector using OpenAI's CLIP model, and how to compare two images mathematically to see if they are the same product.

---

## Concept: Visual Fingerprinting

When you look at a blurry, tiny crop of a Doritos bag, you don't read the word "Doritos". You see a red triangle logo on an orange background. You recognize the _visual pattern_.

An embedding model does exactly this. It passes an image through a neural network and outputs a **Vector** (a list of numbers, typically 512 or 768 numbers long).

- If you pass in two different photos of a Doritos bag, their vectors will be almost identical.
- If you pass in a Doritos bag and a bottle of Coca-Cola, their vectors will be completely different.

To measure how similar two vectors are, we use **Cosine Similarity**.

- `1.0` means they are pointing in the exact same direction (identical).
- `0.0` means they are entirely unrelated.

### The Accuracy Test Pipeline

1. **Reference Database**: Take ONE good crop of each unique product. Generate its embedding. Save it as the "Ground Truth".
2. **Query Image**: Take a messy, blurry crop from a shelf photo. Generate its embedding.
3. **Match**: Compare the Query embedding against every Reference embedding. The one with the highest Cosine Similarity is your match!

---

## Technical Mechanics: CLIP

We will use **CLIP (Contrastive Language-Image Pretraining)** by OpenAI. CLIP is a foundational model trained on hundreds of millions of images. It is incredibly good at extracting high-level visual features (colors, shapes, logos, packaging styles).

We will use the `sentence-transformers` library, which makes running CLIP locally extremely easy.

---

## Step-by-Step Task

### 1. Install Dependencies

You'll need `sentence-transformers` to run the CLIP model easily:

```bash
pip install sentence-transformers
```

### 2. Create a Reference and Query Set

From your `nutivo/cropped_images` folder, manually pick out a few files to act as your reference database.
Create a new folder `nutivo/reference_db/` and copy 3-5 distinct products into it. Rename them so you know what they are (e.g., `efes_draft.jpg`, `lays_paprika.jpg`, `sarikiz.jpg`).

Leave the rest of the crops in the `cropped_images` folder — these will be your "Query" images to test against.

### 3. Generate an Embedding

Create `nutivo/test_embeddings.py`. Let's start by generating an embedding for a single image:

```python
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load the CLIP model
print("Loading CLIP model...")
model = SentenceTransformer('clip-ViT-B-32')

# Load an image using PIL (CLIP expects PIL images, not OpenCV numpy arrays)
img = Image.open('nutivo/reference_db/efes_draft.jpg')

# Generate the embedding
embedding = model.encode(img)

print(f"Embedding shape: {embedding.shape}")
print(f"First 5 numbers of the vector: {embedding[:5]}")
```

_Run this. You should see it outputs a vector of length 512._

### 4. Measure Cosine Similarity

Now, let's load TWO images, generate their embeddings, and compare them.

```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image

model = SentenceTransformer('clip-ViT-B-32')

# Load one reference image, and two query images
ref_img = Image.open('nutivo/reference_db/efes_draft.jpg')
query_match = Image.open('path/to/another/crop/of/efes.jpg')
query_fail = Image.open('path/to/a/crop/of/lays.jpg')

# Generate embeddings
ref_emb = model.encode(ref_img)
match_emb = model.encode(query_match)
fail_emb = model.encode(query_fail)

# Calculate Cosine Similarity
# util.cos_sim returns a matrix, we just want the single value item()
score_match = util.cos_sim(ref_emb, match_emb).item()
score_fail = util.cos_sim(ref_emb, fail_emb).item()

print(f"Similarity (Efes vs Efes): {score_match:.2f}")
print(f"Similarity (Efes vs Lays): {score_fail:.2f}")
```

Run this. You should see the Efes vs Efes score is very high (e.g., >0.85), while the Efes vs Lays score is much lower (e.g., <0.60).

---

## Checkpoint Questions

1. Why does CLIP require `PIL.Image` instead of standard OpenCV (`cv2.imread`) arrays?

- cv2 uses a numpy array?

2. If the cosine similarity between two product crops is `0.92`, what does that physically mean about those two images?

- They're the image of the same thing most likely

3. If a new product is added to the supermarket, do you need to retrain the CLIP model? Why or why not?

- The clip itself? No? it's not clip's job to define my objects?

---

## Challenge (No Guidance)

**Build the Matcher Loop**

Write a script that:

1. Loads all images from your `nutivo/reference_db/` folder and saves their embeddings into a Python dictionary: `{"efes": [vector...], "lays": [vector...]}`.
2. Loops through 10 random crops from your `nutivo/cropped_images/` folder.
3. For each crop, compares its embedding against _every_ vector in the reference dictionary.
4. Prints out the filename and the name of the reference product it matched with (the one with the highest cosine similarity).

Does it correctly identify the blurry crops that OCR failed on?

---

## Supplemental Reading

**For interviews:**

- **"How does CLIP work?"** — "CLIP uses a Contrastive learning approach. It was trained on (image, text) pairs to predict which image goes with which text. As a result, its image encoder learned incredibly robust, high-level visual features that generalize to almost any domain without fine-tuning."
- **"Why Cosine Similarity over Euclidean Distance?"** — "In high-dimensional spaces (like a 512-D vector), the _magnitude_ of the vector can vary, but the _angle_ between vectors is a much more reliable measure of semantic similarity."
