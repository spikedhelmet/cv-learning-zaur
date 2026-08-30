from sentence_transformers import SentenceTransformer, util
from PIL import Image

print("Loading CLIP model")
model = SentenceTransformer("clip-ViT-B-32")

ref_img = Image.open('nutivo/reference_db/efes.jpg')
query_match = Image.open('nutivo/reference_db/efes_side.jpg') 
query_fail = Image.open('nutivo/reference_db/lays.jpg')

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