import os, json, uuid
from pathlib import Path
import google.generativeai as genai
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams, PayloadSchemaType, Filter, FieldCondition, MatchValue, Range

# ---- config (use env; rotate your leaked key!) ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCVnreRO2aIosNcG6FwgvnIYYdeSvDO-YI")
QDRANT_URL = os.getenv("QDRANT_URL", "https://07cc33cb-f09d-4add-b07f-8440c6bbdb54.us-west-2-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.bbIl5bU8oQisaPH4D0TMBr4zz4mkuejR6Zp37izO-N4")
COLLECTION = "etudeai"

def configure_gemini():
    genai.configure(api_key=GEMINI_API_KEY)

def embed(text: str):
    res = genai.embed_content(model="text-embedding-004", content=text)
    emb = res.get("embedding")
    return emb["values"] if isinstance(emb, dict) and "values" in emb else emb

def load_json_items(json_path: str):
    items = json.loads(Path(json_path).read_text(encoding="utf-8"))
    out = []
    for it in items:
        text = (it.get("page_content") or "").strip()
        if not text:
            continue
        # merge top-level fields + metadata, keep 'text' separately
        payload = {k: v for k, v in it.items() if k != "page_content"}
        meta = payload.pop("metadata", {}) or {}
        payload = {"text": text, **meta, **payload}
        out.append((text, payload))
    return out

def recreate_collection_safe(client: QdrantClient, name: str, dim: int):
    try:
        if client.collection_exists(name):
            client.delete_collection(name)
    except Exception as e:
        print("collection_exists check failed (continuing):", repr(e))
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

def index_payload_fields(client: QdrantClient, name: str):
    # optional: make filtering fast
    try:
        client.create_payload_index(name, "page", PayloadSchemaType.INTEGER)
    except Exception as e:
        print("page index:", repr(e))
    try:
        client.create_payload_index(name, "المحور", PayloadSchemaType.KEYWORD)
    except Exception as e:
        print("المحور index:", repr(e))

def upsert_json(json_path: str):
    configure_gemini()
    items = load_json_items(json_path)
    if not items:
        raise RuntimeError("No items found in JSON.")

    # probe dimension
    probe_vec = embed(items[0][0])
    dim = len(probe_vec)

    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=15.0)
    recreate_collection_safe(qdrant, COLLECTION, dim)

    # upsert in small batches
    batch, B = [], 64
    for text, payload in items:
        vec = embed(text)
        batch.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))
        if len(batch) >= B:
            qdrant.upsert(collection_name=COLLECTION, points=batch)
            batch = []
    if batch:
        qdrant.upsert(collection_name=COLLECTION, points=batch)

    index_payload_fields(qdrant, COLLECTION)
    return qdrant

# ---- paths: adjust for your env ----
candidates = [
    os.getenv("JSON_PATH", ""),
    "/kaggle/input/booook/Book_with_axes.json",
    "/mnt/data/Book_with_axes.json",
]
json_path = next((p for p in candidates if p and os.path.exists(p)), None)
if not json_path:
    raise FileNotFoundError("Book_with_axes.json not found; set JSON_PATH or fix the path.")

qdrant = upsert_json(json_path)
print("Upsert complete.")

# ---- example search with optional filtering by المحور & page range ----
configure_gemini()
query = "الوقاية من أمراض العين"
qvec = embed(query)

flt = Filter(must=[
    FieldCondition(key="المحور", match=MatchValue(value="الإبصار")),  # adjust or remove
    FieldCondition(key="page", range=Range(gte=5, lte=50)),            # adjust or remove
])

hits = qdrant.search(
    collection_name=COLLECTION,
    query_vector=qvec,
    with_payload=True,
    limit=5,
    score_threshold=0.35,
    query_filter=flt,   # pass None to search all
)

for h in hits:
    p = h.payload
    print(f"score={h.score:.3f} page={p.get('page')} المحور={p.get('المحور')}\n{p.get('text','')[:220]}...\n")
