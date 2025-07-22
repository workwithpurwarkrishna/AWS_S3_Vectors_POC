import os, uuid, time, boto3, openai
from dotenv import load_dotenv

# Load config
load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")
VECTOR_DIM = 3072
EMBED_MODEL = "text-embedding-3-large"

# AWS clients
s3v = boto3.client("s3vectors",
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"))

def embed(texts):  # Generate OpenAI embeddings
    res = openai.embeddings.create(input=texts, model=EMBED_MODEL)
    return [e.embedding for e in res.data]

def insert(bucket, index, vectors, metadatas):
    vecs = [{
        "key": str(uuid.uuid4()),
        "data": {"float32": vec},
        "metadata": meta
    } for vec, meta in zip(vectors, metadatas)]
    return s3v.put_vectors(vectorBucketName=bucket, indexName=index, vectors=vecs)

def query(bucket, index, vector, top_k=3):
    res = s3v.query_vectors(
        vectorBucketName=bucket, indexName=index,
        queryVector={"float32": vector},
        topK=top_k, returnDistance=True, returnMetadata=True)
    for r in res.get("vectors", []):
        print(f"→ {r['metadata'].get('original_text')} (dist: {r['distance']:.4f})")

# --- Demo ---
bucket = os.getenv("S3_VECTOR_BUCKET_NAME")
index = os.getenv("S3_VECTOR_INDEX_NAME")

texts = ["The quick brown fox...", "Early bird catches the worm"]
vecs = embed(texts)
insert(bucket, index, vecs, [{"original_text": t} for t in texts])
time.sleep(10)  # wait for indexing
query_vec = embed(["Who wakes up early?"])[0]
query(bucket, index, query_vec)
