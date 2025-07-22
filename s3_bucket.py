import boto3
import openai
import os
import uuid
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
VECTOR_DIMENSIONS = 3072
EMBEDDING_MODEL = "text-embedding-3-large"

# Initialize clients
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
s3_vectors_client = boto3.client(
    's3vectors',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def create_vector_bucket(bucket_name):
    """Create a new S3 Vector bucket."""
    try:
        response = s3_vectors_client.create_vector_bucket(
            vectorBucketName=bucket_name
        )
        http_status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        if http_status == 200:
            print(f"✓ Vector bucket '{bucket_name}' created successfully")
        else:
            print(f"❌ Failed to create bucket (HTTP {http_status})")
        return response
    except Exception as e:
        if "BucketAlreadyExists" in str(e) or "already exists" in str(e).lower():
            print(f"✓ Vector bucket '{bucket_name}' already exists")
            return True
        else:
            print(f"❌ Error creating bucket: {e}")
            return None

def create_vector_index(bucket_name, index_name, dimensions=VECTOR_DIMENSIONS, metric="cosine"):
    """Create a new vector index within a bucket."""
    try:
        response = s3_vectors_client.create_index(
            vectorBucketName=bucket_name,
            indexName=index_name,
            dataType='float32',
            dimension=dimensions,
            distanceMetric=metric
        )
        http_status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        if http_status == 200:
            print(f"✓ Vector index '{index_name}' created successfully in bucket '{bucket_name}'")
        else:
            print(f"❌ Failed to create index (HTTP {http_status})")
        return response
    except Exception as e:
        if "IndexAlreadyExists" in str(e) or "already exists" in str(e).lower():
            print(f"✓ Vector index '{index_name}' already exists")
            return True
        else:
            print(f"❌ Error creating index: {e}")
            return None

def list_vector_buckets():
    """List all vector buckets."""
    try:
        response = s3_vectors_client.list_vector_buckets()
        buckets = response.get('vectorBuckets', [])
        print(f"Found {len(buckets)} vector buckets:")
        for bucket in buckets:
            print(f"  - {bucket['vectorBucketName']}")
        return buckets
    except Exception as e:
        print(f"❌ Error listing buckets: {e}")
        return []

def list_vector_indexes(bucket_name):
    """List all indexes in a bucket."""
    try:
        response = s3_vectors_client.list_indexes(vectorBucketName=bucket_name)
        indexes = response.get('indexes', [])
        print(f"Found {len(indexes)} indexes in bucket '{bucket_name}':")
        for index in indexes:
            print(f"  - {index['indexName']} (dimensions: {index.get('dimension', 'N/A')})")
        return indexes
    except Exception as e:
        print(f"❌ Error listing indexes: {e}")
        return []

def generate_embeddings(texts):
    """Generate embeddings using OpenAI's text-embedding-3-large model."""
    response = openai_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]

def insert_vectors(bucket_name, index_name, vectors, metadata_list):
    """Insert vectors into S3 Vector Index."""
    vectors_to_insert = []
    for i, vec in enumerate(vectors):
        vectors_to_insert.append({
            'key': str(uuid.uuid4()),
            'data': {'float32': vec},
            'metadata': metadata_list[i]
        })
    
    try:
        response = s3_vectors_client.put_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            vectors=vectors_to_insert
        )
        
        http_status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        print(f"Insert status: HTTP {http_status} - {'Success' if http_status == 200 else 'Failed'}")
        return response
    except Exception as e:
        print(f"❌ Error inserting vectors: {e}")
        return None

def query_vectors(bucket_name, index_name, query_vector, top_k=5):
    """Query S3 Vector Index for similar vectors."""
    try:
        response = s3_vectors_client.query_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            queryVector={'float32': query_vector},
            topK=top_k,
            returnDistance=True,
            returnMetadata=True
        )
        
        if 'vectors' in response and response['vectors']:
            results = response['vectors']
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                metadata = result.get('metadata', {})
                original_text = metadata.get('original_text', 'No text available')
                distance = result.get('distance', 'N/A')
                print(f"  {i+1}. {original_text} (distance: {distance:.4f})")
        else:
            print("No results found.")
        
        return response
    except Exception as e:
        print(f"❌ Error querying vectors: {e}")
        return None

def setup_infrastructure(bucket_name, index_name):
    """Set up the vector bucket and index infrastructure."""
    print("=== Setting up S3 Vector Infrastructure ===")
    
    # Create bucket
    bucket_result = create_vector_bucket(bucket_name)
    if not bucket_result:
        print("❌ Failed to create/verify bucket. Exiting.")
        return False
    
    # Create index
    index_result = create_vector_index(bucket_name, index_name)
    if not index_result:
        print("❌ Failed to create/verify index. Exiting.")
        return False
    
    print("✓ Infrastructure setup complete\n")
    return True

def test_index_with_simple_operation(bucket_name, index_name):
    """Test if index is ready by attempting a simple query."""
    print("Testing index readiness with a simple query...")
    try:
        # Try a simple query with a dummy vector
        test_vector = [0.1] * VECTOR_DIMENSIONS
        response = s3_vectors_client.query_vectors(
            vectorBucketName=bucket_name,
            indexName=index_name,
            queryVector={'float32': test_vector},
            topK=1,
            returnDistance=True,
            returnMetadata=True
        )
        print("✓ Index appears to be ready for queries")
        return True
    except Exception as e:
        if "not ready" in str(e).lower() or "building" in str(e).lower():
            print("⏳ Index is still building, will wait...")
            return False
        else:
            print(f"⚠️  Index test query failed: {e}")
            print("Proceeding anyway...")
            return True

def main():
    # Generate unique names or use predefined ones
    bucket_name = os.environ.get("S3_VECTOR_BUCKET_NAME", f"my-vector-bucket-{int(time.time())}")
    index_name = os.environ.get("S3_VECTOR_INDEX_NAME", "embeddings-index")
    
    print(f"Using bucket: {bucket_name}")
    print(f"Using index: {index_name}\n")
    
    # Setup infrastructure
    if not setup_infrastructure(bucket_name, index_name):
        return
    
    # Give the index some time to initialize
    print("Giving index time to initialize...")
    time.sleep(30)  # Wait 30 seconds for index to be ready
    
    # Test if index is ready
    max_retries = 3
    for attempt in range(max_retries):
        if test_index_with_simple_operation(bucket_name, index_name):
            break
        if attempt < max_retries - 1:
            print(f"Retrying in 30 seconds... (attempt {attempt + 1}/{max_retries})")
            time.sleep(30)
    
    # Optional: List existing resources
    print("\n=== Current Resources ===")
    list_vector_buckets()
    list_vector_indexes(bucket_name)
    print()
    
    # Sample data
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm."
    ]
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(sample_texts)
    print(f"Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions each")
    
    # Prepare metadata
    metadata = [{"original_text": text, "source": "demo"} for text in sample_texts]
    
    # Insert vectors
    print(f"\nInserting {len(embeddings)} vectors...")
    insert_result = insert_vectors(bucket_name, index_name, embeddings, metadata)
    
    if insert_result:
        # Wait for indexing
        print("Waiting for vectors to be indexed...")
        time.sleep(15)
        
        # Query example
        query_text = "What is the opposite of a late riser?"
        print(f"\nQuerying: '{query_text}'")
        query_embedding = generate_embeddings([query_text])[0]
        query_vectors(bucket_name, index_name, query_embedding)
    else:
        print("❌ Vector insertion failed, skipping query test")

if __name__ == "__main__":
    main()

