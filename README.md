# AWS S3 Vector Search with OpenAI Embeddings

A complete, end-to-end Python implementation for creating, managing, and querying vector indices in AWS S3 Vector Search using OpenAI embeddings. This project demonstrates how to set up the vector storage infrastructure, generate embeddings for text data, insert them, and perform semantic similarity searches.

---

## ✨ Features

- **Infrastructure Management**: Programmatically create and manage S3 vector buckets and indices.
- **Resource Listing**: View existing vector buckets and the indices within them.
- **OpenAI Integration**: Generate high-quality, large-dimension embeddings using OpenAI's `text-embedding-3-large` model.
- **Vector Operations**: Insert vectors with associated metadata and perform efficient `top-k` similarity searches.
- **Robust Error Handling**: Gracefully handles resources that already exist.
- **Index Readiness Polling**: Intelligently waits and retries queries until a new index is ready, overcoming the lack of a formal status API.

---
### Prerequisites

* An AWS account with **S3 Vector Search enabled** (currently in preview).
* AWS credentials configured with the necessary permissions.
* An OpenAI API key.

---

## 🚀 Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/workwithpurwarkrishna/AWS_S3_Vectors_POC.git
   cd AWS_S3_Vectors_POC
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**

   Create a `.env` file in the root directory and populate it with your credentials. You can use the `.env.example` file as a template.

   **.env**

   ```env
   # OpenAI Configuration
   OPENAI_API_KEY="your_openai_api_key_here"

   # AWS Configuration
   AWS_ACCESS_KEY_ID="your_aws_access_key_id"
   AWS_SECRET_ACCESS_KEY="your_aws_secret_access_key"
   AWS_REGION="us-east-1" # Or any other region where S3 Vector Search is available

   # Optional: Custom Resource Names
   S3_VECTOR_BUCKET_NAME="my-vector-bucket"
   S3_VECTOR_INDEX_NAME="my-index"
   ```

---

## 🔐 AWS Permissions Required

Your AWS IAM user or role needs the following permissions to run this script:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3vectors:CreateVectorBucket",
                "s3vectors:ListVectorBuckets",
                "s3vectors:CreateIndex",
                "s3vectors:ListIndexes",
                "s3vectors:PutVectors",
                "s3vectors:QueryVectors"
            ],
            "Resource": "*"
        }
    ]
}
```

---

## ⚙️ Usage

To run the complete demonstration, simply execute the main script:

```bash
python s3_bucket.py
```

The script will perform the following actions:

1. Create a vector bucket and an index inside it.
2. Wait and poll the index to ensure it is ready for queries.
3. Generate embeddings for a list of sample texts.
4. Insert the vectors and their metadata into the index.
5. Wait briefly for the vectors to be indexed.
6. Perform a semantic similarity search with a sample query and print the results.

### Example Output

```text
Using bucket: my-vector-bucket-1679543210
Using index: embeddings-index

=== Setting up S3 Vector Infrastructure ===
✓ Vector bucket 'my-vector-bucket-1679543210' created successfully
✓ Vector index 'embeddings-index' created successfully in bucket 'my-vector-bucket-1679543210'
✓ Infrastructure setup complete

Giving index time to initialize...
Testing index readiness with a simple query...
⏳ Index is still building, will wait...
Retrying in 30 seconds... (attempt 1/3)
Testing index readiness with a simple query...
✓ Index appears to be ready for queries

=== Current Resources ===
Found 1 vector buckets:
  - my-vector-bucket-1679543210
Found 1 indexes in bucket 'my-vector-bucket-1679543210':
  - embeddings-index (dimensions: 3072)

Generating embeddings...
Generated 5 embeddings with 3072 dimensions each

Inserting 5 vectors...
Insert status: HTTP 200 - Success
Waiting for vectors to be indexed...

Querying: 'What is the opposite of a late riser?'
Found 5 results:
  1. The early bird catches the worm. (distance: 0.5494)
  2. The quick brown fox jumps over the lazy dog. (distance: 0.8112)
  3. A journey of a thousand miles begins with a single step. (distance: 0.7234)
  4. All that glitters is not gold. (distance: 0.7985)
  5. To be or not to be, that is the question. (distance: 0.8301)
```

---

## 🔧 Configuration Options

You can easily customize the script by changing the constants at the top of `s3_bucket.py`.

#### OpenAI Embedding Models

The script defaults to `text-embedding-3-large`. To use a different model, change both `EMBEDDING_MODEL` and `VECTOR_DIMENSIONS`.

| Model                    | Dimensions (`VECTOR_DIMENSIONS`) |
| ------------------------ | -------------------------------- |
| `text-embedding-3-large` | 3072                             |
| `text-embedding-3-small` | 1536                             |
| `text-embedding-ada-002` | 1536                             |

#### Vector Index Parameters

The `create_vector_index` function can be modified to change the distance metric.

| Parameter        | Options                   | Default     | Description                            |
| ---------------- | ------------------------- | ----------- | -------------------------------------- |
| `dataType`       | `'float32'`               | `'float32'` | The data type of the vector components |
| `dimension`      | `1-4096`                  | `3072`      | Must match your embedding model        |
| `distanceMetric` | `'cosine'`, `'euclidean'` | `'cosine'`  | The metric for calculating similarity  |

---

## ⚠️ Troubleshooting

* **Authentication Errors**: `NoCredentialsError: Unable to locate credentials`.

  * Ensure your `.env` file is correctly named, located in the root directory, and contains valid AWS/OpenAI keys.

* **Index Not Ready**: If queries fail with messages like `Index not ready for queries`, it means the index is still being provisioned. The script's polling mechanism handles this automatically, but provisioning can sometimes take several minutes.

* **Embedding Dimension Mismatch**: If you receive an error about vector dimensions, ensure the `VECTOR_DIMENSIONS` constant in the script perfectly matches the output dimension of the `EMBEDDING_MODEL` you've chosen.

* **Boto3 Attribute Errors**: An error like `'S3Vectors' object has no attribute 'create_index'` usually means your `boto3` or `botocore` libraries are outdated. Run `pip install --upgrade boto3 botocore` to get the latest version that supports S3 Vector Search.
