# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 03 — Chunk MD Files & Build Vector Search Index
# MAGIC
# MAGIC Pipeline:
# MAGIC 1. Read every `.md` file from the UC Volume
# MAGIC 2. Split content into overlapping character chunks
# MAGIC 3. Write chunks to `jobs_kb_chunks` Delta table (CDF-enabled)
# MAGIC 4. Create a Vector Search **endpoint** (if absent)
# MAGIC 5. Create a **Delta Sync index** backed by BGE embeddings (if absent)
# MAGIC 6. Wait for the index to reach ONLINE status and verify with a test query

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import uuid
import time
import re
from datetime import datetime, timezone

from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, TimestampType
)
from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

# MAGIC %md ## 1 — Read MD files from Volume

md_files = [f for f in dbutils.fs.ls(MD_FILES_PATH) if f.name.endswith(".md")]
print(f"Found {len(md_files)} MD files in {MD_FILES_PATH}\n")
for f in md_files:
    print(f"  {f.name:60s}  {f.size:>8,} bytes")

# COMMAND ----------

# MAGIC %md ## 2 — Chunking helpers

def extract_job_id_from_filename(name: str) -> int:
    """Pull the job_id from  {job_name}__{job_id}.md"""
    m = re.search(r"__(\d+)\.md$", name)
    return int(m.group(1)) if m else -1

def extract_job_name_from_filename(name: str) -> str:
    """Pull the job_name slug from  {job_name}__{job_id}.md"""
    return re.sub(r"__\d+\.md$", "", name).replace("_", " ")

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split *text* into overlapping character-level chunks.
    Tries to break on newlines to avoid cutting mid-sentence.
    """
    chunks = []
    start  = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        # Prefer to break on a newline within the last 20 % of the window
        if end < length:
            snap_start = start + int(chunk_size * 0.80)
            newline_pos = text.rfind("\n", snap_start, end)
            if newline_pos != -1:
                end = newline_pos + 1      # include the newline in this chunk

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap              # step back by overlap

    return chunks

# COMMAND ----------

# MAGIC %md ## 3 — Build chunk rows

chunk_rows = []

for file_info in md_files:
    filename = file_info.name
    job_id   = extract_job_id_from_filename(filename)
    job_name = extract_job_name_from_filename(filename)

    raw_content = dbutils.fs.head(file_info.path, 1_000_000)  # read up to 1 MB
    chunks      = chunk_text(raw_content, CHUNK_SIZE, CHUNK_OVERLAP)

    for idx, chunk in enumerate(chunks):
        chunk_rows.append(Row(
            chunk_id    = str(uuid.uuid4()),
            job_id      = job_id,
            job_name    = job_name,
            chunk_index = idx,
            content     = chunk,
            source_file = file_info.path,
            created_at  = datetime.now(timezone.utc).replace(tzinfo=None),
        ))

print(f"Total chunks generated: {len(chunk_rows)}")
print(f"  Chunk size   : {CHUNK_SIZE} chars")
print(f"  Chunk overlap: {CHUNK_OVERLAP} chars")
print(f"  Avg per file : {len(chunk_rows) / max(len(md_files), 1):.1f}")

# COMMAND ----------

# MAGIC %md ## 4 — Write chunks to KB Delta table

chunks_schema = StructType([
    StructField("chunk_id",    StringType(),    False),
    StructField("job_id",      IntegerType(),   True),
    StructField("job_name",    StringType(),    True),
    StructField("chunk_index", IntegerType(),   True),
    StructField("content",     StringType(),    True),
    StructField("source_file", StringType(),    True),
    StructField("created_at",  TimestampType(), True),
])

chunks_df = spark.createDataFrame(chunk_rows, schema=chunks_schema)

# Overwrite so re-runs stay idempotent; CDF is preserved on overwrite
chunks_df.write.format("delta").mode("overwrite").saveAsTable(KB_DELTA_TABLE)

saved = spark.table(KB_DELTA_TABLE).count()
print(f"Saved {saved} chunks → {KB_DELTA_TABLE}")
display(spark.table(KB_DELTA_TABLE).limit(3))

# COMMAND ----------

# MAGIC %md ## 5 — Create Vector Search endpoint

vsc = VectorSearchClient(disable_notice=True)

def endpoint_exists(name: str) -> bool:
    try:
        ep = vsc.get_endpoint(name)
        return ep.get("endpoint_status", {}).get("state") not in (None, "DELETED")
    except Exception:
        return False

if endpoint_exists(VS_ENDPOINT_NAME):
    print(f"Endpoint '{VS_ENDPOINT_NAME}' already exists — skipping creation.")
else:
    print(f"Creating VS endpoint '{VS_ENDPOINT_NAME}' …")
    vsc.create_endpoint(
        name          = VS_ENDPOINT_NAME,
        endpoint_type = "STANDARD",
    )
    # Wait until provisioned
    for _ in range(30):
        state = vsc.get_endpoint(VS_ENDPOINT_NAME).get("endpoint_status", {}).get("state", "")
        print(f"  endpoint state: {state}")
        if state == "ONLINE":
            break
        time.sleep(20)
    else:
        raise TimeoutError("VS endpoint did not reach ONLINE within 10 minutes.")

print(f"Endpoint '{VS_ENDPOINT_NAME}' is ONLINE.")

# COMMAND ----------

# MAGIC %md ## 6 — Create Delta Sync index
# MAGIC
# MAGIC `SOURCE_DELTA` + `MANAGED` embeddings — Databricks computes and stores
# MAGIC embeddings automatically using the BGE model; no manual embedding step needed.

def index_exists(endpoint: str, index_name: str) -> bool:
    try:
        vsc.get_index(endpoint_name=endpoint, index_name=index_name)
        return True
    except Exception:
        return False

if index_exists(VS_ENDPOINT_NAME, VS_INDEX_NAME):
    print(f"Index '{VS_INDEX_NAME}' already exists — triggering sync instead.")
    vsc.get_index(
        endpoint_name = VS_ENDPOINT_NAME,
        index_name    = VS_INDEX_NAME,
    ).sync()
else:
    print(f"Creating VS index '{VS_INDEX_NAME}' …")
    vsc.create_delta_sync_index(
        endpoint_name          = VS_ENDPOINT_NAME,
        index_name             = VS_INDEX_NAME,
        source_table_name      = KB_DELTA_TABLE,
        pipeline_type          = "TRIGGERED",       # manual sync; use CONTINUOUS for streaming
        primary_key            = "chunk_id",
        embedding_source_column= "content",
        embedding_model_endpoint_name = EMBEDDING_MODEL,
    )

# COMMAND ----------

# MAGIC %md ## 7 — Wait for index to reach ONLINE

print("Waiting for index to become ONLINE …")
for attempt in range(40):
    idx_info     = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME).describe()
    index_state  = idx_info.get("status", {}).get("detailed_state", "UNKNOWN")
    row_count    = idx_info.get("status", {}).get("indexed_row_count", 0)
    print(f"  [{attempt+1:02d}] state={index_state:30s}  indexed_rows={row_count}")
    if index_state in ("ONLINE", "ONLINE_NO_PENDING_UPDATE"):
        break
    time.sleep(15)
else:
    raise TimeoutError("VS index did not reach ONLINE within ~10 minutes.")

print(f"\nIndex is ready — {row_count} rows indexed.")

# COMMAND ----------

# MAGIC %md ## 8 — Smoke test: similarity search

test_query = "jobs that failed recently with spark exception errors"

results = vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME).similarity_search(
    query_text   = test_query,
    columns      = ["chunk_id", "job_id", "job_name", "chunk_index", "content"],
    num_results  = 3,
)

print(f"Query: '{test_query}'\n")
for i, row in enumerate(results.get("result", {}).get("data_array", []), 1):
    chunk_id, job_id, job_name, chunk_idx, content, score = row
    print(f"Result {i} — job: {job_name} (id={job_id})  chunk={chunk_idx}  score={score:.4f}")
    print(f"  {content[:200].replace(chr(10), ' ')}")
    print()

# COMMAND ----------

# MAGIC %md ## Summary

print(f"{'Resource':<28} {'Value'}")
print("-" * 72)
print(f"{'MD files read':<28} {len(md_files)}")
print(f"{'Total chunks written':<28} {saved}")
print(f"{'KB Delta table':<28} {KB_DELTA_TABLE}")
print(f"{'VS endpoint':<28} {VS_ENDPOINT_NAME}")
print(f"{'VS index':<28} {VS_INDEX_NAME}")
print(f"{'Embedding model':<28} {EMBEDDING_MODEL}")
