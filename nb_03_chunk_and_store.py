# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 03 — Chunk MD Files & Build Vector Search Index
# MAGIC
# MAGIC Pipeline:
# MAGIC 1. Read every `.md` file from the UC Volume
# MAGIC 2. Split content into overlapping character chunks
# MAGIC 3. Write chunks to the `jobs_kb_chunks` Delta table (CDF-enabled)
# MAGIC 4. Create a Vector Search **endpoint** (if absent)
# MAGIC 5. Create a **Delta Sync index** backed by BGE embeddings (if absent)
# MAGIC 6. Wait for the index to reach ONLINE status and verify with a smoke-test query

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch --quiet

# COMMAND ----------

# MAGIC %run ./config_loader

# COMMAND ----------

# ── IDE type stubs: Databricks runtime and config_loader inject these at runtime
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import logging
    from pyspark.sql import SparkSession
    spark: SparkSession
    dbutils: Any
    display: Any
    log: logging.Logger
    with_retry: Callable[..., Any]
    CATALOG: str
    SCHEMA: str
    JOBS_TABLE: str
    RUNS_TABLE: str
    VOLUME_NAME: str
    VOLUME_PATH: str
    MD_FILES_PATH: str
    VS_ENDPOINT_NAME: str
    VS_INDEX_NAME: str
    KB_DELTA_TABLE: str
    EMBEDDING_MODEL: str
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int
    CLAUDE_ENDPOINT: str
    AGENT_MODEL_NAME: str
    AGENT_ENDPOINT: str
    MAX_AGENT_ROUNDS: int
    MAX_TOKENS: int
    TEMPERATURE: float
    GRANT_PRINCIPALS: list[str]

# COMMAND ----------

import re
import time
import uuid
from datetime import datetime, timezone

from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField,
    StringType, IntegerType, TimestampType,
)
from databricks.vector_search.client import VectorSearchClient

log.info("nb_03 started")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 — Prerequisite check

# COMMAND ----------

if not spark.catalog.tableExists(KB_DELTA_TABLE):
    raise RuntimeError(
        f"KB table {KB_DELTA_TABLE!r} not found. Run nb_01 first."
    )

_md_files = [f for f in dbutils.fs.ls(MD_FILES_PATH) if f.name.endswith(".md")]
if not _md_files:
    raise RuntimeError(
        f"No .md files found in {MD_FILES_PATH!r}. Run nb_02 first."
    )

log.info(f"Found {len(_md_files)} MD files")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 — Read MD files from Volume

# COMMAND ----------

print(f"MD files in {MD_FILES_PATH} ({len(_md_files)} total):\n")
for _f in _md_files:
    print(f"  {_f.name:62s}  {_f.size:>8,} bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 — Chunking helpers

# COMMAND ----------

def _extract_job_id(filename: str) -> int:
    m = re.search(r"__(\d+)\.md$", filename)
    if not m:
        raise ValueError(f"Cannot extract job_id from filename: {filename!r}")
    return int(m.group(1))


def _extract_job_name(filename: str) -> str:
    return re.sub(r"__\d+\.md$", "", filename).replace("_", " ")


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks, preferring newline boundaries."""
    chunks: list[str] = []
    start  = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            snap = start + int(chunk_size * 0.80)
            nl   = text.rfind("\n", snap, end)
            if nl != -1:
                end = nl + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap

    return chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 — Build chunk rows

# COMMAND ----------

_chunks_schema = StructType([
    StructField("chunk_id",    StringType(),    False),
    StructField("job_id",      IntegerType(),   True),
    StructField("job_name",    StringType(),    True),
    StructField("chunk_index", IntegerType(),   True),
    StructField("content",     StringType(),    True),
    StructField("source_file", StringType(),    True),
    StructField("created_at",  TimestampType(), True),
])

_total_chunks = 0
_now = datetime.now(timezone.utc).replace(tzinfo=None)

for _fi_idx, _fi in enumerate(_md_files):
    _job_id   = _extract_job_id(_fi.name)
    _job_name = _extract_job_name(_fi.name)
    _content  = dbutils.fs.head(_fi.path, 1_000_000)
    _file_chunks = chunk_text(_content, CHUNK_SIZE, CHUNK_OVERLAP)

    _rows = [
        Row(
            chunk_id    = str(uuid.uuid4()),
            job_id      = _job_id,
            job_name    = _job_name,
            chunk_index = _idx,
            content     = _chunk,
            source_file = _fi.path,
            created_at  = _now,
        )
        for _idx, _chunk in enumerate(_file_chunks)
    ]

    _df   = spark.createDataFrame(_rows, schema=_chunks_schema)
    _mode = "overwrite" if _fi_idx == 0 else "append"
    _df.write.format("delta").mode(_mode).saveAsTable(KB_DELTA_TABLE)
    _total_chunks += len(_rows)
    log.info(f"  {_fi.name}: {len(_rows)} chunks")

log.info(
    f"Chunks generated: {_total_chunks}  "
    f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, "
    f"avg/file={_total_chunks/max(len(_md_files),1):.1f})"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 — Write chunks to KB Delta table

# COMMAND ----------

_saved = spark.table(KB_DELTA_TABLE).count()
assert _saved == _total_chunks, \
    f"Row count mismatch: expected {_total_chunks}, saved {_saved}"
log.info(f"Saved {_saved} chunks → {KB_DELTA_TABLE}")
display(spark.table(KB_DELTA_TABLE).limit(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 — Create Vector Search endpoint

# COMMAND ----------

_vsc = VectorSearchClient(disable_notice=True)


def _endpoint_is_online(name: str) -> bool:
    try:
        state = _vsc.get_endpoint(name).get("endpoint_status", {}).get("state", "")
        return state == "ONLINE"
    except Exception:
        return False


@with_retry(max_attempts=3, initial_delay=10.0)
def _create_vs_endpoint() -> None:
    _vsc.create_endpoint(name=VS_ENDPOINT_NAME, endpoint_type="STANDARD")


if _endpoint_is_online(VS_ENDPOINT_NAME):
    log.info(f"VS endpoint {VS_ENDPOINT_NAME!r} already ONLINE — skipping creation")
else:
    log.info(f"Creating VS endpoint {VS_ENDPOINT_NAME!r} …")
    _create_vs_endpoint()

    for _attempt in range(40):
        _state = _vsc.get_endpoint(VS_ENDPOINT_NAME).get("endpoint_status", {}).get("state", "")
        log.info(f"  [{_attempt+1:02d}] endpoint state: {_state}")
        if _state == "ONLINE":
            break
        time.sleep(20)
    else:
        raise TimeoutError(
            f"VS endpoint {VS_ENDPOINT_NAME!r} did not reach ONLINE within ~13 minutes."
        )

log.info(f"VS endpoint {VS_ENDPOINT_NAME!r} is ONLINE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 — Create Delta Sync index
# MAGIC
# MAGIC `TRIGGERED` pipeline — sync is explicit (cheaper on free edition).
# MAGIC Change to `CONTINUOUS` for near-real-time updates in production.

# COMMAND ----------

def _index_exists() -> bool:
    try:
        _vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME)
        return True
    except Exception:
        return False


@with_retry(max_attempts=3, initial_delay=15.0)
def _create_vs_index() -> None:
    _vsc.create_delta_sync_index(
        endpoint_name                 = VS_ENDPOINT_NAME,
        index_name                    = VS_INDEX_NAME,
        source_table_name             = KB_DELTA_TABLE,
        pipeline_type                 = "TRIGGERED",
        primary_key                   = "chunk_id",
        embedding_source_column       = "content",
        embedding_model_endpoint_name = EMBEDDING_MODEL,
    )


if _index_exists():
    log.info(f"Index {VS_INDEX_NAME!r} exists — triggering sync …")
    _vsc.get_index(endpoint_name=VS_ENDPOINT_NAME, index_name=VS_INDEX_NAME).sync()
else:
    log.info(f"Creating VS index {VS_INDEX_NAME!r} …")
    _create_vs_index()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 — Wait for index to reach ONLINE

# COMMAND ----------

_ONLINE_STATES = {"ONLINE", "ONLINE_NO_PENDING_UPDATE"}

log.info("Waiting for index to become ONLINE …")
for _attempt in range(40):
    _idx_info    = _vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME).describe()
    _index_state = _idx_info.get("status", {}).get("detailed_state", "UNKNOWN")
    _row_count   = _idx_info.get("status", {}).get("indexed_row_count", 0)
    log.info(f"  [{_attempt+1:02d}] state={_index_state:35s}  indexed_rows={_row_count}")

    if _index_state in _ONLINE_STATES:
        break
    time.sleep(15)
else:
    raise TimeoutError(
        f"VS index {VS_INDEX_NAME!r} did not reach ONLINE within ~10 minutes."
    )

log.info(f"Index ONLINE — {_row_count} rows indexed")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 — Smoke test

# COMMAND ----------

_test_query = "jobs that failed recently with spark exception errors"

_results = _vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME).similarity_search(
    query_text  = _test_query,
    columns     = ["chunk_id", "job_id", "job_name", "chunk_index", "content"],
    num_results = 3,
)

print(f"Query: '{_test_query}'\n")
for _i, _row in enumerate(
    _results.get("result", {}).get("data_array", []), 1
):
    _cid, _jid, _jname, _cidx, _content, _score = _row
    print(f"Result {_i} — {_jname} (id={_jid})  chunk={_cidx}  score={_score:.4f}")
    print(f"  {_content[:200].replace(chr(10), ' ')}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print(f"{'Resource':<28} Value")
print("-" * 72)
print(f"{'MD files read':<28} {len(_md_files)}")
print(f"{'Total chunks written':<28} {_saved}")
print(f"{'KB Delta table':<28} {KB_DELTA_TABLE}")
print(f"{'VS endpoint':<28} {VS_ENDPOINT_NAME}")
print(f"{'VS index':<28} {VS_INDEX_NAME}")
print(f"{'Embedding model':<28} {EMBEDDING_MODEL}")

log.info("nb_03 complete")
