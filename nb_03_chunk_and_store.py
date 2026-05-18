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
dbutils.library.restartPython()

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

import time
from datetime import datetime, timezone

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, TimestampType
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

_now    = datetime.now(timezone.utc).replace(tzinfo=None)
_stride = CHUNK_SIZE - CHUNK_OVERLAP

# --- read files (JVM only, no Python workers involved) ---
_raw_df = (
    spark.read
    .option("wholetext", "true")
    .text(MD_FILES_PATH)
    .select(
        F.col("_metadata.file_path").alias("source_file"),
        F.col("value").alias("raw_content"),
    )
    .filter(F.col("source_file").like("%.md"))
)

# Filename column reused for job_id / job_name extraction
_fname = F.element_at(F.split(F.col("source_file"), "/"), -1)

# Pure Spark SQL — sequence→substring→posexplode, zero Python workers
_chunks_df = (
    _raw_df
    .withColumn(
        "starts",
        F.sequence(F.lit(1), F.length("raw_content"), F.lit(_stride)),
    )
    .select(
        "source_file", "raw_content",
        F.posexplode("starts").alias("chunk_index", "start_pos"),
    )
    .withColumn("content", F.trim(F.substring("raw_content", F.col("start_pos"), F.lit(CHUNK_SIZE))))
    .drop("raw_content", "start_pos")           # free large column early
    .filter(F.length(F.col("content")) > 0)
    .withColumn("job_id",   F.regexp_extract(_fname, r"__(\d+)\.md$", 1).cast(IntegerType()))
    .withColumn("job_name", F.regexp_replace(F.regexp_replace(_fname, r"__\d+\.md$", ""), "_", " "))
    .withColumn("chunk_id",   F.expr("uuid()"))
    .withColumn("created_at", F.lit(_now).cast(TimestampType()))
    .select("chunk_id", "job_id", "job_name", "chunk_index",
            "content", "source_file", "created_at")
)

_chunks_df.write.format("delta").mode("overwrite").saveAsTable(KB_DELTA_TABLE)

_total_chunks = spark.table(KB_DELTA_TABLE).count()
log.info(
    f"Chunks generated: {_total_chunks}  "
    f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, "
    f"avg/file={_total_chunks / max(len(_md_files), 1):.1f})"
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

_ONLINE_STATES       = {"ONLINE", "ONLINE_NO_PENDING_UPDATE"}
# States where the endpoint itself is still booting — poll slowly to avoid noise
_SLOW_STATES         = {"PROVISIONING_ENDPOINT", "ENDPOINT_PROVISIONING"}
_MAX_WAIT_SECONDS    = 2400   # 40 min — endpoint cold-start can take 15-30 min
_POLL_SLOW           = 60     # seconds between polls while endpoint is booting
_POLL_FAST           = 15     # seconds between polls once endpoint is ready

log.info("Waiting for index to become ONLINE …")
_elapsed = 0
_attempt = 0
while _elapsed < _MAX_WAIT_SECONDS:
    _idx_info    = _vsc.get_index(VS_ENDPOINT_NAME, VS_INDEX_NAME).describe()
    _index_state = _idx_info.get("status", {}).get("detailed_state", "UNKNOWN")
    _row_count   = _idx_info.get("status", {}).get("indexed_row_count", 0)
    _attempt    += 1
    log.info(f"  [{_attempt:02d}] state={_index_state:35s}  indexed_rows={_row_count}  elapsed={_elapsed}s")

    if _index_state in _ONLINE_STATES:
        break

    _sleep = _POLL_SLOW if _index_state in _SLOW_STATES else _POLL_FAST
    time.sleep(_sleep)
    _elapsed += _sleep
else:
    raise TimeoutError(
        f"VS index {VS_INDEX_NAME!r} did not reach ONLINE within {_MAX_WAIT_SECONDS // 60} minutes. "
        f"Last state: {_index_state!r}"
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
