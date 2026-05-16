# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 01 — UC Volume · KB Delta Table · Grants
# MAGIC
# MAGIC Creates all storage infrastructure needed by downstream notebooks:
# MAGIC 1. Unity Catalog **Volume** — holds the generated Markdown files
# MAGIC 2. **KB chunks Delta table** — stores text chunks for Vector Search (CDF enabled)
# MAGIC 3. **Grants** on catalog, schema, volume, and tables for configured principals

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

from pyspark.sql.utils import AnalysisException

log.info("nb_01 started")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 — Catalog & Schema

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")

log.info(f"Using: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 — UC Volume

# COMMAND ----------

spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME_NAME}
    COMMENT 'Stores Markdown knowledge-base files generated from Databricks job metadata'
""")

dbutils.fs.mkdirs(MD_FILES_PATH)

try:
    dbutils.fs.ls(VOLUME_PATH)
    log.info(f"Volume OK: /Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}")
except Exception as exc:
    raise RuntimeError(
        f"Volume not accessible after creation: {exc}. "
        "Verify Unity Catalog is enabled on this workspace."
    ) from exc

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 — KB Chunks Delta Table
# MAGIC
# MAGIC `delta.enableChangeDataFeed = true` is **required** for Vector Search Delta Sync.
# MAGIC Do not remove this property.

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {KB_DELTA_TABLE} (
        chunk_id        STRING    NOT NULL,
        job_id          INT,
        job_name        STRING,
        chunk_index     INT,
        content         STRING,
        source_file     STRING,
        created_at      TIMESTAMP
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.columnMapping.mode'   = 'name'
    )
    COMMENT 'Chunked knowledge-base content for Vector Search ingestion'
""")

# Verify CDF is actually enabled
_props = (
    spark.sql(f"DESCRIBE TABLE EXTENDED {KB_DELTA_TABLE}")
    .filter("col_name = 'Table Properties'")
    .collect()
)
_prop_str = _props[0]["data_type"] if _props else ""
if "enableChangeDataFeed=true" not in _prop_str:
    raise RuntimeError(
        f"CDF not enabled on {KB_DELTA_TABLE}. "
        "Drop and recreate the table, or run: "
        f"ALTER TABLE {KB_DELTA_TABLE} SET TBLPROPERTIES ('delta.enableChangeDataFeed'='true')"
    )

log.info(f"KB table ready (CDF=ON): {KB_DELTA_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 — Grants

# COMMAND ----------

def _safe_grant(sql: str) -> None:
    try:
        spark.sql(sql)
        log.info(f"GRANT OK: {sql}")
    except AnalysisException as exc:
        log.warning(f"GRANT skipped (insufficient privileges): {exc}")


for _principal in GRANT_PRINCIPALS:
    log.info(f"Granting to: {_principal!r}")
    _safe_grant(f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{_principal}`")
    _safe_grant(f"GRANT USE SCHEMA  ON SCHEMA  {CATALOG}.{SCHEMA} TO `{_principal}`")
    _safe_grant(f"GRANT READ VOLUME ON VOLUME  {CATALOG}.{SCHEMA}.{VOLUME_NAME} TO `{_principal}`")
    for _tbl in [JOBS_TABLE, RUNS_TABLE, KB_DELTA_TABLE]:
        _safe_grant(f"GRANT SELECT ON TABLE {_tbl} TO `{_principal}`")

log.info("Grants complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 — Summary

# COMMAND ----------

_summary = [
    ("Catalog",          CATALOG),
    ("Schema",           f"{CATALOG}.{SCHEMA}"),
    ("Volume",           f"{CATALOG}.{SCHEMA}.{VOLUME_NAME}"),
    ("MD files path",    MD_FILES_PATH),
    ("KB chunks table",  KB_DELTA_TABLE),
    ("CDF enabled",      "yes"),
    ("Principals",       ", ".join(GRANT_PRINCIPALS) or "(none)"),
]

print(f"\n{'Resource':<22} Value")
print("-" * 70)
for _label, _value in _summary:
    print(f"{_label:<22} {_value}")

log.info("nb_01 complete")
