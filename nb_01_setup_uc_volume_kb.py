# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 01 — UC Volume · KB Delta Table · Grants
# MAGIC
# MAGIC Creates all storage infrastructure needed by the downstream notebooks:
# MAGIC 1. Unity Catalog **Volume** — holds the generated Markdown files
# MAGIC 2. **KB chunks Delta table** — stores text chunks fed into Vector Search (CDF enabled)
# MAGIC 3. **Grants** on catalog, schema, volume, and table for configured principals

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

# MAGIC %md ## 1 — Ensure catalog & schema exist

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"Using: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md ## 2 — Create UC Volume

spark.sql(f"""
    CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME_NAME}
    COMMENT 'Stores Markdown knowledge-base files generated from Databricks job metadata'
""")

# Create the md_files sub-directory inside the volume
dbutils.fs.mkdirs(MD_FILES_PATH)

print(f"Volume   : {CATALOG}.{SCHEMA}.{VOLUME_NAME}")
print(f"MD path  : {MD_FILES_PATH}")

# COMMAND ----------

# Verify volume is reachable
try:
    dbutils.fs.ls(VOLUME_PATH)
    print("Volume mount OK")
except Exception as e:
    raise RuntimeError(f"Volume not accessible: {e}")

# COMMAND ----------

# MAGIC %md ## 3 — Create KB chunks Delta table
# MAGIC
# MAGIC **Change Data Feed** (`delta.enableChangeDataFeed`) is mandatory so that
# MAGIC Databricks Vector Search can sync updates automatically.

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {KB_DELTA_TABLE} (
        chunk_id        STRING  NOT NULL,   -- uuid — primary key for VS index
        job_id          INT,
        job_name        STRING,
        chunk_index     INT,                -- position of this chunk within the MD file
        content         STRING,             -- text chunk fed into the embedding model
        source_file     STRING,             -- DBFS/Volume path of the source MD file
        created_at      TIMESTAMP
    )
    USING DELTA
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true',
        'delta.columnMapping.mode'   = 'name'
    )
    COMMENT 'Chunked knowledge-base content for Vector Search ingestion'
""")

print(f"KB table : {KB_DELTA_TABLE}")
spark.sql(f"DESCRIBE TABLE EXTENDED {KB_DELTA_TABLE}").filter("col_name = 'Table Properties'").display()

# COMMAND ----------

# MAGIC %md ## 4 — Grants

from pyspark.sql.utils import AnalysisException

def safe_grant(sql: str):
    try:
        spark.sql(sql)
        print(f"  OK  {sql}")
    except AnalysisException as e:
        # On free/community editions Unity Catalog GRANT may be restricted
        print(f"  SKIP (insufficient privileges): {e}")

for principal in GRANT_PRINCIPALS:
    print(f"\nGranting to: {principal}")

    safe_grant(f"GRANT USE CATALOG ON CATALOG {CATALOG} TO `{principal}`")
    safe_grant(f"GRANT USE SCHEMA  ON SCHEMA  {CATALOG}.{SCHEMA} TO `{principal}`")

    # Volume — read access so notebooks can read MD files
    safe_grant(f"GRANT READ VOLUME ON VOLUME {CATALOG}.{SCHEMA}.{VOLUME_NAME} TO `{principal}`")

    # Tables — SELECT on source tables and KB table
    for tbl in [JOBS_TABLE, RUNS_TABLE, KB_DELTA_TABLE]:
        safe_grant(f"GRANT SELECT ON TABLE {tbl} TO `{principal}`")

    # Model serving endpoint (if it already exists)
    # Endpoint grants are applied after nb_05 creates the endpoint

print("\nGrant step complete.")

# COMMAND ----------

# MAGIC %md ## 5 — Summary

rows = [
    ("Catalog",           CATALOG),
    ("Schema",            f"{CATALOG}.{SCHEMA}"),
    ("Volume",            f"{CATALOG}.{SCHEMA}.{VOLUME_NAME}"),
    ("MD files path",     MD_FILES_PATH),
    ("KB chunks table",   KB_DELTA_TABLE),
    ("Principals",        ", ".join(GRANT_PRINCIPALS)),
]

print("\n{:<22} {}".format("Resource", "Value"))
print("-" * 70)
for label, value in rows:
    print(f"{label:<22} {value}")
