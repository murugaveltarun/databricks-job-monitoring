# Databricks notebook source
# MAGIC %md
# MAGIC # Config — Parameterize all settings here before running any notebook

# COMMAND ----------

# ─── Catalog / Schema ────────────────────────────────────────────────────────
CATALOG        = "main"
SCHEMA         = "jobs_agent"

# ─── Raw jobs tables (created by nb_00) ─────────────────────────────────────
JOBS_TABLE     = f"{CATALOG}.{SCHEMA}.jobs"
RUNS_TABLE     = f"{CATALOG}.{SCHEMA}.job_runs"

# ─── UC Volume (created by nb_01) ────────────────────────────────────────────
VOLUME_NAME    = "jobs_kb_volume"
VOLUME_PATH    = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
MD_FILES_PATH  = f"{VOLUME_PATH}/md_files"

# ─── Vector Search (created by nb_01 / nb_03) ────────────────────────────────
VS_ENDPOINT_NAME  = "jobs_agent_vs_endpoint"
VS_INDEX_NAME     = f"{CATALOG}.{SCHEMA}.jobs_kb_index"
KB_DELTA_TABLE    = f"{CATALOG}.{SCHEMA}.jobs_kb_chunks"
EMBEDDING_MODEL   = "databricks-bge-large-en"
CHUNK_SIZE        = 800       # characters
CHUNK_OVERLAP     = 100       # characters

# ─── Agent / Model Serving ────────────────────────────────────────────────────
CLAUDE_ENDPOINT   = "databricks-claude-sonnet-4-5"   # Databricks pay-per-token endpoint
AGENT_MODEL_NAME  = "jobs_nl_agent"
AGENT_ENDPOINT    = "jobs_nl_agent_endpoint"

# ─── Grants (comma-separated principals) ────────────────────────────────────
GRANT_PRINCIPALS  = ["account users"]   # e.g. ["data_engineers", "analysts"]
