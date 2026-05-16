# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Config Loader
# MAGIC Reads `config.yaml` from the same workspace directory and exposes all settings
# MAGIC as notebook-scope variables. Also defines the `with_retry` decorator and
# MAGIC standard logger used by every downstream notebook.
# MAGIC
# MAGIC **Do not run this notebook directly — use `%run ./config_loader` from other notebooks.**

# COMMAND ----------

# ── IDE type stubs: Databricks runtime injects these at execution time ────────
from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    spark: SparkSession
    dbutils: Any
    display: Any

import yaml
import logging
import functools
import time
import os

# COMMAND ----------

# Standard logger — imported notebooks share this instance
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt= "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("jobs_agent")

# COMMAND ----------

# Resolve config.yaml relative to this notebook's workspace path
_nb_path     = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
_nb_dir      = "/Workspace" + "/".join(_nb_path.split("/")[:-1])
_config_path = os.path.join(_nb_dir, "config.yaml")

log.info(f"Loading config: {_config_path}")

with open(_config_path, "r") as _f:
    _cfg = yaml.safe_load(_f)

log.info("config.yaml loaded OK")

# COMMAND ----------

# ── Runtime widgets — override any value without editing config.yaml ──────────
# Leave blank to use the value from config.yaml.

dbutils.widgets.text("catalog_override", "",  "Catalog  (blank → config.yaml)")
dbutils.widgets.text("schema_override",  "",  "Schema   (blank → config.yaml)")

_catalog_ov = dbutils.widgets.get("catalog_override").strip()
_schema_ov  = dbutils.widgets.get("schema_override").strip()

# COMMAND ----------

# ── Catalog / Schema ──────────────────────────────────────────────────────────
CATALOG = _catalog_ov or _cfg["catalog"]
SCHEMA  = _schema_ov  or _cfg["schema"]

# ── Tables ────────────────────────────────────────────────────────────────────
JOBS_TABLE = f"{CATALOG}.{SCHEMA}.{_cfg['tables']['jobs']}"
RUNS_TABLE = f"{CATALOG}.{SCHEMA}.{_cfg['tables']['runs']}"

# ── UC Volume ─────────────────────────────────────────────────────────────────
VOLUME_NAME   = _cfg["volume"]["name"]
VOLUME_PATH   = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"
MD_FILES_PATH = f"{VOLUME_PATH}/{_cfg['volume']['md_subdir']}"

# ── Vector Search ─────────────────────────────────────────────────────────────
VS_ENDPOINT_NAME = _cfg["vector_search"]["endpoint_name"]
VS_INDEX_NAME    = f"{CATALOG}.{SCHEMA}.{_cfg['vector_search']['index_name']}"
KB_DELTA_TABLE   = f"{CATALOG}.{SCHEMA}.{_cfg['vector_search']['kb_table']}"
EMBEDDING_MODEL  = _cfg["vector_search"]["embedding_model"]
CHUNK_SIZE       = int(_cfg["vector_search"]["chunk_size"])
CHUNK_OVERLAP    = int(_cfg["vector_search"]["chunk_overlap"])

# ── Agent / Serving ───────────────────────────────────────────────────────────
CLAUDE_ENDPOINT   = _cfg["agent"]["claude_endpoint"]
AGENT_MODEL_NAME  = _cfg["agent"]["model_name"]
AGENT_ENDPOINT    = _cfg["agent"]["serving_endpoint"]
MAX_AGENT_ROUNDS  = int(_cfg["agent"]["max_agent_rounds"])
MAX_TOKENS        = int(_cfg["agent"]["max_tokens"])
TEMPERATURE       = float(_cfg["agent"]["temperature"])

# ── Grants ────────────────────────────────────────────────────────────────────
GRANT_PRINCIPALS  = _cfg.get("grant_principals", [])

# COMMAND ----------

log.info(
    f"Config resolved — catalog={CATALOG!r}  schema={SCHEMA!r}  "
    f"jobs={JOBS_TABLE!r}  vs_endpoint={VS_ENDPOINT_NAME!r}"
)

# COMMAND ----------

# ── Shared retry decorator ────────────────────────────────────────────────────

def with_retry(
    max_attempts: int   = 3,
    initial_delay: float = 5.0,
    backoff: float       = 2.0,
    exceptions: tuple    = (Exception,),
):
    """Decorator: retry *fn* up to *max_attempts* times with exponential back-off."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay   = initial_delay
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    attempt += 1
                    if attempt >= max_attempts:
                        log.error(f"{fn.__name__} failed after {attempt} attempts: {exc}")
                        raise
                    log.warning(
                        f"{fn.__name__} attempt {attempt}/{max_attempts} failed: {exc}. "
                        f"Retrying in {delay:.0f}s…"
                    )
                    time.sleep(delay)
                    delay *= backoff
        return wrapper
    return decorator
