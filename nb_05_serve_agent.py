# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 05 — Serve the Agent as a REST Endpoint
# MAGIC
# MAGIC Steps:
# MAGIC 1. Resolve the latest registered model version from Unity Catalog
# MAGIC 2. Create (or update) a Databricks Model Serving endpoint
# MAGIC 3. Poll until the endpoint is `READY`
# MAGIC 4. Smoke-test via REST
# MAGIC 5. Print integration examples (cURL, Python, Power Automate / Teams)
# MAGIC 6. Grant `CAN_QUERY` to configured principals

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade --quiet

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
import requests

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    AutoCaptureConfigInput,
)
from databricks.sdk.service import iam
from databricks.sdk.service.serving import PermissionLevel

log.info("nb_05 started")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 — Resolve latest registered model version

# COMMAND ----------

_w = WorkspaceClient()
_registered_name = f"{CATALOG}.{SCHEMA}.{AGENT_MODEL_NAME}"

_versions = list(_w.model_versions.list(full_name=_registered_name))
if not _versions:
    raise RuntimeError(
        f"No versions found for {_registered_name!r}. Run nb_04 first."
    )

_latest_version = str(max(int(v.version) for v in _versions))
log.info(f"Model: {_registered_name}  version: {_latest_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 — Create or update serving endpoint

# COMMAND ----------

_endpoint_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name           = _registered_name,
            entity_version        = _latest_version,
            workload_size         = "Small",   # 4 concurrent requests; free-tier safe
            scale_to_zero_enabled = True,      # scales to zero when idle — cost control
        )
    ],
    auto_capture_config=AutoCaptureConfigInput(
        catalog_name      = CATALOG,
        schema_name       = SCHEMA,
        table_name_prefix = f"{AGENT_MODEL_NAME}_inference",
        enabled           = True,             # logs every request/response to Delta
    ),
)

try:
    _w.serving_endpoints.get(name=AGENT_ENDPOINT)
    log.info(f"Endpoint {AGENT_ENDPOINT!r} exists — updating config …")
    _w.serving_endpoints.update_config(
        name           = AGENT_ENDPOINT,
        served_entities= _endpoint_config.served_entities,
    )
except Exception:
    log.info(f"Creating endpoint {AGENT_ENDPOINT!r} …")
    _w.serving_endpoints.create(
        name   = AGENT_ENDPOINT,
        config = _endpoint_config,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 — Wait for endpoint to reach READY

# COMMAND ----------

log.info(f"Waiting for {AGENT_ENDPOINT!r} to become READY …")

for _attempt in range(60):          # up to ~30 minutes
    _ep    = _w.serving_endpoints.get(name=AGENT_ENDPOINT)
    _state = _ep.state.ready.value if _ep.state and _ep.state.ready else "UNKNOWN"
    _cu    = _ep.state.config_update.value if _ep.state and _ep.state.config_update else ""
    log.info(f"  [{_attempt+1:02d}] ready={_state:20s}  config_update={_cu}")

    if _state == "READY":
        break
    time.sleep(30)
else:
    raise TimeoutError(
        f"Endpoint {AGENT_ENDPOINT!r} did not become READY within ~30 minutes."
    )

_workspace_url = spark.conf.get("spark.databricks.workspaceUrl").strip("/")
_endpoint_url  = f"https://{_workspace_url}/serving-endpoints/{AGENT_ENDPOINT}/invocations"

log.info(f"Endpoint READY: {_endpoint_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 — Smoke test

# COMMAND ----------

_token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().apiToken().get()
)

_test_question = "Which Databricks job had the most failures and what were the errors?"

_response = requests.post(
    _endpoint_url,
    headers = {
        "Authorization": f"Bearer {_token}",
        "Content-Type":  "application/json",
    },
    json    = {"messages": [{"role": "user", "content": _test_question}]},
    timeout = 120,
)
_response.raise_for_status()
_result = _response.json()

print(f"Question : {_test_question}")
print(f"\nAnswer:\n{_result.get('content', _result)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 — Integration reference

# COMMAND ----------

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ENDPOINT REFERENCE                                              ║
╠══════════════════════════════════════════════════════════════════╣
║  Name   : {AGENT_ENDPOINT:<53}║
║  URL    : {_endpoint_url[:53]:<53}║
║  Method : POST                                                   ║
║  Auth   : Bearer <databricks-pat>                                ║
╚══════════════════════════════════════════════════════════════════╝
""")

print("── cURL ─────────────────────────────────────────────────────────")
print(f"""curl -X POST "{_endpoint_url}" \\
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"messages":[{{"role":"user","content":"Which jobs failed last week?"}}]}}'
""")

print("── Python ───────────────────────────────────────────────────────")
print(f"""import requests

resp = requests.post(
    "{_endpoint_url}",
    headers={{"Authorization": "Bearer <PAT>", "Content-Type": "application/json"}},
    json={{"messages": [{{"role": "user", "content": "Your question here"}}]}},
    timeout=120,
)
print(resp.json()["content"])
""")

print("── Microsoft Teams (Power Automate) ─────────────────────────────")
print("""
  1. Create a flow: trigger = "When a message is posted in a channel"
  2. Add HTTP action
       Method : POST
       URL    : <endpoint URL above>
       Headers: Content-Type: application/json
                Authorization: Bearer <PAT from Key Vault>
       Body   : { "messages": [ { "role": "user", "content": @{triggerBody()?['text']} } ] }
  3. Reply to channel: @{body('HTTP')?['content']}

  Tip: store the Databricks PAT in Azure Key Vault and reference it via
       the Power Automate Key Vault connector — never hardcode tokens.
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 — Grant endpoint access to principals

# COMMAND ----------

_ep_id = _w.serving_endpoints.get(name=AGENT_ENDPOINT).id

for _principal in GRANT_PRINCIPALS:
    try:
        _w.serving_endpoints.set_permissions(
            serving_endpoint_id = _ep_id,
            access_control_list = [
                iam.AccessControlRequest(
                    group_name       = _principal,
                    permission_level = PermissionLevel.CAN_QUERY,
                )
            ],
        )
        log.info(f"Granted CAN_QUERY → {_principal!r}")
    except Exception as _exc:
        log.warning(f"Grant skipped for {_principal!r}: {_exc}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 — Inference log table

# COMMAND ----------

_log_table = f"{CATALOG}.{SCHEMA}.{AGENT_MODEL_NAME}_inference_payload"

if spark.catalog.tableExists(_log_table):
    display(spark.sql(f"""
        SELECT
            date_trunc('hour', timestamp) AS hour,
            COUNT(*)                      AS requests,
            AVG(execution_duration_ms)    AS avg_latency_ms
        FROM {_log_table}
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT 24
    """))
else:
    log.info(f"Inference log table {_log_table!r} not yet created — it appears after the first request.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print(f"""
All notebooks complete. Execution order:

  nb_00  →  Create fake jobs + job_runs tables
  nb_01  →  UC Volume + KB Delta table + Grants
  nb_02  →  Generate MD files per job
  nb_03  →  Chunk MD files + Vector Search index
  nb_04  →  Build & log NL-to-SQL agent (Claude Sonnet 4.5)
  nb_05  →  Serve agent as REST endpoint  ← you are here

Endpoint : {_endpoint_url}

To re-index when jobs change:
  Re-run nb_02  →  nb_03  (no redeploy needed — endpoint stays live)
""")

log.info("nb_05 complete")
