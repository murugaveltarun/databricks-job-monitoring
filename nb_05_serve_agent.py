# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 05 — Serve the Agent as a REST Endpoint
# MAGIC
# MAGIC Steps:
# MAGIC 1. Resolve the latest registered model version from Unity Catalog
# MAGIC 2. Create (or update) a Databricks Model Serving endpoint
# MAGIC 3. Poll until the endpoint is `READY`
# MAGIC 4. Smoke-test via the Databricks SDK query API
# MAGIC 5. Show the raw REST call for external callers (Teams bot, Power Automate)
# MAGIC 6. Grant endpoint permissions to configured principals

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import time
import json
import requests

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    AutoCaptureConfigInput,
)
from databricks.sdk.service import iam

# COMMAND ----------

# MAGIC %md ## 1 — Resolve latest registered model version

w = WorkspaceClient()

registered_name = f"{CATALOG}.{SCHEMA}.{AGENT_MODEL_NAME}"

versions = w.registered_models.get(full_name=registered_name)
latest_version = max(
    [int(v.version) for v in w.model_versions.list(full_name=registered_name)],
    default=None,
)

if latest_version is None:
    raise RuntimeError(
        f"No versions found for '{registered_name}'. Run nb_04 first."
    )

print(f"Model    : {registered_name}")
print(f"Version  : {latest_version}")

# COMMAND ----------

# MAGIC %md ## 2 — Create or update serving endpoint

endpoint_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name          = registered_name,
            entity_version       = str(latest_version),
            workload_size        = "Small",       # Small = 4 concurrent requests; free-tier safe
            scale_to_zero_enabled= True,          # cost control — scales down when idle
        )
    ],
    auto_capture_config=AutoCaptureConfigInput(
        catalog_name = CATALOG,
        schema_name  = SCHEMA,
        table_name_prefix = f"{AGENT_MODEL_NAME}_inference",
        enabled      = True,                      # logs every request/response to a Delta table
    ),
)

try:
    existing = w.serving_endpoints.get(name=AGENT_ENDPOINT)
    print(f"Endpoint '{AGENT_ENDPOINT}' exists — updating config …")
    w.serving_endpoints.update_config(name=AGENT_ENDPOINT, served_entities=endpoint_config.served_entities)
except Exception:
    print(f"Creating endpoint '{AGENT_ENDPOINT}' …")
    w.serving_endpoints.create(
        name   = AGENT_ENDPOINT,
        config = endpoint_config,
    )

# COMMAND ----------

# MAGIC %md ## 3 — Wait for endpoint to reach READY state

print(f"Waiting for '{AGENT_ENDPOINT}' to become READY …")

for attempt in range(60):                         # up to ~30 minutes
    ep     = w.serving_endpoints.get(name=AGENT_ENDPOINT)
    state  = ep.state.ready.value  if ep.state and ep.state.ready  else "UNKNOWN"
    config = ep.state.config_update.value if ep.state and ep.state.config_update else ""
    print(f"  [{attempt+1:02d}] ready={state:15s}  config_update={config}")

    if state == "READY":
        break
    if state == "NOT_READY" and attempt > 3:
        # Still provisioning — keep waiting
        pass

    time.sleep(30)
else:
    raise TimeoutError(f"Endpoint did not become READY within the timeout window.")

workspace_url = spark.conf.get("spark.databricks.workspaceUrl").strip("/")
endpoint_url  = f"https://{workspace_url}/serving-endpoints/{AGENT_ENDPOINT}/invocations"

print(f"\nEndpoint READY")
print(f"URL: {endpoint_url}")

# COMMAND ----------

# MAGIC %md ## 4 — Smoke test via SDK

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

test_question = "Which Databricks job had the most failures and what were the errors?"

payload = {
    "messages": [
        {"role": "user", "content": test_question}
    ]
}

response = requests.post(
    endpoint_url,
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    },
    json = payload,
    timeout = 120,
)

response.raise_for_status()
result = response.json()

print(f"Question : {test_question}")
print(f"\nAnswer:\n{result.get('content', result)}")

# COMMAND ----------

# MAGIC %md ## 5 — Endpoint details & external call examples

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║              ENDPOINT REFERENCE                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  Name    : {AGENT_ENDPOINT:<51} ║
║  URL     : {endpoint_url[:51]:<51} ║
║  Method  : POST                                                  ║
║  Auth    : Bearer <databricks-pat>                               ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ── cURL example ───────────────────────────────────────────────────────
print("── cURL ──────────────────────────────────────────────────────────")
print(f"""curl -X POST "{endpoint_url}" \\
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{{"messages": [{{"role": "user", "content": "Which jobs failed last week?"}}]}}'
""")

# ── Python requests example ────────────────────────────────────────────
print("── Python ────────────────────────────────────────────────────────")
print(f"""import requests

response = requests.post(
    "{endpoint_url}",
    headers={{"Authorization": "Bearer <PAT>", "Content-Type": "application/json"}},
    json={{"messages": [{{"role": "user", "content": "Your question here"}}]}},
    timeout=120,
)
print(response.json()["content"])
""")

# ── Power Automate / Teams note ────────────────────────────────────────
print("── Microsoft Teams Integration ───────────────────────────────────")
print("""To connect to Teams:

  Option A — Power Automate (no-code)
    1. Create a flow triggered by "When a message is posted in a channel"
    2. Add HTTP action → POST to the endpoint URL above
       Body: { "messages": [ { "role": "user", "content": @{triggerBody()?['text']} } ] }
       Auth: Raw  →  Bearer <Databricks PAT stored in Key Vault>
    3. Reply to message with  @{body('HTTP')?['content']}

  Option B — Azure Bot Service (full Teams app)
    1. Register a Bot in Azure Bot Service
    2. In the bot's message handler, call the endpoint URL (same HTTP pattern)
    3. Install the bot into your Teams channel

  PAT tip: Store the Databricks token in Azure Key Vault and reference it
           from Power Automate's "Key Vault" connector to avoid hardcoding.
""")

# COMMAND ----------

# MAGIC %md ## 6 — Grant endpoint access to principals

from databricks.sdk.service.serving import PermissionLevel

for principal in GRANT_PRINCIPALS:
    try:
        w.serving_endpoints.set_permissions(
            serving_endpoint_id = w.serving_endpoints.get(name=AGENT_ENDPOINT).id,
            access_control_list = [
                iam.AccessControlRequest(
                    group_name       = principal,
                    permission_level = PermissionLevel.CAN_QUERY,
                )
            ],
        )
        print(f"  Granted CAN_QUERY → {principal}")
    except Exception as e:
        print(f"  SKIP ({principal}): {e}")

# COMMAND ----------

# MAGIC %md ## 7 — Inference log table

print(f"""
Inference logs (every request + response) are auto-captured to:

  {CATALOG}.{SCHEMA}.{AGENT_MODEL_NAME}_inference_payload

Query example:
""")

display(spark.sql(f"""
    SELECT
        date_trunc('hour', timestamp)  AS hour,
        COUNT(*)                       AS requests,
        AVG(execution_duration_ms)     AS avg_latency_ms
    FROM {CATALOG}.{SCHEMA}.{AGENT_MODEL_NAME}_inference_payload
    GROUP BY 1
    ORDER BY 1 DESC
    LIMIT 24
""") if spark.catalog.tableExists(f"{CATALOG}.{SCHEMA}.{AGENT_MODEL_NAME}_inference_payload")
    else spark.createDataFrame([], "hour STRING, requests LONG, avg_latency_ms DOUBLE"))

# COMMAND ----------

# MAGIC %md ## Summary

print(f"""
All notebooks complete. Execution order:

  nb_00  →  Create fake jobs + job_runs tables
  nb_01  →  UC Volume + KB Delta table + Grants
  nb_02  →  Generate MD files per job
  nb_03  →  Chunk MD files + Vector Search index
  nb_04  →  Build & log NL-to-SQL agent (Claude Sonnet 4.5)
  nb_05  →  Serve agent as REST endpoint  ← you are here

Endpoint URL:
  {endpoint_url}

To re-index after new jobs are added:
  1. Re-run nb_02 (generates updated MD files)
  2. Re-run nb_03 (re-chunks + triggers VS sync)
  3. No redeploy needed — the serving endpoint stays live
""")
