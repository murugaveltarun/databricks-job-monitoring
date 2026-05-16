# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 04 — NL-to-SQL Agent (Claude Sonnet 4.5)
# MAGIC
# MAGIC Builds a **two-tool ReAct agent** that:
# MAGIC 1. Searches the Vector Search KB for relevant job context
# MAGIC 2. Generates and executes SQL against the live `jobs` / `job_runs` tables
# MAGIC 3. Returns a concise natural-language answer
# MAGIC
# MAGIC The agent is logged to MLflow as a `pyfunc` model with an OpenAI-compatible
# MAGIC chat signature so Notebook 05 can serve it directly.

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch openai tabulate --quiet

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

import json
import re
import textwrap
import time

import mlflow
import openai

from databricks.vector_search.client import VectorSearchClient

log.info("nb_04 started")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 — Prerequisite check

# COMMAND ----------

if not spark.catalog.tableExists(KB_DELTA_TABLE):
    raise RuntimeError(
        f"KB table {KB_DELTA_TABLE!r} not found. Run nb_01 first."
    )

log.info("Prerequisites OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 — Workspace credentials

# COMMAND ----------

_workspace_url = spark.conf.get("spark.databricks.workspaceUrl").strip("/")
_token = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().apiToken().get()
)

_llm_client = openai.OpenAI(
    api_key  = _token,
    base_url = f"https://{_workspace_url}/serving-endpoints",
)

log.info(f"LLM endpoint: {CLAUDE_ENDPOINT}  workspace: {_workspace_url}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 — Tool: Search Knowledge Base

# COMMAND ----------

_vsc = VectorSearchClient(disable_notice=True)


def search_knowledge_base(query: str, num_results: int = 4) -> str:
    """Search the vector KB for job documentation matching the query."""
    try:
        raw  = _vsc.get_index(
            endpoint_name = VS_ENDPOINT_NAME,
            index_name    = VS_INDEX_NAME,
        ).similarity_search(
            query_text  = query,
            columns     = ["job_id", "job_name", "chunk_index", "content"],
            num_results = min(num_results, 8),
        )
        rows = raw.get("result", {}).get("data_array", [])
        if not rows:
            return "No relevant documentation found in the knowledge base."

        parts = []
        for job_id, job_name, chunk_idx, content, score in rows:
            parts.append(
                f"### {job_name} (job_id={job_id}, chunk={chunk_idx}, score={score:.3f})\n"
                f"{content}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception as exc:
        log.error(f"KB search error: {exc}")
        return f"KB search error: {exc}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 — Tool: Execute SQL

# COMMAND ----------

def _is_safe_sql(sql: str) -> bool:
    """Permit only SELECT statements — block all DDL/DML."""
    clean = re.sub(r"--[^\n]*", "", sql)
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)
    first = clean.strip().split()[0].upper() if clean.strip() else ""
    return first == "SELECT"


def execute_sql(sql: str) -> str:
    """Run a SELECT query and return results as a Markdown table (max 50 rows)."""
    if not _is_safe_sql(sql):
        return "ERROR: Only SELECT statements are permitted."
    try:
        pdf = spark.sql(sql).limit(50).toPandas()
        return "Query returned 0 rows." if pdf.empty else pdf.to_markdown(index=False)
    except Exception as exc:
        log.error(f"SQL error: {exc}\nSQL: {sql}")
        return f"SQL execution error: {exc}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 — Tool schemas (OpenAI function-calling format)

# COMMAND ----------

_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the jobs knowledge base for documentation about specific jobs, "
                "their configuration, schedule, tasks, and historical run patterns. "
                "Call this FIRST to understand a job before writing SQL about it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query about jobs or run history.",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "Number of chunks to retrieve (1–8, default 4).",
                        "default": 4,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Execute a SQL SELECT query against the Databricks jobs tables "
                "and return the result as a Markdown table. Only SELECT is allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A valid Spark SQL SELECT statement.",
                    },
                },
                "required": ["sql"],
            },
        },
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 — System prompt

# COMMAND ----------

_SYSTEM_PROMPT = textwrap.dedent(f"""
    You are a Databricks Jobs Intelligence Agent.
    You have access to two tools:
      1. search_knowledge_base — retrieves documentation about jobs from the knowledge base
      2. execute_sql           — runs a SQL SELECT query against live Delta tables

    Available tables:
      • `{JOBS_TABLE}`
        Columns: job_id (INT), job_name (STRING), job_type (STRING), description (STRING),
                 owner (STRING), cluster_type (STRING), schedule_cron (STRING),
                 tags (STRING/JSON), created_at (TIMESTAMP)

      • `{RUNS_TABLE}`
        Columns: run_id (INT), job_id (INT), job_name (STRING), status (STRING),
                 start_time (TIMESTAMP), end_time (TIMESTAMP), duration_seconds (LONG),
                 tasks (STRING), error_message (STRING), triggered_by (STRING), run_url (STRING)
        Status values: SUCCESS, FAILED, TIMEDOUT, CANCELLED

    Rules:
      - Always search the knowledge base first for context before writing SQL.
      - Write precise Spark SQL; use fully-qualified table names with backticks.
      - Never guess job names — query the jobs table when unsure.
      - Present results clearly with Markdown formatting.
      - If a question cannot be answered from the available tables, say so honestly.
      - Lead with the key insight, then supporting data.
""").strip()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 — Agent loop

# COMMAND ----------

_TOOL_MAP: dict[str, Callable] = {
    "search_knowledge_base": search_knowledge_base,
    "execute_sql":           execute_sql,
}

_FINAL_REASONS = {"stop", "end_turn"}


def ask(question: str, verbose: bool = True) -> str:
    """Run the ReAct loop and return the final answer."""
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    for rnd in range(MAX_AGENT_ROUNDS):
        if verbose:
            log.info(f"[Round {rnd + 1}/{MAX_AGENT_ROUNDS}]  calling {CLAUDE_ENDPOINT} …")

        try:
            response = _llm_client.chat.completions.create(
                model       = CLAUDE_ENDPOINT,
                messages    = messages,
                tools       = _TOOLS,
                max_tokens  = MAX_TOKENS,
                temperature = TEMPERATURE,
            )
        except openai.RateLimitError:
            log.warning("Rate limited — waiting 30 s …")
            time.sleep(30)
            continue
        except openai.APIError as exc:
            log.error(f"API error: {exc}")
            raise

        choice = response.choices[0]

        if choice.finish_reason in _FINAL_REASONS:
            answer = choice.message.content or ""
            if verbose:
                print(f"\n{'='*60}\nFINAL ANSWER\n{'='*60}\n{answer}")
            return answer

        if choice.finish_reason == "tool_calls":
            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                if verbose:
                    log.info(f"  → tool: {fn_name}({json.dumps(fn_args)[:120]})")

                result = (
                    _TOOL_MAP[fn_name](**fn_args)
                    if fn_name in _TOOL_MAP
                    else f"ERROR: unknown tool '{fn_name}'"
                )

                if verbose:
                    log.info(f"    ← {result[:300].replace(chr(10), ' ')} …")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })
            continue

        log.warning(f"Unexpected finish_reason: {choice.finish_reason!r}")
        break

    return "Agent could not produce an answer within the allowed rounds."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 — Interactive tests

# COMMAND ----------

_ = ask("Which jobs have the highest failure rate, and what are the most common errors?")

# COMMAND ----------

_ = ask("Give me a summary of all ML-type jobs and their recent run performance.")

# COMMAND ----------

_ = ask("Which job ran the longest on average, and what tasks does it include?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 — MLflow pyfunc model wrapper

# COMMAND ----------

class JobsAgentModel(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper.
    Input : {"messages": [{"role": "user", "content": "..."}]}
    Output: {"role": "assistant", "content": "..."}
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        import openai as _openai
        import os

        self._workspace_url = os.environ.get("DATABRICKS_HOST", "").strip("/")
        self._token         = os.environ.get("DATABRICKS_TOKEN", "")
        self._llm_client    = _openai.OpenAI(
            api_key  = self._token,
            base_url = f"https://{self._workspace_url}/serving-endpoints",
        )

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: Any,
        params: Any = None,
    ) -> dict:
        import pandas as pd

        payload  = model_input.iloc[0].to_dict() if isinstance(model_input, pd.DataFrame) else model_input
        messages = payload.get("messages", [])

        user_text = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        if not user_text:
            return {"role": "assistant", "content": "No question provided."}

        answer = ask(user_text, verbose=False)
        return {"role": "assistant", "content": answer}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 — Log model to MLflow

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name=AGENT_MODEL_NAME) as _run:
    mlflow.log_params({
        "llm_endpoint":    CLAUDE_ENDPOINT,
        "vs_endpoint":     VS_ENDPOINT_NAME,
        "vs_index":        VS_INDEX_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size":      CHUNK_SIZE,
        "chunk_overlap":   CHUNK_OVERLAP,
        "max_rounds":      MAX_AGENT_ROUNDS,
        "max_tokens":      MAX_TOKENS,
        "temperature":     TEMPERATURE,
    })

    _model_info = mlflow.pyfunc.log_model(
        artifact_path         = AGENT_MODEL_NAME,
        python_model          = JobsAgentModel(),
        registered_model_name = f"{CATALOG}.{SCHEMA}.{AGENT_MODEL_NAME}",
        pip_requirements      = [
            "databricks-vectorsearch",
            "openai",
            "tabulate",
        ],
    )

log.info(f"Model logged  : {_model_info.model_uri}")
log.info(f"Run ID        : {_run.info.run_id}")
log.info(f"Registered as : {CATALOG}.{SCHEMA}.{AGENT_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 — Sanity check: load & call the logged model

# COMMAND ----------

_loaded  = mlflow.pyfunc.load_model(_model_info.model_uri)
_payload = {"messages": [{"role": "user", "content": "How many total job runs are in the system?"}]}
_result  = _loaded.predict(_payload)
print(_result["content"])

log.info("nb_04 complete")
