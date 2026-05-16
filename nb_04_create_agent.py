# Databricks notebook source
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

# MAGIC %pip install databricks-vectorsearch openai --quiet
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import json
import re
import textwrap
import mlflow
import openai

from databricks.vector_search.client import VectorSearchClient

# COMMAND ----------

# MAGIC %md ## 1 — Workspace credentials
# MAGIC
# MAGIC The Databricks token is read from the notebook context — no manual secret needed
# MAGIC when running interactively. For production, store in a Databricks secret scope.

workspace_url = (
    spark.conf.get("spark.databricks.workspaceUrl")
    .strip("/")
)
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# OpenAI-compatible client pointed at Databricks Foundation Model APIs
llm_client = openai.OpenAI(
    api_key  = token,
    base_url = f"https://{workspace_url}/serving-endpoints",
)

print(f"Workspace : {workspace_url}")
print(f"LLM model : {CLAUDE_ENDPOINT}")

# COMMAND ----------

# MAGIC %md ## 2 — Tool: Search Knowledge Base

vsc = VectorSearchClient(disable_notice=True)

def search_knowledge_base(query: str, num_results: int = 4) -> str:
    """Search the vector KB for job documentation matching the query."""
    try:
        raw = vsc.get_index(
            endpoint_name = VS_ENDPOINT_NAME,
            index_name    = VS_INDEX_NAME,
        ).similarity_search(
            query_text  = query,
            columns     = ["job_id", "job_name", "chunk_index", "content"],
            num_results = num_results,
        )
        rows = raw.get("result", {}).get("data_array", [])
        if not rows:
            return "No relevant documentation found."

        parts = []
        for job_id, job_name, chunk_idx, content, score in rows:
            parts.append(
                f"### {job_name} (job_id={job_id}, chunk={chunk_idx}, score={score:.3f})\n{content}"
            )
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        return f"KB search error: {e}"

# COMMAND ----------

# MAGIC %md ## 3 — Tool: Execute SQL

_ALLOWED_TABLES = {JOBS_TABLE, RUNS_TABLE, KB_DELTA_TABLE}

def _is_safe_sql(sql: str) -> bool:
    """Block any write/DDL statement — only SELECT is permitted."""
    clean = re.sub(r"--[^\n]*", "", sql)          # strip single-line comments
    clean = re.sub(r"/\*.*?\*/", "", clean, flags=re.DOTALL)
    first = clean.strip().split()[0].upper() if clean.strip() else ""
    return first == "SELECT"

def execute_sql(sql: str) -> str:
    """Run a SELECT query and return results as a Markdown table (max 50 rows)."""
    if not _is_safe_sql(sql):
        return "ERROR: Only SELECT statements are permitted."
    try:
        df  = spark.sql(sql).limit(50)
        pdf = df.toPandas()
        if pdf.empty:
            return "Query returned 0 rows."
        return pdf.to_markdown(index=False)
    except Exception as e:
        return f"SQL execution error: {e}"

# COMMAND ----------

# MAGIC %md ## 4 — Tool schemas (OpenAI function-calling format)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the jobs knowledge base for documentation about specific jobs, "
                "their configuration, schedule, tasks, and historical run patterns. "
                "Use this FIRST to understand a job before writing SQL about it."
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
                        "description": "Number of chunks to retrieve (default 4, max 8).",
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
                "Execute a SQL SELECT query against the Databricks jobs tables and return "
                "the result as a Markdown table. Only SELECT is allowed."
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

# MAGIC %md ## 5 — System prompt

SYSTEM_PROMPT = textwrap.dedent(f"""
    You are a Databricks Jobs Intelligence Agent.
    You have access to two tools:
      1. search_knowledge_base — retrieves rich documentation about jobs from the knowledge base
      2. execute_sql — runs a SQL SELECT query against live Delta tables

    Available tables:
      • {JOBS_TABLE}
        Columns: job_id (INT), job_name (STRING), job_type (STRING), description (STRING),
                 owner (STRING), cluster_type (STRING), schedule_cron (STRING),
                 tags (STRING/JSON), created_at (TIMESTAMP)

      • {RUNS_TABLE}
        Columns: run_id (INT), job_id (INT), job_name (STRING), status (STRING),
                 start_time (TIMESTAMP), end_time (TIMESTAMP), duration_seconds (LONG),
                 tasks (STRING), error_message (STRING), triggered_by (STRING), run_url (STRING)
        Status values: SUCCESS, FAILED, TIMEDOUT, CANCELLED

    Rules:
      - Always search the knowledge base first for context before writing SQL.
      - Write precise Spark SQL; use backtick-quoted fully-qualified table names.
      - Never guess — if unsure about a job name, query the jobs table.
      - Present results clearly with Markdown formatting.
      - If a question cannot be answered from the available tables, say so honestly.
      - Keep final answers concise: lead with the key insight, then supporting data.
""").strip()

# COMMAND ----------

# MAGIC %md ## 6 — Agent loop (`ask` function)

TOOL_MAP = {
    "search_knowledge_base": search_knowledge_base,
    "execute_sql":           execute_sql,
}

def ask(question: str, max_rounds: int = 6, verbose: bool = True) -> str:
    """
    Run the ReAct agent loop until Claude returns a final answer.
    Returns the assistant's final text response.
    """
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": question},
    ]

    for rnd in range(max_rounds):
        if verbose:
            print(f"\n[Round {rnd + 1}]  Calling {CLAUDE_ENDPOINT} …")

        response = llm_client.chat.completions.create(
            model       = CLAUDE_ENDPOINT,
            messages    = messages,
            tools       = TOOLS,
            max_tokens  = 2048,
            temperature = 0,
        )

        choice = response.choices[0]

        # ── Final answer ──────────────────────────────────────────────────
        if choice.finish_reason in ("stop", "end_turn"):
            answer = choice.message.content
            if verbose:
                print(f"\n{'='*60}\nFINAL ANSWER\n{'='*60}\n{answer}")
            return answer

        # ── Tool call(s) ──────────────────────────────────────────────────
        if choice.finish_reason == "tool_calls":
            assistant_msg = choice.message
            messages.append(assistant_msg)          # assistant turn with tool_calls

            for tc in assistant_msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)

                if verbose:
                    print(f"  → tool: {fn_name}({json.dumps(fn_args)[:120]})")

                if fn_name not in TOOL_MAP:
                    result = f"ERROR: unknown tool '{fn_name}'"
                else:
                    result = TOOL_MAP[fn_name](**fn_args)

                if verbose:
                    preview = result[:300].replace("\n", " ")
                    print(f"    ← {preview} …")

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "content":      result,
                })
            continue

        # Unexpected finish reason
        break

    return "Agent could not produce an answer within the allowed rounds."

# COMMAND ----------

# MAGIC %md ## 7 — Interactive tests

q1 = "Which jobs have the highest failure rate, and what are the most common errors?"
_ = ask(q1)

# COMMAND ----------

q2 = "Give me a summary of all ML-type jobs and their recent run performance."
_ = ask(q2)

# COMMAND ----------

q3 = "Which job ran the longest on average, and what tasks does it include?"
_ = ask(q3)

# COMMAND ----------

# MAGIC %md ## 8 — Wrap as MLflow pyfunc model

class JobsAgentModel(mlflow.pyfunc.PythonModel):
    """
    MLflow pyfunc wrapper.
    Input : {"messages": [{"role": "user", "content": "..."}]}
    Output: {"role": "assistant", "content": "..."}
    """

    def load_context(self, context):
        import json, re, textwrap, openai
        from databricks.vector_search.client import VectorSearchClient
        # All globals (config vars, clients, helpers) are already in scope
        # when running inside Databricks — no serialisation needed.

    def predict(self, context, model_input, params=None):
        import pandas as pd

        # Accept both dict and DataFrame inputs
        if isinstance(model_input, pd.DataFrame):
            payload = model_input.iloc[0].to_dict()
        else:
            payload = model_input

        messages = payload.get("messages", [])
        # Extract the last user message
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break

        if not user_text:
            return {"role": "assistant", "content": "No question provided."}

        answer = ask(user_text, verbose=False)
        return {"role": "assistant", "content": answer}

# COMMAND ----------

# MAGIC %md ## 9 — Log model to MLflow

mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run(run_name=AGENT_MODEL_NAME) as run:
    mlflow.log_params({
        "llm_endpoint":      CLAUDE_ENDPOINT,
        "vs_endpoint":       VS_ENDPOINT_NAME,
        "vs_index":          VS_INDEX_NAME,
        "embedding_model":   EMBEDDING_MODEL,
        "chunk_size":        CHUNK_SIZE,
        "chunk_overlap":     CHUNK_OVERLAP,
    })

    model_info = mlflow.pyfunc.log_model(
        artifact_path      = AGENT_MODEL_NAME,
        python_model       = JobsAgentModel(),
        registered_model_name = f"{CATALOG}.{SCHEMA}.{AGENT_MODEL_NAME}",
        pip_requirements   = [
            "databricks-vectorsearch",
            "openai",
            "tabulate",          # required by pandas .to_markdown()
        ],
    )

    run_id = run.info.run_id

print(f"Model logged  : {model_info.model_uri}")
print(f"Run ID        : {run_id}")
print(f"Registered    : {CATALOG}.{SCHEMA}.{AGENT_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md ## 10 — Quick sanity check: load & call the logged model

loaded = mlflow.pyfunc.load_model(model_info.model_uri)
test_payload = {"messages": [{"role": "user", "content": "How many total job runs are in the system?"}]}
result = loaded.predict(test_payload)
print(result["content"])
