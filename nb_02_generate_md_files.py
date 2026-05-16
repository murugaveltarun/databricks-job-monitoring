# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Notebook 02 — Generate Markdown Files per Job
# MAGIC
# MAGIC For every job in the `jobs` table this notebook:
# MAGIC 1. Runs an aggregation query joining `jobs` + `job_runs`
# MAGIC 2. Builds a rich Markdown document (metadata, stats, run history, errors)
# MAGIC 3. Writes one `.md` file per job to the UC Volume
# MAGIC
# MAGIC **File naming:** `{job_name}__{job_id}.md`

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
from datetime import datetime, timezone

log.info("nb_02 started")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0 — Prerequisite check

# COMMAND ----------

for _tbl in [JOBS_TABLE, RUNS_TABLE]:
    if not spark.catalog.tableExists(_tbl):
        raise RuntimeError(
            f"Required table {_tbl!r} not found. Run nb_00 first."
        )

if not dbutils.fs.ls(MD_FILES_PATH) or True:  # always ensure directory exists
    dbutils.fs.mkdirs(MD_FILES_PATH)

log.info("Prerequisites OK")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 — Aggregate job statistics

# COMMAND ----------

stats_df = spark.sql(f"""
    WITH last_run AS (
        SELECT
            job_id,
            status           AS last_run_status,
            start_time       AS last_run_start,
            duration_seconds AS last_run_duration_sec,
            tasks            AS last_run_tasks,
            triggered_by     AS last_run_triggered_by,
            run_url          AS last_run_url,
            ROW_NUMBER() OVER (PARTITION BY job_id ORDER BY start_time DESC) AS rn
        FROM {RUNS_TABLE}
    ),
    agg AS (
        SELECT
            j.job_id,
            j.job_name,
            j.job_type,
            j.description,
            j.owner,
            j.cluster_type,
            j.schedule_cron,
            j.tags,
            j.created_at,

            COUNT(r.run_id)                                              AS total_runs,
            SUM(CASE WHEN r.status = 'SUCCESS'   THEN 1 ELSE 0 END)     AS success_count,
            SUM(CASE WHEN r.status = 'FAILED'    THEN 1 ELSE 0 END)     AS failed_count,
            SUM(CASE WHEN r.status = 'TIMEDOUT'  THEN 1 ELSE 0 END)     AS timeout_count,
            SUM(CASE WHEN r.status = 'CANCELLED' THEN 1 ELSE 0 END)     AS cancelled_count,
            ROUND(
                SUM(CASE WHEN r.status = 'SUCCESS' THEN 1 ELSE 0 END)
                / NULLIF(COUNT(r.run_id), 0) * 100, 1
            )                                                            AS success_rate_pct,

            ROUND(AVG(r.duration_seconds), 0)                           AS avg_duration_sec,
            ROUND(MIN(r.duration_seconds), 0)                           AS min_duration_sec,
            ROUND(MAX(r.duration_seconds), 0)                           AS max_duration_sec
        FROM {JOBS_TABLE} j
        LEFT JOIN {RUNS_TABLE} r USING (job_id)
        GROUP BY
            j.job_id, j.job_name, j.job_type, j.description, j.owner,
            j.cluster_type, j.schedule_cron, j.tags, j.created_at
    )
    SELECT
        a.*,
        lr.last_run_status,
        lr.last_run_start,
        lr.last_run_duration_sec,
        lr.last_run_tasks,
        lr.last_run_triggered_by,
        lr.last_run_url
    FROM agg a
    LEFT JOIN last_run lr ON a.job_id = lr.job_id AND lr.rn = 1
""")

job_count = stats_df.count()
log.info(f"Jobs to document: {job_count}")
display(stats_df.select(
    "job_id", "job_name", "total_runs",
    "success_count", "failed_count", "success_rate_pct",
).limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 — Query helpers

# COMMAND ----------

def get_recent_runs(job_id: int, limit: int = 10) -> list:
    return spark.sql(f"""
        SELECT
            run_id,
            status,
            date_format(start_time, 'yyyy-MM-dd HH:mm') AS start_time,
            date_format(end_time,   'yyyy-MM-dd HH:mm') AS end_time,
            duration_seconds,
            triggered_by,
            tasks,
            COALESCE(error_message, '')                  AS error_message,
            run_url
        FROM {RUNS_TABLE}
        WHERE job_id = {job_id}
        ORDER BY start_time DESC
        LIMIT {limit}
    """).collect()


def get_top_errors(job_id: int, limit: int = 5) -> list:
    return spark.sql(f"""
        SELECT
            error_message,
            status,
            COUNT(*) AS occurrences
        FROM {RUNS_TABLE}
        WHERE job_id = {job_id}
          AND error_message IS NOT NULL
          AND error_message != ''
        GROUP BY error_message, status
        ORDER BY occurrences DESC
        LIMIT {limit}
    """).collect()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 — Markdown builder

# COMMAND ----------

_STATUS_EMOJI: dict[str, str] = {
    "SUCCESS":   "✅",
    "FAILED":    "❌",
    "TIMEDOUT":  "⏱️",
    "CANCELLED": "🚫",
}


def _fmt_duration(seconds: Any) -> str:
    if seconds is None:
        return "N/A"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _fmt_ts(ts: Any) -> str:
    return ts.strftime("%Y-%m-%d %H:%M UTC") if ts else "N/A"


def _parse_tags(tags_str: Any) -> dict:
    try:
        return json.loads(tags_str) if tags_str else {}
    except (ValueError, TypeError):
        return {}


def build_markdown(row: Any, recent_runs: list, top_errors: list) -> str:
    tags     = _parse_tags(row.tags)
    tags_md  = ", ".join(f"`{k}:{v}`" for k, v in tags.items()) or "—"
    last_icon = _STATUS_EMOJI.get(row.last_run_status or "", "❓")

    lines: list[str] = [
        f"# {row.job_name}",
        "",
        f"> **Job ID:** {row.job_id}  |  **Type:** {row.job_type}  |  **Owner:** {row.owner}",
        "",
        "## Overview",
        "",
        f"{row.description}",
        "",
        "## Configuration",
        "",
        "| Property        | Value |",
        "|-----------------|-------|",
        f"| Job ID          | `{row.job_id}` |",
        f"| Job Name        | `{row.job_name}` |",
        f"| Job Type        | {row.job_type} |",
        f"| Owner           | {row.owner} |",
        f"| Cluster Type    | {row.cluster_type} |",
        f"| Schedule (cron) | `{row.schedule_cron}` |",
        f"| Tags            | {tags_md} |",
        f"| Created At      | {_fmt_ts(row.created_at)} |",
        "",
        "## Run Statistics",
        "",
        "| Metric           | Value |",
        "|------------------|-------|",
        f"| Total Runs       | {row.total_runs} |",
        f"| ✅ Success        | {row.success_count} |",
        f"| ❌ Failed         | {row.failed_count} |",
        f"| ⏱️ Timed Out      | {row.timeout_count} |",
        f"| 🚫 Cancelled      | {row.cancelled_count} |",
        f"| Success Rate     | **{row.success_rate_pct}%** |",
        f"| Avg Duration     | {_fmt_duration(row.avg_duration_sec)} |",
        f"| Min Duration     | {_fmt_duration(row.min_duration_sec)} |",
        f"| Max Duration     | {_fmt_duration(row.max_duration_sec)} |",
        "",
        "## Last Run",
        "",
        "| Property     | Value |",
        "|--------------|-------|",
        f"| Status       | {last_icon} {row.last_run_status} |",
        f"| Start Time   | {_fmt_ts(row.last_run_start)} |",
        f"| Duration     | {_fmt_duration(row.last_run_duration_sec)} |",
        f"| Tasks        | {row.last_run_tasks or 'N/A'} |",
        f"| Triggered By | {row.last_run_triggered_by or 'N/A'} |",
        f"| Run URL      | {row.last_run_url or 'N/A'} |",
        "",
        f"## Recent Runs (last {len(recent_runs)})",
        "",
        "| Run ID | Status | Start Time | End Time | Duration | Triggered By | Tasks |",
        "|--------|--------|------------|----------|----------|--------------|-------|",
    ]

    for r in recent_runs:
        icon = _STATUS_EMOJI.get(r.status, "❓")
        lines.append(
            f"| {r.run_id} | {icon} {r.status} | {r.start_time} | {r.end_time} "
            f"| {_fmt_duration(r.duration_seconds)} | {r.triggered_by} | {r.tasks} |"
        )
    lines.append("")

    if top_errors:
        lines += [
            "## Top Error Messages",
            "",
            "| # | Status | Occurrences | Error Message |",
            "|---|--------|-------------|---------------|",
        ]
        for i, e in enumerate(top_errors, 1):
            icon = _STATUS_EMOJI.get(e.status, "❓")
            msg  = e.error_message.replace("|", "\\|")[:120]
            lines.append(f"| {i} | {icon} {e.status} | {e.occurrences} | `{msg}` |")
        lines.append("")

    lines += [
        "---",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        f"*Source tables: `{JOBS_TABLE}`, `{RUNS_TABLE}`*",
    ]
    return "\n".join(lines)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 — Write one MD file per job

# COMMAND ----------

def _safe_filename(job_name: str) -> str:
    return job_name.replace(" ", "_").replace("/", "-").lower()


_job_rows  = stats_df.collect()
_written: list[tuple[int, str, str]] = []
_failed:  list[tuple[int, str, str]] = []

for _row in _job_rows:
    try:
        _recent = get_recent_runs(_row.job_id, limit=10)
        _errors = get_top_errors(_row.job_id,  limit=5)
        _md     = build_markdown(_row, _recent, _errors)

        _filename    = f"{_safe_filename(_row.job_name)}__{_row.job_id}.md"
        _volume_path = f"{MD_FILES_PATH}/{_filename}"

        dbutils.fs.put(_volume_path, _md, overwrite=True)
        _written.append((_row.job_id, _row.job_name, _filename))
        log.info(f"  written: {_filename}")

    except Exception as _exc:
        _failed.append((_row.job_id, _row.job_name, str(_exc)))
        log.error(f"  FAILED job_id={_row.job_id} ({_row.job_name}): {_exc}")

log.info(f"nb_02 done — {len(_written)} written, {len(_failed)} failed")

if _failed:
    raise RuntimeError(
        f"{len(_failed)} MD file(s) could not be written: "
        + str([f"{jid} {jn}" for jid, jn, _ in _failed])
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 — Verify files in volume

# COMMAND ----------

_files = dbutils.fs.ls(MD_FILES_PATH)
print(f"Files in {MD_FILES_PATH} ({len(_files)} total):\n")
for _f in sorted(_files, key=lambda x: x.name):
    print(f"  {_f.name:62s}  {_f.size:>8,} bytes")
