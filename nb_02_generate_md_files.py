# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 02 — Generate Markdown Files per Job
# MAGIC
# MAGIC For every job in the `jobs` table this notebook:
# MAGIC 1. Runs an aggregation query joining `jobs` + `job_runs`
# MAGIC 2. Builds a rich Markdown document with job metadata, run statistics,
# MAGIC    recent run history, and common errors
# MAGIC 3. Writes one `.md` file per job to the UC Volume path defined in `config.py`
# MAGIC
# MAGIC File naming: `{job_name}__{job_id}.md`  (double-underscore avoids conflicts)

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import json
from datetime import datetime
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## 1 — Aggregate job statistics

stats_df = spark.sql(f"""
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
        ROUND(MAX(r.duration_seconds), 0)                           AS max_duration_sec,

        MAX(r.start_time)                                            AS last_run_start,
        FIRST_VALUE(r.status) OVER (
            PARTITION BY j.job_id ORDER BY r.start_time DESC
        )                                                            AS last_run_status,
        FIRST_VALUE(r.duration_seconds) OVER (
            PARTITION BY j.job_id ORDER BY r.start_time DESC
        )                                                            AS last_run_duration_sec,
        FIRST_VALUE(r.tasks) OVER (
            PARTITION BY j.job_id ORDER BY r.start_time DESC
        )                                                            AS last_run_tasks,
        FIRST_VALUE(r.triggered_by) OVER (
            PARTITION BY j.job_id ORDER BY r.start_time DESC
        )                                                            AS last_run_triggered_by,
        FIRST_VALUE(r.run_url) OVER (
            PARTITION BY j.job_id ORDER BY r.start_time DESC
        )                                                            AS last_run_url
    FROM {JOBS_TABLE} j
    LEFT JOIN {RUNS_TABLE} r USING (job_id)
    GROUP BY ALL
""").dropDuplicates(["job_id"])

print(f"Jobs to document: {stats_df.count()}")
display(stats_df.select("job_id","job_name","total_runs","success_count","failed_count","success_rate_pct").limit(5))

# COMMAND ----------

# MAGIC %md ## 2 — Helper: recent runs per job

def get_recent_runs(job_id: int, limit: int = 10):
    return spark.sql(f"""
        SELECT
            run_id,
            status,
            date_format(start_time, 'yyyy-MM-dd HH:mm') AS start_time,
            date_format(end_time,   'yyyy-MM-dd HH:mm') AS end_time,
            duration_seconds,
            triggered_by,
            tasks,
            COALESCE(error_message, '') AS error_message,
            run_url
        FROM {RUNS_TABLE}
        WHERE job_id = {job_id}
        ORDER BY start_time DESC
        LIMIT {limit}
    """).collect()

# COMMAND ----------

# MAGIC %md ## 3 — Helper: top error messages per job

def get_top_errors(job_id: int, limit: int = 5):
    return spark.sql(f"""
        SELECT
            error_message,
            COUNT(*) AS occurrences,
            status
        FROM {RUNS_TABLE}
        WHERE job_id = {job_id}
          AND error_message IS NOT NULL
          AND error_message != ''
        GROUP BY error_message, status
        ORDER BY occurrences DESC
        LIMIT {limit}
    """).collect()

# COMMAND ----------

# MAGIC %md ## 4 — Markdown builder

STATUS_EMOJI = {
    "SUCCESS":   "✅",
    "FAILED":    "❌",
    "TIMEDOUT":  "⏱️",
    "CANCELLED": "🚫",
}

def fmt_duration(seconds):
    if seconds is None:
        return "N/A"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"

def fmt_ts(ts):
    return ts.strftime("%Y-%m-%d %H:%M UTC") if ts else "N/A"

def parse_tags(tags_str):
    try:
        return json.loads(tags_str) if tags_str else {}
    except Exception:
        return {}

def build_markdown(row, recent_runs, top_errors) -> str:
    tags = parse_tags(row.tags)
    tags_md = ", ".join(f"`{k}:{v}`" for k, v in tags.items()) or "—"

    last_status_icon = STATUS_EMOJI.get(row.last_run_status, "❓")

    lines = [
        f"# {row.job_name}",
        f"",
        f"> **Job ID:** {row.job_id}  |  **Type:** {row.job_type}  |  "
        f"**Owner:** {row.owner}",
        f"",
        f"## Overview",
        f"",
        f"{row.description}",
        f"",
        f"## Configuration",
        f"",
        f"| Property        | Value |",
        f"|-----------------|-------|",
        f"| Job ID          | `{row.job_id}` |",
        f"| Job Name        | `{row.job_name}` |",
        f"| Job Type        | {row.job_type} |",
        f"| Owner           | {row.owner} |",
        f"| Cluster Type    | {row.cluster_type} |",
        f"| Schedule (cron) | `{row.schedule_cron}` |",
        f"| Tags            | {tags_md} |",
        f"| Created At      | {fmt_ts(row.created_at)} |",
        f"",
        f"## Run Statistics",
        f"",
        f"| Metric              | Value |",
        f"|---------------------|-------|",
        f"| Total Runs          | {row.total_runs} |",
        f"| ✅ Success           | {row.success_count} |",
        f"| ❌ Failed            | {row.failed_count} |",
        f"| ⏱️ Timed Out         | {row.timeout_count} |",
        f"| 🚫 Cancelled         | {row.cancelled_count} |",
        f"| Success Rate        | **{row.success_rate_pct}%** |",
        f"| Avg Duration        | {fmt_duration(row.avg_duration_sec)} |",
        f"| Min Duration        | {fmt_duration(row.min_duration_sec)} |",
        f"| Max Duration        | {fmt_duration(row.max_duration_sec)} |",
        f"",
        f"## Last Run",
        f"",
        f"| Property     | Value |",
        f"|--------------|-------|",
        f"| Status       | {last_status_icon} {row.last_run_status} |",
        f"| Start Time   | {fmt_ts(row.last_run_start)} |",
        f"| Duration     | {fmt_duration(row.last_run_duration_sec)} |",
        f"| Tasks        | {row.last_run_tasks or 'N/A'} |",
        f"| Triggered By | {row.last_run_triggered_by or 'N/A'} |",
        f"| Run URL      | {row.last_run_url or 'N/A'} |",
        f"",
    ]

    # ── Recent runs table ──────────────────────────────────────────────────
    lines += [
        f"## Recent Runs (last {len(recent_runs)})",
        f"",
        f"| Run ID | Status | Start Time | End Time | Duration | Triggered By | Tasks |",
        f"|--------|--------|------------|----------|----------|--------------|-------|",
    ]
    for r in recent_runs:
        icon = STATUS_EMOJI.get(r.status, "❓")
        lines.append(
            f"| {r.run_id} | {icon} {r.status} | {r.start_time} | {r.end_time} "
            f"| {fmt_duration(r.duration_seconds)} | {r.triggered_by} | {r.tasks} |"
        )
    lines.append("")

    # ── Top errors ─────────────────────────────────────────────────────────
    if top_errors:
        lines += [
            f"## Top Error Messages",
            f"",
            f"| # | Status | Occurrences | Error Message |",
            f"|---|--------|-------------|---------------|",
        ]
        for i, e in enumerate(top_errors, 1):
            icon = STATUS_EMOJI.get(e.status, "❓")
            msg  = e.error_message.replace("|", "\\|")[:120]
            lines.append(f"| {i} | {icon} {e.status} | {e.occurrences} | `{msg}` |")
        lines.append("")

    lines += [
        f"---",
        f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
        f"*Source tables: `{JOBS_TABLE}`, `{RUNS_TABLE}`*",
    ]

    return "\n".join(lines)

# COMMAND ----------

# MAGIC %md ## 5 — Write one MD file per job

def safe_filename(job_name: str) -> str:
    return job_name.replace(" ", "_").replace("/", "-").lower()

job_rows = stats_df.collect()
written  = []
failed   = []

for row in job_rows:
    try:
        recent_runs = get_recent_runs(row.job_id, limit=10)
        top_errors  = get_top_errors(row.job_id,  limit=5)
        md_content  = build_markdown(row, recent_runs, top_errors)

        filename    = f"{safe_filename(row.job_name)}__{row.job_id}.md"
        volume_path = f"{MD_FILES_PATH}/{filename}"

        # Write via dbutils (works with UC Volumes)
        dbutils.fs.put(volume_path, md_content, overwrite=True)
        written.append((row.job_id, row.job_name, filename))
        print(f"  ✓  {filename}")

    except Exception as e:
        failed.append((row.job_id, row.job_name, str(e)))
        print(f"  ✗  job_id={row.job_id} ({row.job_name}): {e}")

print(f"\nDone — {len(written)} written, {len(failed)} failed.")

# COMMAND ----------

# MAGIC %md ## 6 — Verify files in volume

files = dbutils.fs.ls(MD_FILES_PATH)
print(f"Files in {MD_FILES_PATH}:\n")
for f in sorted(files, key=lambda x: x.name):
    print(f"  {f.name:60s}  {f.size:>8,} bytes")
