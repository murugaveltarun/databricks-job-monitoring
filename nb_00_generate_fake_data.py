# Databricks notebook source
# MAGIC %md
# MAGIC # Notebook 00 — Generate Fake Jobs & Job Runs Tables
# MAGIC
# MAGIC Creates two Delta tables under the catalog/schema defined in `config.py`:
# MAGIC - **`jobs`** — one row per job (metadata, schedule, owner, description)
# MAGIC - **`job_runs`** — run history for each job (status, duration, tasks, errors)
# MAGIC
# MAGIC Run this once to bootstrap the data before running any other notebook.

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

import random
from datetime import datetime, timedelta
from pyspark.sql import Row
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, StringType, TimestampType, LongType
)

random.seed(42)

# COMMAND ----------
# MAGIC %md ## 1 — Schema setup

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
print(f"Catalog '{CATALOG}' and schema '{SCHEMA}' are ready.")

# COMMAND ----------
# MAGIC %md ## 2 — Reference data

JOB_TEMPLATES = [
    ("customer_etl_pipeline",       "ETL",         "Ingests raw customer transactions from S3 and loads into gold layer."),
    ("ml_churn_training",           "ML",          "Retrains the customer churn XGBoost model on the latest 90-day window."),
    ("daily_revenue_aggregation",   "ETL",         "Aggregates daily revenue metrics and writes to the finance summary table."),
    ("inventory_sync_job",          "ETL",         "Syncs inventory snapshots from the ERP system into Delta Lake."),
    ("data_quality_checks",         "DQ",          "Runs Great Expectations suites against all silver-layer tables."),
    ("product_recommendation_train","ML",          "Trains the collaborative-filtering recommendation model weekly."),
    ("clickstream_processing",      "Streaming",   "Processes clickstream events from Kafka and writes to silver tables."),
    ("nightly_user_profile_update", "ETL",         "Refreshes user profile features used by downstream ML models."),
    ("fraud_detection_scoring",     "ML",          "Batch-scores all transactions from the past 24 h for fraud risk."),
    ("hr_reporting_export",         "Reporting",   "Generates HR KPI reports and exports them to SharePoint."),
    ("log_archival_job",            "Maintenance", "Compresses and archives application logs older than 90 days."),
    ("db_health_monitor",           "Maintenance", "Checks cluster health, auto-optimises tables, and sends alerts."),
    ("sales_forecast_pipeline",     "ML",          "Runs the ARIMA sales forecasting pipeline for the next 30 days."),
    ("partner_data_ingestion",      "ETL",         "Pulls partner API feeds and normalises them into the bronze layer."),
    ("compliance_audit_export",     "Reporting",   "Exports compliance audit logs to the legal team's secure storage."),
]

OWNERS = [
    "alice.chen@company.com", "bob.kumar@company.com",
    "carol.smith@company.com", "david.lee@company.com",
    "emily.jones@company.com",
]

CLUSTER_TYPES = ["job_cluster", "all_purpose_cluster"]

SCHEDULES = [
    "0 2 * * *",    # daily 2 AM
    "0 6 * * 1",    # weekly Monday 6 AM
    "0 */4 * * *",  # every 4 hours
    "30 1 * * *",   # daily 1:30 AM
    "0 0 * * 0",    # weekly Sunday midnight
    "0 8 * * 1-5",  # weekdays 8 AM
]

TASK_POOL = [
    "ingest_raw_data", "validate_schema", "transform_silver",
    "aggregate_gold", "train_model", "evaluate_model",
    "export_results", "send_notifications", "run_dq_checks",
    "update_feature_store", "archive_logs", "send_alerts",
    "load_lookup_tables", "refresh_dashboard_cache",
]

ERROR_MESSAGES = {
    "FAILED": [
        "SparkException: Job aborted due to stage failure: Task 12 in stage 3.0 failed 4 times.",
        "AnalysisException: Table or view not found: bronze.raw_events.",
        "py4j.protocol.Py4JJavaError: An error occurred while calling o234.parquet.",
        "ConnectionRefusedError: [Errno 111] Connection refused (upstream API unreachable).",
        "AssertionError: Data quality check failed — null rate 18% exceeds threshold 5%.",
        "OutOfMemoryError: GC overhead limit exceeded on executor 7.",
        "FileNotFoundException: s3://data-lake/raw/2024-03-15/ does not exist.",
    ],
    "TIMEDOUT": [
        "Job exceeded maximum allowed duration of 3600 seconds.",
        "Cluster autoscaling could not provision nodes within timeout window.",
        "Task 'train_model' ran for 7200 s and was forcefully terminated.",
    ],
    "CANCELLED": [
        "Manually cancelled by alice.chen@company.com.",
        "Cancelled due to upstream job failure (dependency check).",
        "Cancelled by orchestration system — maintenance window started.",
    ],
}

STATUSES = ["SUCCESS", "FAILED", "TIMEDOUT", "CANCELLED"]
STATUS_WEIGHTS = [0.72, 0.17, 0.07, 0.04]

# COMMAND ----------
# MAGIC %md ## 3 — Generate jobs table

def make_jobs():
    rows = []
    for idx, (name, job_type, description) in enumerate(JOB_TEMPLATES, start=1001):
        rows.append(Row(
            job_id=idx,
            job_name=name,
            job_type=job_type,
            description=description,
            owner=random.choice(OWNERS),
            cluster_type=random.choice(CLUSTER_TYPES),
            schedule_cron=random.choice(SCHEDULES),
            tags=f'{{"env":"production","team":"{job_type.lower()}"}}',
            created_at=datetime(2023, random.randint(1,12), random.randint(1,28),
                                random.randint(0,23), random.randint(0,59)),
        ))
    return rows

jobs_schema = StructType([
    StructField("job_id",        IntegerType(),   False),
    StructField("job_name",      StringType(),    False),
    StructField("job_type",      StringType(),    False),
    StructField("description",   StringType(),    True),
    StructField("owner",         StringType(),    True),
    StructField("cluster_type",  StringType(),    True),
    StructField("schedule_cron", StringType(),    True),
    StructField("tags",          StringType(),    True),
    StructField("created_at",    TimestampType(), True),
])

jobs_df = spark.createDataFrame(make_jobs(), schema=jobs_schema)
jobs_df.write.format("delta").mode("overwrite").saveAsTable(JOBS_TABLE)
print(f"Written {jobs_df.count()} rows → {JOBS_TABLE}")

# COMMAND ----------
# MAGIC %md ## 4 — Generate job_runs table

def random_tasks():
    n = random.randint(2, 6)
    return ", ".join(random.sample(TASK_POOL, n))

def make_runs(jobs):
    rows = []
    run_id = 5001
    base_date = datetime(2024, 1, 1)

    for job in jobs:
        # 30–90 historical runs per job
        n_runs = random.randint(30, 90)
        for i in range(n_runs):
            status = random.choices(STATUSES, weights=STATUS_WEIGHTS, k=1)[0]

            # spread runs across the past ~16 months
            start = base_date + timedelta(
                days=random.randint(0, 480),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )

            if status == "SUCCESS":
                duration = random.randint(120, 3200)
            elif status == "TIMEDOUT":
                duration = random.randint(3600, 7200)
            elif status == "FAILED":
                duration = random.randint(30, 2000)
            else:  # CANCELLED
                duration = random.randint(10, 600)

            end = start + timedelta(seconds=duration)
            error = None
            if status in ERROR_MESSAGES:
                error = random.choice(ERROR_MESSAGES[status])

            rows.append(Row(
                run_id=run_id,
                job_id=job["job_id"],
                job_name=job["job_name"],
                status=status,
                start_time=start,
                end_time=end,
                duration_seconds=duration,
                tasks=random_tasks(),
                error_message=error,
                triggered_by=random.choice(["SCHEDULE", "MANUAL", "API"]),
                run_url=f"https://adb-xxxx.azuredatabricks.net/#job/{job['job_id']}/run/{run_id}",
            ))
            run_id += 1
    return rows

jobs_local = [row.asDict() for row in jobs_df.collect()]

runs_schema = StructType([
    StructField("run_id",           IntegerType(),   False),
    StructField("job_id",           IntegerType(),   False),
    StructField("job_name",         StringType(),    False),
    StructField("status",           StringType(),    False),
    StructField("start_time",       TimestampType(), True),
    StructField("end_time",         TimestampType(), True),
    StructField("duration_seconds", LongType(),      True),
    StructField("tasks",            StringType(),    True),
    StructField("error_message",    StringType(),    True),
    StructField("triggered_by",     StringType(),    True),
    StructField("run_url",          StringType(),    True),
])

runs_df = spark.createDataFrame(make_runs(jobs_local), schema=runs_schema)
runs_df.write.format("delta").mode("overwrite").saveAsTable(RUNS_TABLE)
print(f"Written {runs_df.count()} rows → {RUNS_TABLE}")

# COMMAND ----------
# MAGIC %md ## 5 — Verify

display(spark.sql(f"SELECT * FROM {JOBS_TABLE} LIMIT 5"))

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        j.job_id, j.job_name, j.job_type,
        COUNT(r.run_id)                                          AS total_runs,
        SUM(CASE WHEN r.status = 'SUCCESS'  THEN 1 ELSE 0 END)  AS success,
        SUM(CASE WHEN r.status = 'FAILED'   THEN 1 ELSE 0 END)  AS failed,
        SUM(CASE WHEN r.status = 'TIMEDOUT' THEN 1 ELSE 0 END)  AS timed_out,
        SUM(CASE WHEN r.status = 'CANCELLED'THEN 1 ELSE 0 END)  AS cancelled,
        ROUND(AVG(r.duration_seconds), 0)                        AS avg_duration_sec,
        MAX(r.start_time)                                        AS last_run
    FROM {JOBS_TABLE} j
    LEFT JOIN {RUNS_TABLE} r USING (job_id)
    GROUP BY j.job_id, j.job_name, j.job_type
    ORDER BY j.job_id
"""))
