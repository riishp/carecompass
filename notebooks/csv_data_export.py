# Databricks notebook source
# DBTITLE 1,Export Virtue tables to CSV
# Export all 3 tables from the Virtue Foundation dataset catalog to single CSV files
# Output location: /Volumes/workspace/datathon_2026/csv_data/

catalog = "databricks_virtue_foundation_dataset_dais_2026"
schema = "virtue_foundation_dataset"
out = "/Volumes/workspace/datathon_2026/csv_data"
tables = [
    "facilities",
    "india_post_pincode_directory",
    "nfhs_5_district_health_indicators",
]

for t in tables:
    path = f"{out}/{t}.csv"
    df = spark.table(f"{catalog}.{schema}.{t}")
    # coalesce(1) -> one part file, written to a temp dir then renamed to <table>.csv
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(path + "_tmp")
    part = [f.name for f in dbutils.fs.ls(path + "_tmp") if f.name.startswith("part-")][0]
    dbutils.fs.cp(path + "_tmp/" + part, path)
    dbutils.fs.rm(path + "_tmp", recurse=True)

    # verify: read back and compare row/column counts with the source table
    csv = spark.read.option("header", True).csv(path)
    ok = csv.count() == df.count() and csv.columns == df.columns
    print(f"{t}: {df.count()} rows x {len(df.columns)} cols -> {path} | verified: {ok}")