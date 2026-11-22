# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 09b — Gold, verified against the government hospital directory
# MAGIC
# MAGIC **Input:** `/Volumes/workspace/datathon_2026/csv_data/gold_facility_search_directory_verified.csv`. This is `gold_facility_search` after a teammate matched every facility to the government hospital directory: `directory_match` = STRONG / PROBABLE / WEAK / NONE.
# MAGIC Only **STRONG** matches were applied. For those, coordinates and/or PIN were replaced by the directory's (`location_source = 'directory'`, 633 facilities). The original values are kept in `orig_*`.
# MAGIC
# MAGIC **Output (CREATE OR REPLACE):** `workspace.datathon_2026.gold_facility_search_verified`. It has the same columns as `gold_facility_search`, with real types and arrays, plus the directory columns. **The recommendation engine reads this table.**
# MAGIC
# MAGIC **Rules**
# MAGIC 1. List columns are stored in the CSV as `a|b|c` and are split back into arrays. Numbers are cast to real types.
# MAGIC 2. **Team decision:** government directory values are ground truth where present.
# MAGIC    - Facilities whose location came from the directory (`directory`), or already matched it (`original (matches directory)`), become `location_confidence = 'high'`.
# MAGIC    - They get a government-directory location note.
# MAGIC    - Warnings about the old location (vote / merged-record location flags) are cleared.
# MAGIC
# MAGIC    The CSV had kept the pre-verification confidence and note for these rows.
# MAGIC 3. `location_vote` is set to `government_directory` for those rows, so the source is visible everywhere.
# MAGIC 4. **Hand-reviewed matches marked unverified** (`UNVERIFIED_MATCHES`). Their directory changes are undone: the original coordinates and PIN are restored, `directory_match = 'UNVERIFIED'`, confidence becomes low, and the note says to confirm by phone.
# MAGIC
# MAGIC Validation at the end: all checks must pass.

# COMMAND ----------

# DBTITLE 1,0. Setup
from datetime import datetime, timezone
from pyspark.sql import functions as F

TGT = "workspace.datathon_2026"
SRC_CSV = "/Volumes/workspace/datathon_2026/csv_data/gold_facility_search_directory_verified.csv"
GOLD = f"{TGT}.gold_facility_search"
OUT = f"{TGT}.gold_facility_search_verified"
AUDIT = f"{TGT}.cleaning_audit"
RUN_ID = "verified-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
audit_rows = []
def log(step, rule, n, note=""):
    audit_rows.append((RUN_ID, f"verified-{step}", rule, int(n), note))
    print(f"[{step}] {rule}: {n:,} {note}")

ARRAY_COLS = ["location_warnings", "specialty_codes", "specialty_labels", "specialties_other", "procedures",
              "equipment", "capabilities", "phone_numbers"]
DOUBLE_COLS = ["latitude", "longitude", "location_error_km", "dir_name_score", "km_moved", "orig_latitude", "orig_longitude"]
INT_COLS = ["n_specialty_codes", "capacity_beds", "number_doctors", "year_established"]
OLD_LOCATION_WARNINGS = ["moved_to_pin_area", "pin_disagrees_with_coords", "location_unresolved", "coords_disagree"]

raw = (spark.read.option("header", True).option("multiLine", True).option("escape", '"')
       .csv(SRC_CSV))
RAW_N = raw.count()
log(0, "rows in verified CSV", RAW_N)

# COMMAND ----------

# DBTITLE 1,1. Types and arrays
def arr(c):
    return F.when(F.trim(F.col(c)) == "", F.array().cast("array<string>")).otherwise(F.split(F.col(c), r"\|"))

df = raw
for c in raw.columns:                                     # empty strings → NULL for scalar columns
    if c not in ARRAY_COLS:
        df = df.withColumn(c, F.when(F.trim(F.col(c)) == "", None).otherwise(F.col(c)))
for c in ARRAY_COLS:
    df = df.withColumn(c, F.coalesce(arr(c), F.array().cast("array<string>")))
for c in DOUBLE_COLS:
    df = df.withColumn(c, F.col(c).cast("double"))
for c in INT_COLS:
    df = df.withColumn(c, F.col(c).cast("double").cast("int"))

# COMMAND ----------

# DBTITLE 1,1b. Hand-reviewed directory matches marked unverified
# Reviewed 2026-11-22: the directory match looks wrong (different name / different city), so it is not applied.
UNVERIFIED_MATCHES = {
    "c1897c75-95fe-4983-92b0-a81efe22686a": "Joshi Hospital (city Pune) matched to Joshi Hospital, Jalna; moved 1,008 km",
    "975610da-5fbf-4c18-963e-d652382cce87": "OM Clinical Laboratory matched to a differently named lab (Voras Clinical Laboratory)",
}
um = F.col("unique_id").isin(list(UNVERIFIED_MATCHES))
df = (df.withColumn("latitude", F.when(um & F.col("orig_latitude").isNotNull(), F.col("orig_latitude")).otherwise(F.col("latitude")))
        .withColumn("longitude", F.when(um & F.col("orig_longitude").isNotNull(), F.col("orig_longitude")).otherwise(F.col("longitude")))
        .withColumn("pincode", F.when(um & F.col("orig_pin").isNotNull(), F.col("orig_pin")).otherwise(F.col("pincode")))
        .withColumn("state", F.when(um & F.col("orig_state").isNotNull(), F.col("orig_state")).otherwise(F.col("state")))
        .withColumn("directory_match", F.when(um, F.lit("UNVERIFIED")).otherwise(F.col("directory_match")))
        .withColumn("location_source", F.when(um, F.lit("original (directory match unverified)")).otherwise(F.col("location_source")))
        .withColumn("location_confidence", F.when(um, F.lit("low")).otherwise(F.col("location_confidence")))
        .withColumn("location_note", F.when(um, F.lit("Government directory match could not be verified (name or place does not "
                                                       "match), so the original location is shown. It may be wrong — confirm "
                                                       "the address by phone.")).otherwise(F.col("location_note")))
        .withColumn("verification_note", F.when(um, F.lit("Directory match marked UNVERIFIED by hand review; not applied."))
                                          .otherwise(F.col("verification_note"))))
log("1b", "directory matches marked unverified by hand review (not applied)", df.where(um).count())

# COMMAND ----------

# DBTITLE 1,2. Government directory = ground truth for location
gov = F.col("location_source").isin("directory", "original (matches directory)")
log(2, "facilities with a government-directory location", df.where(gov).count())
log(2, "… of which moved to the directory's coordinates", df.where("location_source = 'directory'").count())
big = df.where("location_source = 'directory' AND km_moved > 300").orderBy(F.desc("km_moved"))
log(2, "… moved more than 300 km (trusted per team rule; listed below for a sanity check)", big.count())
display(big.select("name", "city", "state", "directory_name", "directory_district", "km_moved"))

note = F.concat(F.lit("Location from the government hospital directory (matched '"), F.col("directory_name"),
                F.lit("'"), F.when(F.col("directory_district").isNotNull(),
                                   F.concat(F.lit(", "), F.col("directory_district"))).otherwise(F.lit("")),
                F.lit(")."))
df = (df.withColumn("location_confidence", F.when(gov, F.lit("high")).otherwise(F.col("location_confidence")))
        .withColumn("location_error_km", F.when(gov, F.lit(1.0)).otherwise(F.col("location_error_km")))
        .withColumn("location_note", F.when(gov, note).otherwise(F.col("location_note")))
        .withColumn("location_vote", F.when(gov, F.lit("government_directory")).otherwise(F.col("location_vote")))
        .withColumn("location_warnings", F.when(gov, F.filter("location_warnings", lambda w: ~w.isin(OLD_LOCATION_WARNINGS)))
                                          .otherwise(F.col("location_warnings")))
        .withColumn("verified_run_id", F.lit(RUN_ID)))
for r in df.groupBy("location_confidence").count().orderBy("location_confidence").collect():
    log(2, f"location_confidence = {r['location_confidence']}", r["count"])

df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(OUT)
OUT_N = spark.read.table(OUT).count()
log(2, "gold_facility_search_verified rows written", OUT_N, OUT)
(spark.createDataFrame(audit_rows, "run_id string, step string, rule string, rows_affected long, note string")
      .withColumn("logged_at", F.current_timestamp())
      .write.mode("append").option("mergeSchema", "true").saveAsTable(AUDIT))

# COMMAND ----------

# DBTITLE 1,3. Validation
v = spark.read.table(OUT)
gold_ids = spark.read.table(GOLD).select("unique_id")
checks = []
def check(name, ok, detail=""):
    checks.append((name, "PASS" if ok else "FAIL", str(detail)))

check("Same row count as the CSV", OUT_N == RAW_N, f"{OUT_N} vs {RAW_N}")
check("unique_id is unique", v.select("unique_id").distinct().count() == OUT_N)
missing = v.join(gold_ids, "unique_id", "left_anti").count()
check("Every facility exists in gold_facility_search", missing == 0, f"{missing} unknown ids")
bad = v.where("latitude IS NULL OR NOT (latitude BETWEEN 6 AND 38 AND longitude BETWEEN 68 AND 98)").count()
check("Every facility has coordinates inside India", bad == 0, f"{bad} rows")
bad = v.where(~F.col("location_confidence").isin("high", "medium", "low")).count()
check("location_confidence is high/medium/low", bad == 0, f"{bad} rows")
bad = v.where("location_source IN ('directory', 'original (matches directory)') AND location_confidence != 'high'").count()
check("Government-located facilities are high confidence", bad == 0, f"{bad} rows")
types = dict(v.dtypes)
bad = [c for c in ARRAY_COLS if not types[c].startswith("array")] + [c for c in DOUBLE_COLS if types[c] != "double"]
check("List columns are arrays and numbers are numeric", not bad, bad or "ok")
n_codes = v.where("size(specialty_codes) > 0").count()
bad = v.where(F.col("unique_id").isin(list(UNVERIFIED_MATCHES)) & (F.col("directory_match") != "UNVERIFIED")).count()
check("Hand-reviewed matches are marked UNVERIFIED and not applied", bad == 0 and v.where("directory_match = 'UNVERIFIED'").count() == len(UNVERIFIED_MATCHES), f"{bad} rows")
check("Specialty codes parsed (≥ 99% of facilities have some)", n_codes / OUT_N >= 0.99, f"{n_codes / OUT_N:.1%}")

display(spark.createDataFrame(checks, "check string, result string, detail string"))
failed = [c for c in checks if c[1] == "FAIL"]
if failed:
    raise AssertionError(f"{len(failed)} verified-Gold check(s) FAILED: " + "; ".join(c[0] for c in failed))
print(f"All {len(checks)} verified-Gold checks PASSED.")
