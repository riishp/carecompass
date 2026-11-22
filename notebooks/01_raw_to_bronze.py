# Databricks notebook source
# MAGIC %md
# MAGIC # Data Cleaning: Raw → Bronze (facilities)
# MAGIC
# MAGIC **Input (read-only):** `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities` + `india_post_pincode_directory`
# MAGIC
# MAGIC **Outputs (new tables, source is never modified):**
# MAGIC - `workspace.datathon_2026.facilities_raw_copy` / `pincode_directory_raw_copy` — untouched copies of the source; all work reads from these
# MAGIC - `workspace.datathon_2026.pincode_reference` — one clean row per pincode (state + median lat/long)
# MAGIC - `workspace.datathon_2026.facilities_bronze` — cleaned facilities, one row per facility
# MAGIC - `workspace.datathon_2026.facilities_quarantine` — rows too broken to trust (column-shifted), kept for review, not deleted
# MAGIC - `workspace.datathon_2026.cleaning_audit` — one log row per rule: how many rows each rule touched
# MAGIC
# MAGIC **Rules come from `02_Facilities_EDA` plus follow-up checks:**
# MAGIC 1. Clean the pincode reference table first (it has impossible coordinates like lat 620)
# MAGIC 2. Turn fake-missing values (`"null"`, `"nan"`, `"[]"`, `"-"`, …) into real NULLs
# MAGIC 3. Remove exact duplicate rows (same `unique_id`) and same-name-same-address duplicates
# MAGIC 4. Quarantine column-shifted rows (facility type holds coordinates/URLs/IDs, city holds a date, etc.)
# MAGIC 5. Fix pincodes (`"201 301"` → `"201301"`), check they exist in India Post
# MAGIC 6. Standardize state names to the 36 official states/UTs, using the **pincode's state** as the source of truth
# MAGIC 7. Validate coordinates (India bounds + distance from the pincode) and give each facility a `geo_confidence`
# MAGIC 8. Convert `capacity` / `numberDoctors` to integers and flag implausible doctor counts
# MAGIC 9. Parse list columns (specialties, procedure, equipment, …) from JSON text into real arrays
# MAGIC 10. Drop columns that are >95% empty
# MAGIC
# MAGIC **How we know it's clean:** the last section runs a validation suite. Every check must say PASS, and the row counts must reconcile (raw = bronze + quarantine + removed duplicates). If any check fails, the notebook stops with an error.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup
# MAGIC Settings, output table names, and **copies of the source tables** (`facilities_raw_copy`, `pincode_directory_raw_copy`). Everything below reads from the copies; the shared originals are never modified.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from datetime import datetime, timezone

SRC = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
FAC_SRC = f"{SRC}.facilities"
PIN_SRC = f"{SRC}.india_post_pincode_directory"

TGT = "workspace.datathon_2026"
PIN_REF = f"{TGT}.pincode_reference"
BRONZE = f"{TGT}.facilities_bronze"
QUARANTINE = f"{TGT}.facilities_quarantine"
AUDIT = f"{TGT}.cleaning_audit"
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {TGT}")

# India bounding box (lat up to 37.5 so northern Ladakh is included)
LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 6.0, 37.5, 68.0, 98.0
# A facility more than this far from its pincode's centre counts as a location conflict
GEO_TOL_KM = 25

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
audit_rows = []  # (step, rule, rows_affected, note)

def log(step, rule, n, note=""):
    audit_rows.append((RUN_ID, step, rule, int(n), note))
    print(f"[{step}] {rule}: {n:,} {note}")

# Work on COPIES: the shared source tables are only ever read, never written.
FAC_COPY = f"{TGT}.facilities_raw_copy"
PIN_COPY = f"{TGT}.pincode_directory_raw_copy"
spark.sql(f"CREATE OR REPLACE TABLE {FAC_COPY} AS SELECT * FROM {FAC_SRC}")
spark.sql(f"CREATE OR REPLACE TABLE {PIN_COPY} AS SELECT * FROM {PIN_SRC}")

raw = spark.read.table(FAC_COPY)
RAW_COUNT = raw.count()
assert RAW_COUNT == spark.read.table(FAC_SRC).count(), "copy does not match source"
print(f"Raw facilities rows (copy): {RAW_COUNT:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Canonical state list + name normalizer
# MAGIC Any spelling ("Tamilnadu", "TAMIL NADU", "Jammu & Kashmir", "Orissa") is reduced to a key of lowercase letters only, then matched to one of the 36 official names.

# COMMAND ----------

CANON_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan",
    "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]
ALIASES = {
    "orissa": "Odisha", "pondicherry": "Puducherry", "uttaranchal": "Uttarakhand",
    "nctofdelhi": "Delhi", "newdelhi": "Delhi", "delhinct": "Delhi", "nationalcapitalterritoryofdelhi": "Delhi",
    "andamannicobar": "Andaman and Nicobar Islands", "andamanandnicobar": "Andaman and Nicobar Islands",
    "dadraandnagarhaveli": "Dadra and Nagar Haveli and Daman and Diu",
    "damananddiu": "Dadra and Nagar Haveli and Daman and Diu",
    "jammukashmir": "Jammu and Kashmir", "jk": "Jammu and Kashmir",
    "tamilnadu": "Tamil Nadu", "chattisgarh": "Chhattisgarh", "telengana": "Telangana",
}

def state_key(col):
    """lowercase letters only, '&' -> 'and', leading 'the' removed"""
    c = F.lower(F.regexp_replace(col, "&", "and"))
    c = F.regexp_replace(c, "[^a-z]", "")
    return F.regexp_replace(c, "^the", "")

_key = lambda s: "".join(ch for ch in s.lower().replace("&", "and") if ch.isalpha()).removeprefix("the")
state_map_rows = {_key(s): s for s in CANON_STATES}
state_map_rows.update(ALIASES)
state_map = spark.createDataFrame(list(state_map_rows.items()), "skey string, state_canon string")
print(f"{len(CANON_STATES)} canonical states/UTs, {state_map.count()} accepted spellings")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Clean the pincode reference table
# MAGIC The India Post table has offices with impossible coordinates (e.g. lat 620, long 2219). We drop those coordinates, then keep **one row per pincode**: the most common state and the **median** lat/long (the median ignores a few bad offices; an average does not).

# COMMAND ----------

pin_raw = spark.read.table(PIN_COPY)
cols = {c.lower(): c for c in pin_raw.columns}
print("Pincode directory columns:", pin_raw.columns)

pin = (
    pin_raw.select(
        F.regexp_replace(F.col(cols["pincode"]).cast("string"), r"\D", "").alias("pincode"),
        F.col(cols["statename"]).alias("statename_raw"),
        F.expr(f"try_cast(`{cols['latitude']}` as double)").alias("lat"),
        F.expr(f"try_cast(`{cols['longitude']}` as double)").alias("lon"),
    )
    .filter(F.col("pincode").rlike(r"^[1-9][0-9]{5}$"))
    .withColumn("skey", state_key(F.col("statename_raw")))
    .join(F.broadcast(state_map), "skey", "left")
)

bad_office_coords = pin.filter(
    F.col("lat").isNotNull() & ~(F.col("lat").between(LAT_MIN, LAT_MAX) & F.col("lon").between(LON_MIN, LON_MAX))
).count()
log("2", "pincode-directory offices with impossible coordinates (set to NULL)", bad_office_coords)
unmapped_pin_states = pin.filter(F.col("state_canon").isNull()).select("statename_raw").distinct().collect()
if unmapped_pin_states:
    print("WARNING unmapped directory state names:", [r[0] for r in unmapped_pin_states])

pin = pin.withColumn(
    "lat_ok", F.when(F.col("lat").between(LAT_MIN, LAT_MAX) & F.col("lon").between(LON_MIN, LON_MAX), F.col("lat"))
).withColumn(
    "lon_ok", F.when(F.col("lat").between(LAT_MIN, LAT_MAX) & F.col("lon").between(LON_MIN, LON_MAX), F.col("lon"))
)

pin_ref = pin.groupBy("pincode").agg(
    F.expr("mode(state_canon)").alias("pin_state"),
    F.countDistinct("state_canon").alias("pin_n_states"),
    F.expr("percentile_approx(lat_ok, 0.5)").alias("pin_lat"),
    F.expr("percentile_approx(lon_ok, 0.5)").alias("pin_lon"),
    F.count("*").alias("pin_n_offices"),
)
pin_ref.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(PIN_REF)
pin_ref = spark.read.table(PIN_REF)
log("2", "clean pincode reference rows written", pin_ref.count(), PIN_REF)
log("2", "pincodes with no usable coordinates", pin_ref.filter("pin_lat IS NULL").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fake-missing values → real NULL
# MAGIC `"null"`, `"nan"`, `"none"`, `"n/a"`, `"na"`, `"-"`, `""`, `"[]"`, `'[""]'`, `"{}"` all become NULL in every text column, and surrounding spaces are trimmed.

# COMMAND ----------

MISSING_TOKENS = ["", "null", "nan", "none", "n/a", "na", "-", "[]", '[""]', "[null]", "{}"]
str_cols = [f.name for f in raw.schema.fields if f.dataType.simpleString() == "string"]

fake_missing_before = raw.select(
    sum(F.when(F.lower(F.trim(F.col(c))).isin(MISSING_TOKENS), 1).otherwise(0) for c in str_cols).alias("n")
).first()["n"]

df = raw
for c in str_cols:
    t = F.trim(F.col(c))
    df = df.withColumn(c, F.when(F.lower(t).isin(MISSING_TOKENS), None).otherwise(t))
log("3", "fake-missing cells converted to NULL", fake_missing_before)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Remove duplicates
# MAGIC - **Exact duplicates:** 11 `unique_id`s appear twice. Keep the copy with the most filled-in fields.
# MAGIC - **Same facility, different ID:** same normalized name + address line 1. Keep the most complete one.

# COMMAND ----------

filled = sum(F.when(F.col(c).isNotNull(), 1).otherwise(0) for c in df.columns)
df = df.withColumn("_filled", filled)

w_id = Window.partitionBy("unique_id").orderBy(F.desc("_filled"))
before = df.count()
df = df.withColumn("_rn", F.row_number().over(w_id)).filter("_rn = 1").drop("_rn")
dup_id_removed = before - df.count()
log("4", "duplicate unique_id rows removed", dup_id_removed)

df = (
    df.withColumn("_name_key", F.lower(F.regexp_replace(F.col("name"), r"[^A-Za-z0-9]", "")))
      .withColumn("_addr_key", F.lower(F.regexp_replace(F.col("address_line1"), r"[^A-Za-z0-9]", "")))
)
w_na = Window.partitionBy("_name_key", "_addr_key").orderBy(F.desc("_filled"), "unique_id")
has_key = F.col("_name_key").isNotNull() & F.col("_addr_key").isNotNull() & (F.col("_name_key") != "") & (F.col("_addr_key") != "")
before = df.count()
df = df.withColumn("_rn", F.when(has_key, F.row_number().over(w_na)).otherwise(1)).filter("_rn = 1").drop("_rn")
name_addr_removed = before - df.count()
log("4", "same name+address duplicates removed (different IDs)", name_addr_removed)
DUPES_REMOVED = dup_id_removed + name_addr_removed

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Quarantine column-shifted / unusable rows
# MAGIC A row is quarantined (moved to `facilities_quarantine`, not deleted) if:
# MAGIC - `facilityTypeId` is not a real type (it holds coordinates, URLs, hashes, procedure names…)
# MAGIC - `address_city` looks like a date or a number
# MAGIC - `name` is missing or longer than 150 characters (description text landed in the name column)
# MAGIC
# MAGIC Typos are fixed first: `farmacy` → `pharmacy`. Missing facility type becomes `unknown` (kept, not quarantined).

# COMMAND ----------

VALID_TYPES = ["hospital", "clinic", "dentist", "doctor", "pharmacy", "nursing_home", "unknown"]

df = df.withColumn("facility_type", F.lower(F.col("facilityTypeId")))
df = df.withColumn(
    "facility_type",
    F.when(F.col("facility_type") == "farmacy", "pharmacy")
     .when(F.col("facility_type").isNull(), "unknown")
     .otherwise(F.col("facility_type")),
)
log("5", "farmacy -> pharmacy", df.filter(F.lower("facilityTypeId") == "farmacy").count())
log("5", "missing facility type -> unknown", df.filter(F.col("facilityTypeId").isNull()).count())

df = df.withColumn(
    "quarantine_reason",
    F.when(~F.col("facility_type").isin(VALID_TYPES), "facilityTypeId is not a facility type (column shift)")
     .when(F.col("address_city").rlike(r"^\s*[0-9\-/.: ]+\s*$"), "address_city is a date/number (column shift)")
     .when(F.col("name").isNull(), "name missing")
     .when(F.length("name") > 150, "name is long free text (column shift)"),
)
quarantine = df.filter(F.col("quarantine_reason").isNotNull())
df = df.filter(F.col("quarantine_reason").isNull()).drop("quarantine_reason")
for r in quarantine.groupBy("quarantine_reason").count().collect():
    log("5", f"quarantined: {r['quarantine_reason']}", r["count"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Pincodes
# MAGIC Remove spaces and other non-digits (`"201 301"` → `"201301"`). A valid Indian pincode is 6 digits and doesn't start with 0. Then check it exists in the clean India Post reference.

# COMMAND ----------

df = df.withColumn("pincode_raw", F.col("address_zipOrPostcode"))
df = df.withColumn("_pin_digits", F.regexp_replace("address_zipOrPostcode", r"\D", ""))
df = df.withColumn("pincode", F.when(F.col("_pin_digits").rlike(r"^[1-9][0-9]{5}$"), F.col("_pin_digits"))).drop("_pin_digits")

log("6", "pincodes repaired by removing spaces/punctuation",
    df.filter(F.col("pincode").isNotNull() & (F.col("pincode") != F.col("pincode_raw"))).count())
log("6", "pincodes invalid after repair (set to NULL)",
    df.filter(F.col("pincode").isNull() & F.col("pincode_raw").isNotNull()).count())

df = df.join(F.broadcast(pin_ref), "pincode", "left")
df = df.withColumn("pincode_in_directory", F.col("pin_n_offices").isNotNull())
log("6", "valid-format pincodes NOT found in India Post directory",
    df.filter(F.col("pincode").isNotNull() & ~F.col("pincode_in_directory")).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. States
# MAGIC `state` = the **pincode's state** when the pincode is known (most reliable), otherwise the stated state mapped to the official list. `state_conflict = true` when the two disagree, e.g. a facility says Andhra Pradesh but its pincode is in Telangana.

# COMMAND ----------

df = df.withColumn("state_raw", F.col("address_stateOrRegion"))
df = df.withColumn("skey", state_key(F.col("state_raw"))).join(
    F.broadcast(state_map.withColumnRenamed("state_canon", "state_from_text")), "skey", "left"
).drop("skey")

df = (
    df.withColumn("state", F.coalesce("pin_state", "state_from_text"))
      .withColumn("state_source", F.when(F.col("pin_state").isNotNull(), "pincode")
                                   .when(F.col("state_from_text").isNotNull(), "stated")
                                   .otherwise("unknown"))
      .withColumn("state_conflict", F.col("pin_state").isNotNull() & F.col("state_from_text").isNotNull()
                                    & (F.col("pin_state") != F.col("state_from_text")))
)
log("7", "distinct raw state values", df.select("state_raw").distinct().count())
log("7", "state taken from pincode", df.filter("state_source = 'pincode'").count())
log("7", "state conflicts (stated vs pincode)", df.filter("state_conflict").count())
log("7", "state unknown", df.filter("state IS NULL").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Coordinates → `geo_confidence`
# MAGIC | geo_confidence | meaning | what the hospital finder should do |
# MAGIC |---|---|---|
# MAGIC | `high` | coords in India **and** within 25 km of the pincode centre | trust distance |
# MAGIC | `medium` | coords in India, pincode missing/unknown so can't cross-check | use, small penalty |
# MAGIC | `low` | coords in India but **>25 km** from the pincode centre (conflict) | use with warning / "call ahead" |
# MAGIC | `imputed` | coords missing or outside India → replaced with pincode centre | approximate distance only |
# MAGIC | `no_location` | no usable coords and no pincode | can't route to it |
# MAGIC
# MAGIC Original coordinates are kept in `latitude_raw` / `longitude_raw`.

# COMMAND ----------

def haversine_km(lat1, lon1, lat2, lon2):
    return 6371.0 * 2 * F.asin(F.sqrt(
        F.pow(F.sin(F.radians(lat2 - lat1) / 2), 2)
        + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.pow(F.sin(F.radians(lon2 - lon1) / 2), 2)
    ))

in_india = F.col("latitude_raw").between(LAT_MIN, LAT_MAX) & F.col("longitude_raw").between(LON_MIN, LON_MAX)
df = (
    df.withColumn("latitude_raw", F.col("latitude").cast("double"))
      .withColumn("longitude_raw", F.col("longitude").cast("double"))
      .withColumn("_coords_ok", F.coalesce(in_india, F.lit(False)))
      .withColumn("km_from_pincode", F.when(F.col("_coords_ok") & F.col("pin_lat").isNotNull(),
                  F.round(haversine_km(F.col("latitude_raw"), F.col("longitude_raw"), F.col("pin_lat"), F.col("pin_lon")), 2)))
      .withColumn("geo_confidence",
          F.when(F.col("_coords_ok") & (F.col("km_from_pincode") <= GEO_TOL_KM), "high")
           .when(F.col("_coords_ok") & F.col("km_from_pincode").isNull(), "medium")
           .when(F.col("_coords_ok"), "low")
           .when(F.col("pin_lat").isNotNull(), "imputed")
           .otherwise("no_location"))
      .withColumn("latitude", F.when(F.col("_coords_ok"), F.col("latitude_raw")).otherwise(F.col("pin_lat")))
      .withColumn("longitude", F.when(F.col("_coords_ok"), F.col("longitude_raw")).otherwise(F.col("pin_lon")))
      .drop("_coords_ok")
)
for r in df.groupBy("geo_confidence").count().orderBy("geo_confidence").collect():
    log("8", f"geo_confidence = {r['geo_confidence']}", r["count"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Numbers: capacity and doctors
# MAGIC Converted to integers (commas removed, anything non-numeric → NULL). `numberDoctors` often counts only the doctors named on the website (e.g. KGMU Lucknow: 1 doctor, 4,000 beds), so `doctor_count_plausible = false` when a 100+ bed facility lists ≤2 doctors. **Don't use the doctor count to decide what a hospital can do.**

# COMMAND ----------

def to_int(c):
    return F.expr(f"try_cast(regexp_replace(`{c}`, '[, ]', '') as int)")

df = (
    df.withColumn("capacity_beds", to_int("capacity"))
      .withColumn("number_doctors", to_int("numberDoctors"))
      .withColumn("year_established", to_int("yearEstablished"))
)
df = (
    df.withColumn("capacity_beds", F.when(F.col("capacity_beds").between(1, 20000), F.col("capacity_beds")))
      .withColumn("number_doctors", F.when(F.col("number_doctors").between(1, 20000), F.col("number_doctors")))
      .withColumn("year_established", F.when(F.col("year_established").between(1800, datetime.now().year), F.col("year_established")))
      .withColumn("doctor_count_plausible",
          ~((F.col("capacity_beds") >= 100) & (F.col("number_doctors") <= 2)) | F.col("capacity_beds").isNull() | F.col("number_doctors").isNull())
)
log("9", "rows with usable capacity_beds", df.filter("capacity_beds IS NOT NULL").count())
log("9", "rows with usable number_doctors", df.filter("number_doctors IS NOT NULL").count())
log("9", "implausible doctor counts flagged", df.filter("NOT doctor_count_plausible").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. List columns → real arrays
# MAGIC Text like `'["cardiology","neurology",null]'` becomes an array `["cardiology","neurology"]`. If the text isn't JSON, it's split on commas. Empty lists become NULL. A count column (`n_specialties`, …) is added for each.

# COMMAND ----------

LIST_COLS = ["specialties", "procedure", "equipment", "capability", "phone_numbers", "websites",
             "source_urls", "source_types", "affiliationTypeIds"]
LIST_COLS = [c for c in LIST_COLS if c in df.columns]

for c in LIST_COLS:
    parsed = F.from_json(F.col(c), "array<string>")
    fallback = F.split(F.regexp_replace(F.col(c), r'[\[\]"]', ""), r"\s*,\s*")
    arr = F.when(F.col(c).startswith("["), parsed).otherwise(fallback)
    arr = F.filter(arr, lambda x: x.isNotNull() & (F.trim(x) != "") & (F.lower(F.trim(x)) != "null"))
    arr = F.array_distinct(F.transform(arr, lambda x: F.trim(x)))
    df = df.withColumn(c, F.when(F.size(arr) > 0, arr))
    df = df.withColumn(f"n_{c}", F.when(F.col(c).isNull(), F.lit(0)).otherwise(F.size(c)))
    log("10", f"{c}: rows with at least one value", df.filter(F.col(c).isNotNull()).count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Drop near-empty columns, write Bronze + Quarantine + Audit

# COMMAND ----------

n_rows = df.count()
null_share = df.select([(F.sum(F.col(c).isNull().cast("int")) / n_rows).alias(c) for c in df.columns]).first().asDict()
DROP_COLS = [c for c, s in null_share.items() if s > 0.95 and c not in ("latitude", "longitude", "pincode", "state")]
log("11", "columns dropped (>95% empty)", len(DROP_COLS), ", ".join(DROP_COLS))

HELPER_COLS = ["_filled", "_name_key", "_addr_key", "address_stateOrRegion", "address_zipOrPostcode",
               "facilityTypeId", "capacity", "numberDoctors", "yearEstablished",
               "pin_n_states", "pin_n_offices", "pin_lat", "pin_lon"]
bronze = df.drop(*[c for c in DROP_COLS + HELPER_COLS if c in df.columns])
bronze = bronze.withColumn("cleaned_run_id", F.lit(RUN_ID))

# Second pass: catch fake-missing tokens in any text column (incl. ones created after step 3)
scols = [f.name for f in bronze.schema.fields if f.dataType.simpleString() == "string"]
counts = bronze.select([F.sum(F.lower(F.trim(F.col(c))).isin(MISSING_TOKENS).cast("int")).alias(c) for c in scols]).first().asDict()
leftover = {c: n for c, n in counts.items() if n}
for c in leftover:
    bronze = bronze.withColumn(c, F.when(F.lower(F.trim(F.col(c))).isin(MISSING_TOKENS), None).otherwise(F.col(c)))
log("11", "leftover fake-missing cells fixed in second pass", sum(leftover.values()), str(leftover))

bronze.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(BRONZE)
(quarantine.drop(*[c for c in ["_filled", "_name_key", "_addr_key"] if c in quarantine.columns])
    .withColumn("cleaned_run_id", F.lit(RUN_ID))
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(QUARANTINE))

log("11", "bronze rows written", spark.read.table(BRONZE).count(), BRONZE)
log("11", "quarantine rows written", spark.read.table(QUARANTINE).count(), QUARANTINE)

audit_df = spark.createDataFrame(audit_rows, "run_id string, step string, rule string, rows_affected long, note string") \
                .withColumn("logged_at", F.current_timestamp())
audit_df.write.mode("append").option("mergeSchema", "true").saveAsTable(AUDIT)
display(audit_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Validation — how we know the data is clean
# MAGIC Each check is a rule the Bronze table must satisfy. **All must PASS**; if any fail, this cell raises an error so a broken table can't be used silently.

# COMMAND ----------

b = spark.read.table(BRONZE)
q = spark.read.table(QUARANTINE)
B, Q = b.count(), q.count()
checks = []

def check(name, passed, detail=""):
    checks.append((name, "PASS" if passed else "FAIL", str(detail)))

# Row accounting: nothing silently lost
check("Row reconciliation: raw = bronze + quarantine + duplicates removed",
      RAW_COUNT == B + Q + DUPES_REMOVED, f"{RAW_COUNT} = {B} + {Q} + {DUPES_REMOVED}")

# Uniqueness
check("unique_id is unique", b.select("unique_id").distinct().count() == B,
      f"{B - b.select('unique_id').distinct().count()} duplicate ids")

# No fake-missing strings left
bstr = [f.name for f in b.schema.fields if f.dataType.simpleString() == "string"]
fake_by_col = b.select([F.sum(F.lower(F.trim(F.col(c))).isin(MISSING_TOKENS).cast("int")).alias(c) for c in bstr]).first().asDict()
fake_by_col = {c: n for c, n in fake_by_col.items() if n}
check("No 'null' / 'nan' / '[]' strings left", not fake_by_col, f"{sum(fake_by_col.values())} cells {fake_by_col}")

# States
bad_states = b.filter(F.col("state").isNotNull() & ~F.col("state").isin(CANON_STATES)).count()
n_states = b.select("state").distinct().filter("state IS NOT NULL").count()
check("Every state is one of the 36 official states/UTs", bad_states == 0, f"{n_states} distinct states, {bad_states} invalid")

# Pincodes
bad_pins = b.filter(F.col("pincode").isNotNull() & ~F.col("pincode").rlike(r"^[1-9][0-9]{5}$")).count()
check("Every pincode is 6 digits (or NULL)", bad_pins == 0, f"{bad_pins} invalid")

# Coordinates
bad_geo = b.filter(F.col("latitude").isNotNull() & ~(F.col("latitude").between(LAT_MIN, LAT_MAX) & F.col("longitude").between(LON_MIN, LON_MAX))).count()
check("Every coordinate is inside India (or NULL)", bad_geo == 0, f"{bad_geo} outside")
no_conf = b.filter("geo_confidence IS NULL").count()
check("Every row has a geo_confidence label", no_conf == 0, f"{no_conf} missing")
geo_none_with_coords = b.filter("geo_confidence = 'no_location' AND latitude IS NOT NULL").count()
check("geo_confidence agrees with coordinates", geo_none_with_coords == 0, f"{geo_none_with_coords} mismatched")

# Types
bad_types = b.filter(~F.col("facility_type").isin(VALID_TYPES)).count()
check("Every facility_type is a valid type", bad_types == 0, f"{bad_types} invalid")
check("capacity_beds / number_doctors are integers",
      dict(b.dtypes)["capacity_beds"] == "int" and dict(b.dtypes)["number_doctors"] == "int",
      f"{dict(b.dtypes)['capacity_beds']}, {dict(b.dtypes)['number_doctors']}")
neg = b.filter("capacity_beds <= 0 OR number_doctors <= 0").count()
check("No zero/negative capacity or doctor counts", neg == 0, f"{neg} rows")

# Lists
not_arrays = [c for c in LIST_COLS if c in b.columns and not dict(b.dtypes)[c].startswith("array")]
check("List columns are real arrays", not not_arrays, ", ".join(not_arrays) or "all arrays")

# Quarantine really holds broken rows only
q_bad = q.filter(F.col("quarantine_reason").isNull()).count()
check("Every quarantined row has a reason", q_bad == 0, f"{q_bad} without reason")

result = spark.createDataFrame(checks, "check string, result string, detail string")
display(result)
failed = [c for c in checks if c[1] == "FAIL"]
if failed:
    raise AssertionError(f"{len(failed)} validation check(s) FAILED: " + "; ".join(c[0] for c in failed))
print(f"All {len(checks)} validation checks PASSED.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 13. Before vs after + spot checks
# MAGIC Numbers to compare against `02_Facilities_EDA`, plus known-bad rows we can eyeball.

# COMMAND ----------

before_after = spark.createDataFrame([
    ("Rows", RAW_COUNT, B),
    ("Distinct state values", raw.select("address_stateOrRegion").distinct().count(), n_states),
    ("Duplicate unique_ids", RAW_COUNT - raw.select("unique_id").distinct().count(), B - b.select("unique_id").distinct().count()),
    ("Coordinates outside India", raw.filter(~(F.col("latitude").between(LAT_MIN, LAT_MAX) & F.col("longitude").between(LON_MIN, LON_MAX))).count(), bad_geo),
    ("Badly formatted pincodes", raw.filter(F.col("address_zipOrPostcode").isNotNull() & ~F.col("address_zipOrPostcode").rlike(r"^[0-9]{6}$")).count(),
         bad_pins),
    ("Invalid facility types", raw.filter(~F.lower("facilityTypeId").isin(VALID_TYPES + ["farmacy"]) & F.col("facilityTypeId").isNotNull()).count(), bad_types),
], "metric string, raw long, bronze long")
display(before_after)

# Known-bad rows from the EDA: what happened to them?
spot = ["Aravind Eye Hospital", "Krishna Hospital Multispeciality", "Sanjivani Multi Speciality Hospital",
        "Sumitra Hospital", "King George's Medical University, Lucknow", "Navjeevan Nursing Home"]
display(
    b.filter(F.col("name").isin(spot))
     .select("name", "address_city", "state_raw", "state", "state_source", "state_conflict", "pincode_raw", "pincode",
             "latitude_raw", "longitude_raw", "latitude", "longitude", "km_from_pincode", "geo_confidence",
             "capacity_beds", "number_doctors", "doctor_count_plausible")
)
display(q.groupBy("quarantine_reason").count())