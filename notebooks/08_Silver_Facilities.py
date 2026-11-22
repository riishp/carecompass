# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 08 — Silver: one clean, combined row per facility
# MAGIC
# MAGIC **Pipeline position:** `Data Cleaning: Raw to Bronze` → `03_location_cleaning_and_uncertainty` → `06_NFHS_Cleaning` + `07_Specialty_Standardization` → **this notebook** → Gold.
# MAGIC
# MAGIC **Inputs (read-only here):**
# MAGIC | table | owner | used for |
# MAGIC |---|---|---|
# MAGIC | `facilities_bronze` | Raw → Bronze | which facilities exist; identity, pincode, state, types, numbers, lists |
# MAGIC | `facilities_located` | 03 Location | final coordinates, error radius, confidence, explanation, district |
# MAGIC | `specialty_map` | 07 Specialties | raw specialty string → standard code + label |
# MAGIC | `nfhs_district_crosswalk` | 06 NFHS | facility district → NFHS district (for joining health indicators later) |
# MAGIC
# MAGIC **Output:** `workspace.datathon_2026.facilities_silver` — exactly one row per Bronze facility.
# MAGIC
# MAGIC **What Silver adds on top of Bronze + Location** (gaps found in the Bronze review and `03_Data_Quality_Audit`):
# MAGIC 1. **One location answer.** Location notebook's coordinates replace Bronze's `geo_confidence`; adds a plain `location_confidence` (high/medium/low) because `uncertainty_tier = 'low'` means *good*.
# MAGIC 2. **State fix.** Bronze always trusts the pincode's state; when the pincode is >200 km from the facility's coordinates the pincode itself is wrong, so the stated state is kept (e.g. Aravind Eye Hospital, Hyderabad was labeled Delhi).
# MAGIC 3b. **Location vote.** When a facility's coordinates are > 30 km from its PIN, the PIN centre and the city's centre break the tie. If PIN and city agree, the coordinates were wrong and the point moves to the PIN area. If coordinates and city agree, the PIN was wrong and the district follows the coordinates. If nothing agrees, the facility stays flagged `unresolved`.
# MAGIC 3. **Current city names** (Bangalore → Bengaluru, Gurgaon → Gurugram, …) and consistent capitalization; raw city kept.
# MAGIC 4. **Possible duplicates** (same name + city, different IDs) grouped and flagged — one primary record per group. Nothing deleted.
# MAGIC 5. **Junk phone numbers** — an official phone shared by many unrelated facilities is flagged and blanked in the clean column.
# MAGIC 6. **Implausible doctor counts** (> 2,000) set to NULL and flagged.
# MAGIC 7. **Operator type** unified (`government` → `public`); leftover non-India country values flagged.
# MAGIC 8. **Standard specialties**: `specialty_codes` / `specialty_labels` from the 07 lookup; unusual ones kept in `specialties_other`.
# MAGIC 9. **NFHS link**: `nfhs_state` + `nfhs_district` so health indicators can be joined in Gold; districts created after the survey link to their parent district (`nfhs_link_type`).
# MAGIC 10. **Junk specialty items** (JSON fragments from column-shifted rows) dropped.
# MAGIC
# MAGIC Every change is flagged, never silent. Section 11 runs validation checks; all must PASS.

# COMMAND ----------

# DBTITLE 1,0. Setup and input checks
from datetime import datetime, timezone
from pyspark.sql import functions as F, Window

TGT = "workspace.datathon_2026"
BRONZE = f"{TGT}.facilities_bronze"
LOCATED = f"{TGT}.facilities_located"
SPEC_MAP = f"{TGT}.specialty_map"
XWALK = f"{TGT}.nfhs_district_crosswalk"
PIN_CENTERS = f"{TGT}.pin_centers"   # 03_geo
GEO_LOOKUP = f"{TGT}.geo_lookup"     # 03_geo
PIN_LOOKUP = f"{TGT}.pin_lookup"     # 03 Location (its district names match the NFHS crosswalk)
SILVER = f"{TGT}.facilities_silver"
AUDIT = f"{TGT}.cleaning_audit"

VOTE_AGREE_KM = 30               # two location clues "agree" when their points are within this distance
STATE_OVERRIDE_KM = 200          # pincode this far from the coordinates => don't trust the pincode's state
MAX_PLAUSIBLE_DOCTORS = 2000
SHARED_PHONE_MIN_FACILITIES = 5  # same official phone on >= 5 facilities in >= 2 cities => directory/aggregator number

for t, who in [(BRONZE, "Data Cleaning: Raw to Bronze"), (LOCATED, "03_location_cleaning_and_uncertainty"),
               (SPEC_MAP, "07_Specialty_Standardization"), (XWALK, "06_NFHS_Cleaning"),
               (PIN_CENTERS, "03_geo"), (GEO_LOOKUP, "03_geo")]:
    assert spark.catalog.tableExists(t), f"{t} is missing — run '{who}' first"

RUN_ID = "silver-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
audit_rows = []
def log(step, rule, n, note=""):
    audit_rows.append((RUN_ID, f"silver-{step}", rule, int(n), note))
    print(f"[{step}] {rule}: {n:,} {note}")

bronze = spark.read.table(BRONZE)
located = spark.read.table(LOCATED)
BRONZE_N = bronze.count()
BRONZE_RUN = bronze.select(F.max("cleaned_run_id")).first()[0]
log(0, "bronze rows in", BRONZE_N, f"bronze run {BRONZE_RUN}")
log(0, "located rows in", located.count())

# COMMAND ----------

# DBTITLE 1,1. Join Bronze + Location (one location answer)
# MAGIC %md
# MAGIC ## 1. Join Bronze with the Location notebook's output
# MAGIC Bronze decides **which rows exist** (left join from Bronze). From Location we take only location columns.
# MAGIC Bronze's own `geo_confidence` / imputed coordinates are dropped so the app can't use two competing answers; Bronze's original coordinates stay as `latitude_raw` / `longitude_raw`.

# COMMAND ----------

LOC_COLS = ["pin", "final_lat", "final_lon", "coord_source", "pin_source", "match_method", "primary_district",
            "dist_to_pin_km", "location_error_km", "location_uncertainty", "uncertainty_tier",
            "location_evidence", "location_warnings", "possible_merged_record", "conflict_state"]
loc = located.select("unique_id", F.col("state").alias("loc_state"),
                     *[F.col(c).alias({"conflict_state": "location_state_conflict", "pin": "location_pin"}.get(c, c)) for c in LOC_COLS])

df = (bronze.drop("geo_confidence", "latitude", "longitude")
            .join(loc, "unique_id", "left"))
missing_loc = df.where("final_lat IS NULL AND uncertainty_tier IS NULL").count()
log(1, "bronze facilities with no Location row", missing_loc)
log(1, "location rows not in bronze (quarantined/duplicates upstream, ignored)",
    located.join(bronze, "unique_id", "left_anti").count())

df = (df.withColumnRenamed("final_lat", "latitude").withColumnRenamed("final_lon", "longitude")
        .withColumn("location_confidence",
                    F.when(F.col("uncertainty_tier") == "low", "high")
                     .when(F.col("uncertainty_tier") == "medium", "medium")
                     .when(F.col("uncertainty_tier") == "high", "low")
                     .otherwise("unknown")))
for r in df.groupBy("location_confidence").count().collect():
    log(1, f"location_confidence = {r['location_confidence']}", r["count"])

# COMMAND ----------

# DBTITLE 1,2. State fix when the pincode is the wrong field
# MAGIC %md
# MAGIC ## 2. State: don't trust a pincode that is far from the facility
# MAGIC Bronze sets `state` = pincode's state. That's right in most conflicts, but when the facility's own coordinates are > 200 km from the pincode, the **pincode** is the wrong field — keep the written state instead.

# COMMAND ----------

fix = (F.col("state_conflict") & F.col("state_from_text").isNotNull() & (F.col("km_from_pincode") > STATE_OVERRIDE_KM))
log(2, "state restored from written state (pincode > 200 km away)", df.where(fix).count(),
    "; ".join(f"{r.name}: {r.state} -> {r.state_from_text}" for r in df.where(fix).select("name", "state", "state_from_text").limit(5).collect()))
df = (df.withColumn("state_before_silver", F.col("state"))
        .withColumn("state", F.when(fix, F.col("state_from_text")).otherwise(F.col("state")))
        .withColumn("state_source", F.when(fix, F.lit("stated (pincode far from coordinates)")).otherwise(F.col("state_source"))))

# COMMAND ----------

# DBTITLE 1,3. City names: current names, consistent case
# MAGIC %md
# MAGIC ## 3. City: current official names and consistent capitalization
# MAGIC `city` = trimmed, single-spaced, Title Case, with renamed cities updated. Raw value kept in `city_raw`.

# COMMAND ----------

CITY_RENAMES = {
    "gurgaon": "Gurugram", "bangalore": "Bengaluru", "bombay": "Mumbai", "calcutta": "Kolkata",
    "madras": "Chennai", "poona": "Pune", "mysore": "Mysuru", "trivandrum": "Thiruvananthapuram",
    "cochin": "Kochi", "baroda": "Vadodara", "allahabad": "Prayagraj", "benares": "Varanasi",
    "banaras": "Varanasi", "pondicherry": "Puducherry", "vizag": "Visakhapatnam", "calicut": "Kozhikode",
    "mangalore": "Mangaluru", "belgaum": "Belagavi", "hubli": "Hubballi", "gulbarga": "Kalaburagi",
    "trichy": "Tiruchirappalli", "tiruchirapalli": "Tiruchirappalli", "shimoga": "Shivamogga",
    "tumkur": "Tumakuru", "bellary": "Ballari", "faizabad": "Ayodhya",
}
rename_map = F.create_map([F.lit(x) for kv in CITY_RENAMES.items() for x in kv])
tidy = F.initcap(F.regexp_replace(F.trim(F.col("address_city")), r"\s+", " "))
ckey = F.lower(F.regexp_replace(F.trim(F.col("address_city")), r"\s+", " "))
df = (df.withColumn("city_raw", F.col("address_city"))
        .withColumn("city", F.coalesce(rename_map[ckey], tidy)))
log(3, "cities renamed to current name", df.where(rename_map[ckey].isNotNull()).count())
log(3, "cities re-capitalized/trimmed only", df.where(rename_map[ckey].isNull() & (F.col("city") != F.col("city_raw"))).count())

# COMMAND ----------

# DBTITLE 1,3b. Location vote (coordinates vs PIN vs city)
# MAGIC %md
# MAGIC ## 3b. Location vote: coordinates vs PIN vs city
# MAGIC Each facility has three independent location clues: its **coordinates**, its **PIN** (centre from `03_geo.pin_centers`), and its **city name** (centre from `03_geo.geo_lookup`, same state).
# MAGIC Two clues *agree* when they are within 30 km. When the coordinates are more than 30 km from the PIN:
# MAGIC
# MAGIC | PIN ~ city | coords ~ city | decision (`location_vote`) | point | district |
# MAGIC |---|---|---|---|---|
# MAGIC | yes | — | `coords_wrong_moved_to_pin` | PIN centre | unchanged (PIN's) |
# MAGIC | no | yes | `pin_wrong_kept_coords` | coordinates | from the PIN nearest the coordinates |
# MAGIC | no | no | `unresolved` | unchanged | unchanged |
# MAGIC | city unknown | | `unresolved_no_city` | unchanged | unchanged |
# MAGIC
# MAGIC Facilities whose coordinates agree with the PIN get `coords_agree_with_pin`. Facilities missing either one get `not_checked`.
# MAGIC Resolved facilities become `medium` confidence (two of three clues agree). Unresolved ones stay `low`. The original point is kept in `latitude_located` / `longitude_located`.

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

def hav_km(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2))

v = df.select("unique_id", "latitude", "longitude", "location_pin", "pincode", "city", "city_raw", "state").toPandas()
pc = spark.read.table(PIN_CENTERS).select("pincode", "lat", "lon", "state").toPandas()
pl = spark.read.table(PIN_LOOKUP).select(F.col("pincode_str").alias("pincode"), "primary_district").toPandas()
gl = (spark.read.table(GEO_LOOKUP).where("type IN ('district', 'town') AND lat IS NOT NULL")
      .select("place", "type", "lat", "lon", "state", "n_pins").toPandas())

# PIN clue: Location's PIN, else Bronze's
v["vote_pin"] = v["location_pin"].fillna(v["pincode"])
v = v.merge(pc.rename(columns={"pincode": "vote_pin", "lat": "p_lat", "lon": "p_lon", "state": "p_state"}), on="vote_pin", how="left")

# City clue: best place with that name in the facility's state (district beats town, then more PINs)
gl = gl.assign(_r=(gl["type"] != "district").astype(int)).sort_values(["_r", "n_pins"], ascending=[True, False])
city_idx = gl.drop_duplicates(["place", "state"]).set_index(["place", "state"])[["lat", "lon"]]
def city_point(names):
    k = pd.MultiIndex.from_arrays([names.fillna("").str.lower().str.replace(r"\s+", " ", regex=True).str.strip(), v["state"]])
    return city_idx.reindex(k).to_numpy()
cp = city_point(v["city"]); cr = city_point(v["city_raw"])
cp = np.where(np.isnan(cp), cr, cp)
v["c_lat"], v["c_lon"] = cp[:, 0], cp[:, 1]

v["km_coords_to_pin"] = hav_km(v["latitude"], v["longitude"], v["p_lat"], v["p_lon"])
v["km_coords_to_city"] = hav_km(v["latitude"], v["longitude"], v["c_lat"], v["c_lon"])
v["km_pin_to_city"] = hav_km(v["p_lat"], v["p_lon"], v["c_lat"], v["c_lon"])
A = VOTE_AGREE_KM
far = v["km_coords_to_pin"] > A
v["location_vote"] = np.select(
    [v["km_coords_to_pin"].isna(), ~far,
     far & (v["km_pin_to_city"] <= A),
     far & (v["km_coords_to_city"] <= A),
     far & v["c_lat"].isna()],
    ["not_checked", "coords_agree_with_pin", "coords_wrong_moved_to_pin", "pin_wrong_kept_coords", "unresolved_no_city"],
    default="unresolved")

# District for "PIN was wrong": the PIN nearest the coordinates, using Location's district names
pcl = pc.dropna(subset=["lat"]).reset_index(drop=True)
m = v["location_vote"] == "pin_wrong_kept_coords"
v["vote_district"] = None
v["vote_state"] = None
if m.any():
    _, j = BallTree(np.radians(pcl[["lat", "lon"]]), metric="haversine").query(np.radians(v.loc[m, ["latitude", "longitude"]]), k=1)
    near = pcl.iloc[j[:, 0]][["pincode", "state"]].merge(pl, on="pincode", how="left")
    v.loc[m, "vote_district"] = near["primary_district"].to_numpy()
    v.loc[m, "vote_state"] = near["state"].to_numpy()

vote_cols = ["unique_id", "location_vote", "p_lat", "p_lon", "km_coords_to_pin", "km_coords_to_city", "km_pin_to_city",
             "vote_district", "vote_state"]
vote = spark.createDataFrame(v[vote_cols].astype({"vote_district": "object", "vote_state": "object"}),
                             "unique_id string, location_vote string, p_lat double, p_lon double, km_coords_to_pin double, "
                             "km_coords_to_city double, km_pin_to_city double, vote_district string, vote_state string")
for k, n in v["location_vote"].value_counts().items():
    log("3b", f"location_vote = {k}", n)

moved = F.col("location_vote") == "coords_wrong_moved_to_pin"
pinbad = F.col("location_vote") == "pin_wrong_kept_coords"
unres = F.col("location_vote").startswith("unresolved")
df = (df.join(vote, "unique_id", "left")
        .withColumn("latitude_located", F.col("latitude")).withColumn("longitude_located", F.col("longitude"))
        .withColumn("latitude", F.when(moved, F.col("p_lat")).otherwise(F.col("latitude")))
        .withColumn("longitude", F.when(moved, F.col("p_lon")).otherwise(F.col("longitude")))
        .withColumn("primary_district", F.when(pinbad & F.col("vote_district").isNotNull(), F.col("vote_district"))
                                         .otherwise(F.col("primary_district")))
        .withColumn("state", F.when(pinbad & F.col("vote_state").isNotNull(), F.col("vote_state")).otherwise(F.col("state")))
        .withColumn("location_confidence", F.when(moved | pinbad, F.lit("medium"))
                                            .when(unres, F.lit("low")).otherwise(F.col("location_confidence")))
        .withColumn("location_error_km", F.when(moved, F.greatest(F.lit(5.0), F.col("km_pin_to_city")))
                                          .when(pinbad, F.greatest(F.lit(2.0), F.col("km_coords_to_city")))
                                          .otherwise(F.col("location_error_km")))
        .withColumn("location_warnings", F.when(moved, F.array_union(F.coalesce("location_warnings", F.array().cast("array<string>")), F.array(F.lit("moved_to_pin_area"))))
                                          .when(pinbad, F.array_union(F.coalesce("location_warnings", F.array().cast("array<string>")), F.array(F.lit("pin_disagrees_with_coords"))))
                                          .when(unres, F.array_union(F.coalesce("location_warnings", F.array().cast("array<string>")), F.array(F.lit("location_unresolved"))))
                                          .otherwise(F.col("location_warnings")))
        .withColumn("location_evidence", F.concat_ws(" ", "location_evidence",
            F.when(moved, F.format_string("Vote: the listed coordinates were %.0f km from both the PIN and the city, so the point was moved to the PIN area.", F.col("km_coords_to_pin")))
             .when(pinbad, F.format_string("Vote: the coordinates match the city but the PIN is %.0f km away, so the PIN was treated as the error.", F.col("km_coords_to_pin")))
             .when(unres, F.lit("Vote: coordinates, PIN and city disagree; this location is uncertain."))))
        .drop("p_lat", "p_lon", "vote_district", "vote_state"))
log("3b", "districts changed because the PIN was wrong", v.loc[m].shape[0])

# COMMAND ----------

# DBTITLE 1,4. Possible duplicates (same name + city)
# MAGIC %md
# MAGIC ## 4. Possible duplicates: same normalized name **and** city under different IDs
# MAGIC Bronze removed exact copies and same name + street. Here, rows sharing name + city get a `duplicate_group_id`; the most complete one is `is_primary_record = true`.
# MAGIC Chains (Apollo in many cities) are not affected because the city differs. **Nothing is deleted** — Gold uses only primary records.

# COMMAND ----------

name_key = F.lower(F.regexp_replace("name", r"[^A-Za-z0-9]", ""))
city_key = F.lower(F.regexp_replace("city", r"[^A-Za-z]", ""))
filled = sum(F.when(F.col(c).isNotNull(), 1).otherwise(0) for c in ["officialPhone", "email", "officialWebsite", "address_line1", "pincode",
                                                                    "latitude", "capacity_beds", "number_doctors", "specialties", "procedure", "equipment"])
df = df.withColumn("_k", F.when(name_key.isNotNull() & city_key.isNotNull() & (F.length(name_key) > 0), F.concat_ws("|", name_key, city_key)))
w_cnt = Window.partitionBy("_k")
w_rank = Window.partitionBy("_k").orderBy(F.desc(filled), "unique_id")
df = (df.withColumn("_n", F.when(F.col("_k").isNotNull(), F.count("*").over(w_cnt)).otherwise(1))
        .withColumn("duplicate_group_id", F.when(F.col("_n") > 1, F.sha2(F.col("_k"), 256).substr(1, 12)))
        .withColumn("is_primary_record", F.when(F.col("_n") > 1, F.row_number().over(w_rank) == 1).otherwise(True))
        .drop("_k", "_n"))
log(4, "rows in a possible-duplicate group", df.where("duplicate_group_id IS NOT NULL").count())
log(4, "non-primary duplicate rows (excluded from Gold, kept here)", df.where("NOT is_primary_record").count())

# COMMAND ----------

# DBTITLE 1,5. Junk shared phone numbers
# MAGIC %md
# MAGIC ## 5. Official phones shared by many unrelated facilities
# MAGIC A number listed as the official phone of ≥ 5 facilities in ≥ 2 different cities with different names is a directory/aggregator number (e.g. one number on 87 facilities).
# MAGIC `official_phone` = cleaned phone, blanked when suspect; `officialPhone` raw kept; `phone_suspect` flag.

# COMMAND ----------

digits = F.regexp_replace("officialPhone", r"[^0-9]", "")
phone_stats = (df.where(F.col("officialPhone").isNotNull())
                 .groupBy(digits.alias("_p"))
                 .agg(F.count("*").alias("_pn"), F.countDistinct("city").alias("_pc"),
                      F.countDistinct(F.lower(F.regexp_replace("name", r"[^A-Za-z]", ""))).alias("_pnames")))
suspect = phone_stats.where(f"_pn >= {SHARED_PHONE_MIN_FACILITIES} AND _pc >= 2 AND _pnames >= 2").select("_p", F.lit(True).alias("phone_suspect"))
df = (df.withColumn("_p", digits).join(F.broadcast(suspect), "_p", "left").drop("_p")
        .withColumn("phone_suspect", F.coalesce("phone_suspect", F.lit(False)))
        .withColumn("official_phone", F.when(~F.col("phone_suspect"), F.col("officialPhone"))))
log(5, "distinct suspect shared phone numbers", suspect.count())
log(5, "facilities whose official phone was blanked as suspect", df.where("phone_suspect").count())

# COMMAND ----------

# DBTITLE 1,6. Numbers, operator type, country
df = (df.withColumn("doctor_count_implausible", F.col("number_doctors") > MAX_PLAUSIBLE_DOCTORS)
        .withColumn("number_doctors", F.when(F.col("number_doctors") <= MAX_PLAUSIBLE_DOCTORS, F.col("number_doctors")))
        .withColumn("doctor_count_implausible", F.coalesce("doctor_count_implausible", F.lit(False))))
log(6, "doctor counts > 2,000 set to NULL", df.where("doctor_count_implausible").count())

df = df.withColumn("operator_type", F.when(F.lower("operatorTypeId").isin("public", "government"), "public")
                                     .when(F.lower("operatorTypeId") == "private", "private").otherwise("unknown"))
log(6, "operatorTypeId 'government' unified to 'public'", df.where(F.lower("operatorTypeId") == "government").count())

df = df.withColumn("country_suspect", F.col("address_country").isNotNull() & (F.lower(F.trim("address_country")) != "india"))
log(6, "rows with non-India country value (flagged)", df.where("country_suspect").count(),
    ", ".join(r[0] for r in df.where("country_suspect").select("address_country").limit(5).collect()))

# COMMAND ----------

# DBTITLE 1,7. Standard specialties
# MAGIC %md
# MAGIC ## 7. Specialties → standard codes (from `07_Specialty_Standardization`)
# MAGIC - `specialty_codes`: distinct standard codes, e.g. `["cardiology","pediatrics"]` — use this for search/filters
# MAGIC - `specialty_labels`: readable labels in the same order
# MAGIC - `specialties_other`: unusual specialties with no standard code (kept, not lost)
# MAGIC - `specialties`: Bronze's raw list, unchanged

# COMMAND ----------

smap = spark.read.table(SPEC_MAP).select(F.col("raw_specialty").alias("_raw"), "specialty_code", "specialty_label", "is_standard")
ex = df.select("unique_id", F.explode("specialties").alias("_raw")).withColumn("_raw", F.trim("_raw"))
# junk left by column-shifted rows (e.g. '{coordinates:80.98', '26.88', 'type:Point}') is dropped, not mapped
JUNK = r"^[^A-Za-z]*$|[{}:]"
log(7, "junk specialty items dropped (no letters, or JSON fragments)", ex.where(F.col("_raw").rlike(JUNK)).count(),
    ", ".join(r[0] for r in ex.where(F.col("_raw").rlike(JUNK)).select("_raw").distinct().limit(5).collect()))
ex = ex.where(~F.col("_raw").rlike(JUNK)).join(F.broadcast(smap), "_raw", "left")
unmapped_raw = ex.where("is_standard IS NULL").select("_raw").distinct().count()
spec = ex.groupBy("unique_id").agg(
    F.array_sort(F.array_distinct(F.collect_list(F.when(F.col("is_standard"), F.col("specialty_code"))))).alias("specialty_codes"),
    F.array_distinct(F.collect_list(F.when(~F.col("is_standard") | F.col("is_standard").isNull(), F.col("specialty_label")))).alias("specialties_other"),
)
labels = smap.where("is_standard").select("specialty_code", "specialty_label").distinct() \
             .groupBy("specialty_code").agg(F.first("specialty_label").alias("lbl"))
label_map = {r.specialty_code: r.lbl for r in labels.collect()}
lmap = F.create_map([F.lit(x) for kv in label_map.items() for x in kv])
df = (df.join(spec, "unique_id", "left")
        .withColumn("specialty_codes", F.when(F.size("specialty_codes") > 0, F.col("specialty_codes")))
        .withColumn("specialties_other", F.when(F.size("specialties_other") > 0, F.col("specialties_other")))
        .withColumn("specialty_labels", F.transform("specialty_codes", lambda c: lmap[c]))
        .withColumn("n_specialty_codes", F.coalesce(F.size("specialty_codes"), F.lit(0))))
log(7, "raw specialty strings not found in specialty_map (re-run 07 if > 0)", unmapped_raw)
log(7, "facilities with >= 1 standard specialty", df.where("n_specialty_codes > 0").count())

# COMMAND ----------

# DBTITLE 1,8. Link to NFHS district
# MAGIC %md
# MAGIC ## 8. NFHS link: facility district → NFHS district
# MAGIC Location gives each facility an India Post `primary_district`; the 06 crosswalk maps that to the NFHS district name. Gold joins health indicators on (`nfhs_state`, `nfhs_district`).

# COMMAND ----------

xw = (spark.read.table(XWALK)
        .select(F.col("pin_state").alias("_ps"), F.col("pin_district").alias("_pd"),
                F.col("state").alias("nfhs_state"), F.col("district").alias("nfhs_district"))
        .where("_pd IS NOT NULL")
        .groupBy("_ps", "_pd").agg(F.first("nfhs_state").alias("nfhs_state"), F.first("nfhs_district").alias("nfhs_district"),
                                   F.count("*").alias("nfhs_districts_sharing")))
# Districts created after NFHS-5 (2019–21) have no survey row: link them to the district they were split from.
NEW_DISTRICT_PARENT = {  # (state, India Post district) -> NFHS district (same state)
    ("Andhra Pradesh", "ntr"): "Krishna", ("Andhra Pradesh", "tirupati"): "Chittoor",
    ("Andhra Pradesh", "annamayya"): "Y.S.R.", ("Andhra Pradesh", "eluru"): "West Godavari",
    ("Andhra Pradesh", "nandyal"): "Kurnool", ("Andhra Pradesh", "bapatla"): "Guntur",
    ("Andhra Pradesh", "palnadu"): "Guntur", ("Andhra Pradesh", "sri sathya sai"): "Anantapur",
    ("Andhra Pradesh", "kakinada"): "East Godavari", ("Andhra Pradesh", "konaseema"): "East Godavari",
    ("Andhra Pradesh", "anakapalli"): "Visakhapatnam", ("Andhra Pradesh", "parvathipuram manyam"): "Vizianagaram",
    ("Tamil Nadu", "chengalpattu"): "Kancheepuram", ("Tamil Nadu", "tenkasi"): "Tirunelveli",
    ("Tamil Nadu", "ranipet"): "Vellore", ("Tamil Nadu", "tirupathur"): "Vellore",
    ("Tamil Nadu", "kallakurichi"): "Viluppuram", ("Tamil Nadu", "mayiladuthurai"): "Nagapattinam",
    ("Telangana", "hanumakonda"): "Warangal Urban", ("West Bengal", "kalimpong"): "Darjiling",
    ("Punjab", "malerkotla"): "Sangrur",
}
nfhs_names = {(r.state, r.district) for r in spark.read.table(f"{TGT}.nfhs_clean").select("state", "district").collect()}
bad_parents = [f"{st}/{d}" for (st, _), d in NEW_DISTRICT_PARENT.items() if (st, d) not in nfhs_names]
assert not bad_parents, f"Parent districts not found in nfhs_clean: {bad_parents}"
parent = spark.createDataFrame([(st.lower(), d, st, p) for (st, d), p in NEW_DISTRICT_PARENT.items()],
                               "_ps string, _pd string, p_state string, p_district string")

df = (df.withColumn("_ps", F.lower("state")).withColumn("_pd", F.lower(F.trim("primary_district")))
        .join(F.broadcast(xw), ["_ps", "_pd"], "left")
        .join(F.broadcast(parent), ["_ps", "_pd"], "left")
        .withColumn("nfhs_link_type", F.when(F.col("nfhs_district").isNotNull(), "same district")
                                       .when(F.col("p_district").isNotNull(), "parent district (created after NFHS-5)"))
        .withColumn("nfhs_state", F.coalesce("nfhs_state", "p_state"))
        .withColumn("nfhs_district", F.coalesce("nfhs_district", "p_district"))
        .drop("_ps", "_pd", "p_state", "p_district"))
log(8, "facilities linked via parent district (district created after NFHS-5)",
    df.where("nfhs_link_type LIKE 'parent%'").count())
log(8, "facilities linked to an NFHS district", df.where("nfhs_district IS NOT NULL").count())
log(8, "facilities with a district but no NFHS match", df.where("primary_district IS NOT NULL AND nfhs_district IS NULL").count(),
    ", ".join(f"{r.state}/{r.primary_district}" for r in df.where("primary_district IS NOT NULL AND nfhs_district IS NULL")
              .groupBy("state", "primary_district").count().orderBy(F.desc("count")).limit(8).collect()))

# COMMAND ----------

# DBTITLE 1,9. Column order and write
# MAGIC %md
# MAGIC ## 9. Write `facilities_silver`
# MAGIC Key columns first; everything else from Bronze follows unchanged.

# COMMAND ----------

FRONT = ["unique_id", "name", "facility_type", "operator_type", "is_primary_record", "duplicate_group_id",
         "city", "state", "pincode", "primary_district", "nfhs_state", "nfhs_district",
         "nfhs_link_type", "latitude", "longitude", "location_vote", "location_confidence", "location_error_km", "location_evidence", "location_warnings",
         "specialty_codes", "specialty_labels", "specialties_other", "n_specialty_codes",
         "official_phone", "phone_suspect", "email", "officialWebsite",
         "capacity_beds", "number_doctors", "doctor_count_plausible", "doctor_count_implausible", "year_established",
         "state_source", "state_conflict", "state_before_silver", "city_raw", "country_suspect"]
silver = df.select(*FRONT, *[c for c in df.columns if c not in FRONT]) \
           .withColumn("silver_run_id", F.lit(RUN_ID)).withColumn("bronze_run_id", F.lit(BRONZE_RUN))
silver.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SILVER)
SILVER_N = spark.read.table(SILVER).count()
log(9, "facilities_silver rows written", SILVER_N, SILVER)

(spark.createDataFrame(audit_rows, "run_id string, step string, rule string, rows_affected long, note string")
      .withColumn("logged_at", F.current_timestamp())
      .write.mode("append").option("mergeSchema", "true").saveAsTable(AUDIT))
display(spark.createDataFrame(audit_rows, "run_id string, step string, rule string, rows_affected long, note string"))

# COMMAND ----------

# DBTITLE 1,10. Validation
# MAGIC %md
# MAGIC ## 10. Validation — all checks must PASS (raises an error otherwise)

# COMMAND ----------

s = spark.read.table(SILVER)
CANON_STATES = [
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
    "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
]
std_codes = [r[0] for r in spark.read.table(SPEC_MAP).where("is_standard").select("specialty_code").distinct().collect()]
checks = []
def check(name, ok, detail=""):
    checks.append((name, "PASS" if ok else "FAIL", str(detail)))

check("Same rows as Bronze", SILVER_N == BRONZE_N, f"silver {SILVER_N} vs bronze {BRONZE_N}")
check("unique_id is unique", s.select("unique_id").distinct().count() == SILVER_N)
check("Every facility has a Location result", missing_loc == 0, f"{missing_loc} without")
bad = s.where(F.col("latitude").isNotNull() & ~(F.col("latitude").between(6, 38) & F.col("longitude").between(68, 98))).count()
check("Every coordinate inside India (or NULL)", bad == 0, f"{bad} outside")
bad = s.where(F.col("state").isNotNull() & ~F.col("state").isin(CANON_STATES)).count()
check("Every state is canonical", bad == 0, f"{bad} invalid")
bad = s.where(F.lower("city").isin(list(CITY_RENAMES))).count()
check("No old city names left", bad == 0, f"{bad} rows")
bad = s.where(F.exists("specialty_codes", lambda c: ~c.isin(std_codes))).count()
check("specialty_codes only contain standard codes", bad == 0, f"{bad} rows")
check("Every raw specialty found in specialty_map", unmapped_raw == 0, f"{unmapped_raw} unmapped strings")
bad = s.where(f"number_doctors > {MAX_PLAUSIBLE_DOCTORS}").count()
check("No implausible doctor counts", bad == 0, f"{bad} rows")
bad = s.where("phone_suspect AND official_phone IS NOT NULL").count()
check("Suspect phones are blanked in official_phone", bad == 0, f"{bad} rows")
groups = s.where("duplicate_group_id IS NOT NULL").groupBy("duplicate_group_id").agg(F.sum(F.col("is_primary_record").cast("int")).alias("p"))
bad = groups.where("p != 1").count()
check("Each duplicate group has exactly one primary record", bad == 0, f"{bad} groups")
bad = s.where(~F.col("location_confidence").isin("high", "medium", "low", "unknown")).count()
check("location_confidence is high/medium/low/unknown", bad == 0, f"{bad} rows")
bad = s.where("km_coords_to_pin > 30 AND location_vote = 'coords_agree_with_pin'").count()
check("Every facility > 30 km from its PIN went through the vote", bad == 0, f"{bad} rows")
bad = s.where("location_vote = 'coords_wrong_moved_to_pin' AND (abs(latitude - latitude_located) < 1e-9 AND abs(longitude - longitude_located) < 1e-9)").count()
check("Moved facilities really moved to the PIN area", bad == 0, f"{bad} rows")
bad = s.where("location_vote IN ('coords_wrong_moved_to_pin', 'pin_wrong_kept_coords') AND location_confidence != 'medium'").count()
check("Vote-resolved facilities are medium confidence", bad == 0, f"{bad} rows")
nf_share = s.where("nfhs_district IS NOT NULL").count() / SILVER_N
check("≥ 90% of facilities linked to an NFHS district", nf_share >= 0.90, f"{nf_share:.1%}")

result = spark.createDataFrame(checks, "check string, result string, detail string")
display(result)
failed = [x for x in checks if x[1] == "FAIL"]
if failed:
    raise AssertionError(f"{len(failed)} Silver validation check(s) FAILED: " + "; ".join(x[0] for x in failed))
print(f"All {len(checks)} Silver validation checks PASSED.")
