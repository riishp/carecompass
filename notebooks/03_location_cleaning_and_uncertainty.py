# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 03 — Location Cleaning and Uncertainty
# MAGIC
# MAGIC **Purpose:** feed the facility-recommender with the most trustworthy point per facility, an error radius in km, and a plain-language explanation of where each location came from. A confident wrong location can send a patient hours in the wrong direction — this notebook flags doubt instead of hiding it.
# MAGIC
# MAGIC **Source table used:** the read-only Delta-Shared `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities` (10,088 raw rows). At authoring time no Bronze/deduped facilities table existed (the *Data Cleaning: Raw to Bronze* notebook has not been run), so notebook **02_Facilities_EDA**'s deduplication rules — one row per `unique_id` keeping the most complete record, then one row per normalized (name, address_line1) — are applied in **Step 4a**. Both sources (`facilities`, `india_post_pincode_directory`) are treated as **read-only**.
# MAGIC
# MAGIC **Outputs (all CREATE OR REPLACE — safe to re-run):** `workspace.datathon_2026.pincode_clean`, `.pin_lookup`, `.place_lookup`, `.facilities_clean`, `.facilities_located`.
# MAGIC
# MAGIC Pipeline: clean PIN directory → per-PIN cardinality → pin & place lookups → clean + dedup facilities → tiered match → final location & error → uncertainty score → plain-language evidence.
# MAGIC
# MAGIC > **Note:** PIN centroids are the approximate center of a PIN's post offices, not the facility's actual location.

# COMMAND ----------

# DBTITLE 1,Config and helpers
from pyspark.sql import functions as F, Window

# ---- read-only sources (never written back) ----
SRC = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
FACILITIES_TABLE = f"{SRC}.facilities"                  # 10,088 raw rows (notebooks 01/02)
PINCODE_TABLE = f"{SRC}.india_post_pincode_directory"  # one row per post office

# ---- output tables (created / replaced by this notebook) ----
OUT = "workspace.datathon_2026"
PINCODE_CLEAN_TABLE = f"{OUT}.pincode_clean"
PIN_LOOKUP_TABLE = f"{OUT}.pin_lookup"
PLACE_LOOKUP_TABLE = f"{OUT}.place_lookup"
FACILITIES_CLEAN_TABLE = f"{OUT}.facilities_clean"
FACILITIES_LOCATED_TABLE = f"{OUT}.facilities_located"

# ---- India bounding box (applies to both the directory and facilities) ----
LAT_MIN, LAT_MAX = 6.0, 38.0
LON_MIN, LON_MAX = 68.0, 98.0
EARTH_RADIUS_KM = 6371.0088

PIN_REGEX = r"^[1-8][0-9]{5}$"              # valid 6-digit PIN
PIN_IN_TEXT = r"\b[1-8]\d{2}\s?\d{3}\b"     # PIN embedded in an address line
JSON_POINT = r'"coordinates":\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'  # GeoJSON [lon, lat]

# literal strings that mean "missing" in the facilities table
MISSING_TOKENS = ["", "null", "nan", "none", "n/a", "na", "-"]

# canonical renames applied AFTER basic normalization (historical splits / mergers / renames)
STATE_RENAMES = {
    "orissa": "odisha",
    "pondicherry": "puducherry",
    "tamilnadu": "tamil nadu",
    "new delhi": "delhi",
    "dadra and nagar haveli": "dadra and nagar haveli and daman and diu",
    "daman and diu": "dadra and nagar haveli and daman and diu",
    "uttranchal": "uttarakhand",
    "chattisgarh": "chhattisgarh",
    "andaman and nicobar": "andaman and nicobar islands",
}


def write_replace(df, table):
    """CREATE OR REPLACE semantics — the notebook can be re-run safely."""
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)


def std_name(col):
    """trim, lowercase, collapse whitespace, '&' -> 'and', drop leading 'the ';
    missing tokens ('null', 'na', ...) become NULL."""
    s = F.lower(F.trim(F.coalesce(col, F.lit(""))))
    s = F.regexp_replace(s, r"\s+", " ")
    s = F.regexp_replace(s, r"\s*&\s*", " and ")
    s = F.regexp_replace(s, r"^the\s+", "")
    return F.when(s.isin(MISSING_TOKENS), F.lit(None).cast("string")).otherwise(s)


def std_state(col):
    """std_name + canonical state renames (Orissa->Odisha, Pondicherry->Puducherry,
    Telangana / Ladakh-era splits, merged DNH + Daman and Diu)."""
    s = std_name(col)
    mapping = F.create_map([F.lit(x) for kv in STATE_RENAMES.items() for x in kv])
    return F.coalesce(F.element_at(mapping, s), s)


def parse_coord(colname):
    """First whitespace token of the coordinate string, stripped of degree/compass
    symbols, cast to double only when it looks like a number (ANSI-mode Spark
    throws CAST_INVALID_INPUT on ''/'NA'). NA / empty / garbage -> NULL."""
    tok = F.split(F.trim(F.coalesce(F.col(colname), F.lit(""))), r"\s+").getItem(0)
    tok = F.regexp_replace(tok, r"[^0-9.\-]", "")
    return F.when(tok.rlike(r"^-?\d+(\.\d+)?$"), tok.cast("double"))


def in_india(lat, lon):
    """Both present, non-zero, inside the bounding box."""
    return (lat.isNotNull() & lon.isNotNull() & (lat != 0) & (lon != 0)
            & lat.between(LAT_MIN, LAT_MAX) & lon.between(LON_MIN, LON_MAX))


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km (native expressions, no UDF)."""
    dlat = F.radians(lat2) - F.radians(lat1)
    dlon = F.radians(lon2) - F.radians(lon1)
    a = F.sin(dlat / 2) ** 2 + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.sin(dlon / 2) ** 2
    return F.lit(2 * EARTH_RADIUS_KM) * F.asin(F.sqrt(a))

# COMMAND ----------

# DBTITLE 1,Install plotly
# MAGIC %pip install plotly

# COMMAND ----------

# DBTITLE 1,Step 1 — Clean the PIN directory
# MAGIC %md
# MAGIC ## Step 1 — Clean the PIN directory → `pincode_clean`
# MAGIC
# MAGIC Cast `pincode` to a 6-digit string (drop anything not matching `^[1-8][0-9]{5}$`), standardize state/district names (trim, lowercase, `&` → `and`, historical renames/mergers), parse coordinates (NA / 0 / outside India → NULL; likely swapped lat/lon pairs flagged and repaired), and remove exact duplicate rows. Row counts printed after each step.

# COMMAND ----------

# DBTITLE 1,Step 1 — Clean the PIN directory
pin_raw = spark.table(PINCODE_TABLE)
n_raw = pin_raw.count()
print(f"PIN directory rows (raw): {n_raw:,}")

# --- 1a. pincode -> 6-digit string, keep only valid PINs ---
pin_s1 = (
    pin_raw
    .withColumn("pincode_str", F.lpad(F.col("pincode").cast("string"), 6, "0"))
    .filter(F.col("pincode_str").rlike(PIN_REGEX))
)
n_valid = pin_s1.count()
print(f"after PIN validation: {n_valid:,}  (dropped {n_raw - n_valid:,})")

# --- 1b. standardize state / district / office / division names ---
pin_s2 = (
    pin_s1
    .withColumn("district_std", std_name(F.col("district")))
    .withColumn("state_std", std_state(F.col("statename")))
    .withColumn("officename_std",
                F.regexp_replace(std_name(F.col("officename")), r"\s+(b\.o|s\.o|h\.o|g\.p\.o)$", ""))
    .withColumn("divisionname_std", std_name(F.col("divisionname")))
)

# --- 1c. coordinates: parse, null-out NA/0/out-of-bounds, flag+repair likely swaps ---
pin_s3 = (
    pin_s2
    .withColumn("lat_raw", parse_coord("latitude"))
    .withColumn("lon_raw", parse_coord("longitude"))
    # a pair whose numbers sit in each other's range is almost certainly swapped
    .withColumn("likely_swapped",
                F.col("lat_raw").between(LON_MIN, LON_MAX) & F.col("lon_raw").between(LAT_MIN, LAT_MAX))
    .withColumn("lat_fix", F.when(F.col("likely_swapped"), F.col("lon_raw")).otherwise(F.col("lat_raw")))
    .withColumn("lon_fix", F.when(F.col("likely_swapped"), F.col("lat_raw")).otherwise(F.col("lon_raw")))
    .withColumn("pin_lat", F.when(in_india(F.col("lat_fix"), F.col("lon_fix")), F.col("lat_fix")))
    .withColumn("pin_lon", F.when(in_india(F.col("lat_fix"), F.col("lon_fix")), F.col("lon_fix")))
)
n_geocoded = pin_s3.filter("pin_lat IS NOT NULL").count()
n_swapped = pin_s3.filter("likely_swapped").count()
print(f"after coordinate cleaning: {n_geocoded:,} usable coordinates ({n_swapped:,} likely-swapped pairs repaired + flagged)")

# --- 1d. exact duplicate rows (on standardized columns) ---
pincode_clean = (
    pin_s3.select(
        "pincode_str", "district_std", "state_std", "officename_std", "officetype", "delivery",
        "divisionname_std", "pin_lat", "pin_lon", "likely_swapped")
    .distinct()
)
n_clean = pincode_clean.count()
print(f"after removing exact duplicates: {n_clean:,}  (dropped {n_valid - n_clean:,})")
print(f"\nDistinct PINs: {pincode_clean.select('pincode_str').distinct().count():,}")

write_replace(pincode_clean, PINCODE_CLEAN_TABLE)
print(f"\nWrote {PINCODE_CLEAN_TABLE}")

# COMMAND ----------

# DBTITLE 1,Step 2 — Cardinality per PIN
# MAGIC %md
# MAGIC ## Step 2 — Cardinality per PIN
# MAGIC
# MAGIC One directory row = one post office, so a PIN can span several districts or even states. Count distinct districts / states / post offices / geocoded post offices per PIN; report how many PINs span more than one district or state, with examples.

# COMMAND ----------

# DBTITLE 1,Step 2 — Cardinality per PIN
pincode_clean_df = spark.table(PINCODE_CLEAN_TABLE)

pin_cardinality = (
    pincode_clean_df.groupBy("pincode_str")
    .agg(
        F.size(F.collect_set("district_std")).alias("n_districts"),
        F.size(F.collect_set("state_std")).alias("n_states"),
        F.count("*").alias("n_post_offices"),
        F.sum(F.when(F.col("pin_lat").isNotNull(), 1).otherwise(0)).alias("n_geocoded"),
        F.collect_set("district_std").alias("districts_card"),
    )
)
n_pins = pin_cardinality.count()
multi_district = pin_cardinality.filter("n_districts > 1")
multi_state = pin_cardinality.filter("n_states > 1")

print(f"Distinct PINs: {n_pins:,}")
print(f"PINs spanning more than one district: {multi_district.count():,} ({100 * multi_district.count() / n_pins:.1f}%)")
print(f"PINs spanning more than one state:    {multi_state.count():,} ({100 * multi_state.count() / n_pins:.1f}%)")

print("\nExamples — PINs spanning multiple districts:")
display(multi_district.orderBy(F.desc("n_districts")).limit(10)
        .select("pincode_str", "n_districts", "n_states", "districts_card"))

print("Examples — PINs spanning multiple states:")
display(multi_state.orderBy(F.desc("n_states")).limit(10).select("pincode_str", "n_states", "n_districts"))

# COMMAND ----------

# DBTITLE 1,Step 3 — pin_lookup and place_lookup
# MAGIC %md
# MAGIC ## Step 3 — Lookups for facilities *and* user queries
# MAGIC
# MAGIC **`pin_lookup`** (one row per PIN): `primary_district` / `primary_state` (mode of the PIN's offices), `district_share`, `n_districts`, `n_states`, `is_ambiguous`, `centroid_lat` / `centroid_lon`, `coord_spread_km` (max distance from the centroid to any of the PIN's offices), `geocode_coverage`, and the sets of districts / office names / division names in the PIN.
# MAGIC
# MAGIC **`place_lookup`** (for user inputs like "Darbhanga" or "Pune district"): one row per district, and one per office name + district — each with a centroid and `radius_km` (max distance from the centroid to its post offices). Names that exist in more than one state are flagged (`multi_state`).

# COMMAND ----------

# DBTITLE 1,Step 3a — pin_lookup
# ---- mode (most common value) of district and state per PIN ----
w_dist = Window.partitionBy("pincode_str").orderBy(F.desc("n"), F.asc("district_std"))
primary_district_df = (
    pincode_clean_df.filter("district_std IS NOT NULL")
    .groupBy("pincode_str", "district_std").agg(F.count("*").alias("n"))
    .withColumn("rn", F.row_number().over(w_dist)).filter("rn = 1")
    .select("pincode_str", F.col("district_std").alias("primary_district"), F.col("n").alias("n_primary_district"))
)
w_state = Window.partitionBy("pincode_str").orderBy(F.desc("n"), F.asc("state_std"))
primary_state_df = (
    pincode_clean_df.filter("state_std IS NOT NULL")
    .groupBy("pincode_str", "state_std").agg(F.count("*").alias("n"))
    .withColumn("rn", F.row_number().over(w_state)).filter("rn = 1")
    .select("pincode_str", F.col("state_std").alias("primary_state"))
)

pin_agg = (
    pincode_clean_df.groupBy("pincode_str")
    .agg(
        F.count("*").alias("n_post_offices"),
        F.sum(F.when(F.col("pin_lat").isNotNull(), 1).otherwise(0)).alias("n_geocoded"),
        F.size(F.collect_set("district_std")).alias("n_districts"),
        F.size(F.collect_set("state_std")).alias("n_states"),
        F.collect_set("district_std").alias("districts_set"),
        F.collect_set("officename_std").alias("office_names"),
        F.collect_set("divisionname_std").alias("division_names"),
        F.avg("pin_lat").alias("centroid_lat"),
        F.avg("pin_lon").alias("centroid_lon"),
    )
)

# coord_spread_km = max distance from the PIN centroid to any of its post offices
spread = (
    pincode_clean_df
    .join(pin_agg.select("pincode_str", "centroid_lat", "centroid_lon"), "pincode_str")
    .withColumn("d_km", haversine_km(F.col("pin_lat"), F.col("pin_lon"), F.col("centroid_lat"), F.col("centroid_lon")))
    .groupBy("pincode_str").agg(F.max("d_km").alias("coord_spread_km"))
)

pin_lookup = (
    pin_agg
    .join(primary_district_df, "pincode_str", "left")
    .join(primary_state_df, "pincode_str", "left")
    .join(spread, "pincode_str", "left")
    .withColumn("district_share", F.col("n_primary_district") / F.col("n_post_offices"))
    .withColumn("is_ambiguous", F.col("n_districts") > 1)
    .withColumn("geocode_coverage", F.col("n_geocoded") / F.col("n_post_offices"))
)
write_replace(pin_lookup, PIN_LOOKUP_TABLE)
print(f"Wrote {PIN_LOOKUP_TABLE}: {pin_lookup.count():,} rows (one per PIN)")
display(pin_lookup.filter("is_ambiguous").orderBy(F.desc("n_districts")).limit(5)
        .select("pincode_str", "primary_district", "primary_state", "district_share", "n_districts", "coord_spread_km", "geocode_coverage"))

# COMMAND ----------

# DBTITLE 1,Step 3b — place_lookup
# ---- district-level places: one row per district ----
w_ds = Window.partitionBy("district_std").orderBy(F.desc("n"), F.asc("state_std"))
district_primary_state = (
    pincode_clean_df.filter("district_std IS NOT NULL AND state_std IS NOT NULL")
    .groupBy("district_std", "state_std").agg(F.count("*").alias("n"))
    .withColumn("rn", F.row_number().over(w_ds)).filter("rn = 1")
    .select("district_std", F.col("state_std").alias("d_state"))
)
district_agg = (
    pincode_clean_df.filter("district_std IS NOT NULL")
    .groupBy("district_std")
    .agg(
        F.count("*").alias("n_post_offices"),
        F.avg("pin_lat").alias("centroid_lat"),
        F.avg("pin_lon").alias("centroid_lon"),
        F.collect_set("state_std").alias("states_set"),
    )
)
district_radius = (
    pincode_clean_df
    .join(district_agg.filter("centroid_lat IS NOT NULL")
          .select("district_std", "centroid_lat", "centroid_lon"), "district_std")
    .withColumn("d_km", haversine_km(F.col("pin_lat"), F.col("pin_lon"), F.col("centroid_lat"), F.col("centroid_lon")))
    .groupBy("district_std").agg(F.max("d_km").alias("radius_km"))
)
district_rows = (
    district_agg
    .join(district_primary_state, "district_std", "left")
    .join(district_radius, "district_std", "left")
    .select(
        F.col("district_std").alias("place_name"),
        F.lit("district").alias("place_type"),
        F.lit(None).cast("string").alias("district"),
        F.col("d_state").alias("primary_state"),
        "states_set", "centroid_lat", "centroid_lon", "radius_km", "n_post_offices",
        (F.size("states_set") > 1).alias("multi_state"),
    )
)

# ---- office-level places: one row per (office name, district) ----
w_os = Window.partitionBy("officename_std", "district_std").orderBy(F.desc("n"), F.asc("state_std"))
office_primary_state = (
    pincode_clean_df.filter("officename_std IS NOT NULL AND district_std IS NOT NULL AND state_std IS NOT NULL")
    .groupBy("officename_std", "district_std", "state_std").agg(F.count("*").alias("n"))
    .withColumn("rn", F.row_number().over(w_os)).filter("rn = 1")
    .select("officename_std", "district_std", F.col("state_std").alias("o_state"))
)
office_agg = (
    pincode_clean_df.filter("officename_std IS NOT NULL AND district_std IS NOT NULL")
    .groupBy("officename_std", "district_std")
    .agg(
        F.count("*").alias("n_post_offices"),
        F.avg("pin_lat").alias("centroid_lat"),
        F.avg("pin_lon").alias("centroid_lon"),
        F.collect_set("state_std").alias("states_set"),
    )
)
office_radius = (
    pincode_clean_df
    .join(office_agg.filter("centroid_lat IS NOT NULL")
          .select("officename_std", "district_std", "centroid_lat", "centroid_lon"),
          ["officename_std", "district_std"])
    .withColumn("d_km", haversine_km(F.col("pin_lat"), F.col("pin_lon"), F.col("centroid_lat"), F.col("centroid_lon")))
    .groupBy("officename_std", "district_std").agg(F.max("d_km").alias("radius_km"))
)
# the same office NAME in more than one state -> multi_state flag
name_states = (
    pincode_clean_df.filter("officename_std IS NOT NULL AND state_std IS NOT NULL")
    .select("officename_std", "state_std").distinct()
    .groupBy("officename_std").agg(F.collect_set("state_std").alias("name_states_set"))
)
office_rows = (
    office_agg
    .join(office_primary_state, ["officename_std", "district_std"], "left")
    .join(office_radius, ["officename_std", "district_std"], "left")
    .join(name_states, "officename_std", "left")
    .select(
        F.col("officename_std").alias("place_name"),
        F.lit("office").alias("place_type"),
        F.col("district_std").alias("district"),
        F.col("o_state").alias("primary_state"),
        "states_set", "centroid_lat", "centroid_lon", "radius_km", "n_post_offices",
        (F.size("name_states_set") > 1).alias("multi_state"),
    )
)

place_lookup = district_rows.unionByName(office_rows)
write_replace(place_lookup, PLACE_LOOKUP_TABLE)

place_tbl = spark.table(PLACE_LOOKUP_TABLE)
print(f"Wrote {PLACE_LOOKUP_TABLE}: {place_tbl.count():,} rows "
      f"({place_tbl.filter("place_type = 'district'").count():,} district places, "
      f"{place_tbl.filter("place_type = 'office'").count():,} office places)")
print(f"place names existing in more than one state (multi_state): {place_tbl.filter('multi_state').select('place_name').distinct().count():,} distinct names")
display(place_tbl.filter("place_type = 'district' AND multi_state").limit(5))

# COMMAND ----------

# DBTITLE 1,Step 4 — Clean facility locations
# MAGIC %md
# MAGIC ## Step 4 — Clean facility locations → `facilities_clean`
# MAGIC
# MAGIC **4a. Deduplication (notebook 02's rules).** No Bronze facilities table exists yet, so this notebook starts from the read-only shared `facilities` table and applies the rules notebook 02 documented: one row per `unique_id` (keeping the most complete record), then one row per normalized (name, address_line1) pair (catching the same facility under different IDs).
# MAGIC
# MAGIC **4b. Cleaning.** Flag non-India country values (never drop); normalize the PIN field (strip spaces, drop a trailing `.0`, digits only) and fall back to extracting a PIN from the address lines; standardize city/state like Step 1; reconcile the `coordinates` GeoJSON string with the `latitude`/`longitude` doubles (fill nulls, flag disagreements, bounds + swap checks).
# MAGIC
# MAGIC **`possible_merged_record`** is set when the address contains ≥2 different valid PINs, names ≥2 different known places (districts), names a state different from the state field, or the facility's coordinates are >100 km from its PIN centroid.

# COMMAND ----------

# DBTITLE 1,Step 4a — Deduplicate facilities
# ---- 4a. deduplication (notebook 02's rules) ----
fac_all = spark.table(FACILITIES_TABLE)   # READ-ONLY — never written back
n_raw = fac_all.count()

# completeness = number of the 51 columns holding a real (non-missing) value
completeness = F.lit(0)
for c in fac_all.columns:
    if c in ("latitude", "longitude"):
        completeness = completeness + F.when(F.col(c).isNotNull(), 1).otherwise(0)
    else:
        s = F.lower(F.trim(F.coalesce(F.col(c), F.lit(""))))
        completeness = completeness + F.when(~s.isin(MISSING_TOKENS) & (s != ""), 1).otherwise(0)
fac_scored = fac_all.withColumn("completeness", completeness)

# one row per unique_id: keep the most complete record (stable tie-break)
w_id = Window.partitionBy("unique_id").orderBy(F.desc("completeness"), F.asc("name"), F.asc("address_line1"))
fac_by_id = fac_scored.withColumn("rn", F.row_number().over(w_id)).filter("rn = 1").drop("rn")
n_by_id = fac_by_id.count()

# one row per normalized (name, address_line1): same facility under different unique_ids
fac_norm = (
    fac_by_id
    .withColumn("name_norm", std_name(F.col("name")))
    .withColumn("addr1_norm", std_name(F.col("address_line1")))
)
w_pair = Window.partitionBy("name_norm", "addr1_norm").orderBy(F.desc("completeness"), F.asc("unique_id"))
fac_dedup = (
    fac_norm
    .withColumn("rn_pair",
                F.when(F.col("name_norm").isNotNull() & F.col("addr1_norm").isNotNull(),
                       F.row_number().over(w_pair)))
    .filter("rn_pair IS NULL OR rn_pair = 1")
    .drop("rn_pair", "name_norm", "addr1_norm", "completeness")
)
n_dedup = fac_dedup.count()

print(f"Facilities rows (raw, from {FACILITIES_TABLE}): {n_raw:,}")
print(f"After unique_id dedup: {n_by_id:,}  (removed {n_raw - n_by_id:,})")
print(f"After (name, address) dedup: {n_dedup:,}  (removed {n_by_id - n_dedup:,})")

# COMMAND ----------

# DBTITLE 1,Step 4b — Clean locations and merged flags
# ---- 4b. clean locations ----
f1 = (
    fac_dedup
    .withColumn("addr_text",
                F.lower(F.concat_ws(" ", F.col("address_line1"), F.col("address_line2"), F.col("address_line3"))))
    # country / countryCode: flag anything that is not India / IN (never drop)
    .withColumn("country_std", std_name(F.col("address_country")))
    .withColumn("country_code_std",
                F.when(std_name(F.col("address_countryCode")).isNotNull(),
                       F.upper(std_name(F.col("address_countryCode")))))
    .withColumn("non_india_country",
                (F.col("country_std").isNotNull() & (F.col("country_std") != "india"))
                | (F.col("country_code_std").isNotNull() & (F.col("country_code_std") != "IN")))
    # PIN: normalize the zip field, else extract from the address lines
    .withColumn("zip_digits",
                F.regexp_replace(
                    F.regexp_replace(F.trim(F.coalesce(F.col("address_zipOrPostcode"), F.lit(""))), r"\.0$", ""),
                    r"[^0-9]", ""))
    .withColumn("pin_from_zip", F.when(F.col("zip_digits").rlike(PIN_REGEX), F.col("zip_digits")))
    .withColumn("pins_in_addr",
                F.array_distinct(F.filter(
                    F.transform(F.regexp_extract_all(F.col("addr_text"), F.lit(PIN_IN_TEXT), 0),
                                lambda x: F.regexp_replace(x, r"\s+", "")),
                    lambda x: x.rlike(PIN_REGEX))))
    .withColumn("pins_present",
                F.array_distinct(F.concat(
                    F.col("pins_in_addr"),
                    F.when(F.col("pin_from_zip").isNotNull(), F.array(F.col("pin_from_zip"))).otherwise(F.array()))))
    .withColumn("pin_from_addr", F.try_element_at(F.col("pins_in_addr"), F.lit(1)))
    .withColumn("clean_pin", F.coalesce("pin_from_zip", "pin_from_addr"))
    .withColumn("pin_source",
                F.when(F.col("pin_from_zip").isNotNull(), F.lit("zip_field"))
                 .when(F.col("pin_from_addr").isNotNull(), F.lit("extracted_from_address"))
                 .otherwise(F.lit("none")))
    # names: same standardization as Step 1
    .withColumn("city_std", std_name(F.col("address_city")))
    .withColumn("state_std", std_state(F.col("address_stateOrRegion")))
    # coordinates: reconcile the GeoJSON string with the doubles
    .withColumn("json_lon", F.when(F.regexp_extract(F.col("coordinates"), JSON_POINT, 1) != "",
                                   F.regexp_extract(F.col("coordinates"), JSON_POINT, 1).cast("double")))
    .withColumn("json_lat", F.when(F.regexp_extract(F.col("coordinates"), JSON_POINT, 2) != "",
                                   F.regexp_extract(F.col("coordinates"), JSON_POINT, 2).cast("double")))
    .withColumn("coords_disagree",
                F.col("latitude").isNotNull() & F.col("json_lat").isNotNull()
                & ((F.abs(F.col("latitude") - F.col("json_lat")) > 1e-4)
                   | (F.abs(F.col("longitude") - F.col("json_lon")) > 1e-4)))
    .withColumn("lat_filled", F.coalesce("latitude", "json_lat"))
    .withColumn("lon_filled", F.coalesce("longitude", "json_lon"))
    .withColumn("fac_likely_swapped",
                F.col("lat_filled").between(LON_MIN, LON_MAX) & F.col("lon_filled").between(LAT_MIN, LAT_MAX))
    .withColumn("fac_lat", F.when(F.col("fac_likely_swapped"), F.col("lon_filled")).otherwise(F.col("lat_filled")))
    .withColumn("fac_lon", F.when(F.col("fac_likely_swapped"), F.col("lat_filled")).otherwise(F.col("lon_filled")))
    .withColumn("fac_lat", F.when(in_india(F.col("fac_lat"), F.col("fac_lon")), F.col("fac_lat")))
    .withColumn("fac_lon", F.when(in_india(F.col("fac_lat"), F.col("fac_lon")), F.col("fac_lon")))
)

# ---- possible_merged_record evidence ----
# (i) known places (districts) named in the address text
#     drop district names that are substrings of other district names (e.g. 'warangal' vs
#     'warangal rural') so one mention doesn't count as two places
dn_rows = (spark.table(PLACE_LOOKUP_TABLE)
           .filter("place_type = 'district' AND length(place_name) >= 4")
           .select("place_name").distinct().collect())
dn_names = sorted({r["place_name"] for r in dn_rows})
district_gazetteer = [n for n in dn_names if not any(n != o and n in o for o in dn_names)]
gazetteer_df = spark.createDataFrame([(n,) for n in district_gazetteer], "place_name string")

places_in_addr = (
    f1.select("unique_id", "addr_text")
    .join(F.broadcast(gazetteer_df), F.instr(F.col("addr_text"), F.col("place_name")) > 0, "inner")
    .groupBy("unique_id").agg(F.collect_set("place_name").alias("places_in_addr"))
)

# (ii) states named in the address text
state_names = [r["state_std"] for r in
               pincode_clean_df.filter("state_std IS NOT NULL").select("state_std").distinct().collect()]
STATE_PATTERN = "(" + "|".join(sorted(state_names)) + ")"

f2 = (
    f1.join(places_in_addr, "unique_id", "left")
    .withColumn("states_in_addr", F.array_distinct(F.regexp_extract_all(F.col("addr_text"), F.lit(STATE_PATTERN), 0)))
    .withColumn("n_places_in_addr", F.coalesce(F.size("places_in_addr"), F.lit(0)))
)

# (iii) facility coordinates >100 km from the PIN centroid
pin_cent = (spark.table(PIN_LOOKUP_TABLE)
            .select(F.col("pincode_str").alias("pc_pin"),
                    F.col("centroid_lat").alias("pc_lat"), F.col("centroid_lon").alias("pc_lon")))
f3 = f2.join(pin_cent, F.col("clean_pin") == F.col("pc_pin"), "left")

facilities_clean = (
    f3
    .withColumn("dist_to_pin_km",
                F.when(F.col("fac_lat").isNotNull() & F.col("pc_lat").isNotNull(),
                       haversine_km(F.col("fac_lat"), F.col("fac_lon"), F.col("pc_lat"), F.col("pc_lon"))))
    .withColumn("possible_merged_record",
                (F.size("pins_present") >= 2)
                | (F.col("n_places_in_addr") >= 2)
                | ((F.size("states_in_addr") >= 2)
                   | ((F.size("states_in_addr") == 1) & F.col("state_std").isNotNull()
                      & ~F.expr("array_contains(states_in_addr, state_std)")))
                | F.coalesce(F.col("dist_to_pin_km") > 100, F.lit(False)))
    .select(
        "unique_id", "name",
        "address_line1", "address_line2", "address_line3",
        "address_city", "address_stateOrRegion", "address_zipOrPostcode",
        "address_country", "address_countryCode", "coordinates",
        "clean_pin", "pin_source", "city_std", "state_std",
        "non_india_country", "coords_disagree", "fac_likely_swapped",
        "fac_lat", "fac_lon", "dist_to_pin_km", "possible_merged_record",
        F.size("pins_present").alias("n_pins_in_address"),
        F.col("n_places_in_addr").alias("n_places_in_address"))
)
write_replace(facilities_clean, FACILITIES_CLEAN_TABLE)

n_fc = facilities_clean.count()
print(f"Wrote {FACILITIES_CLEAN_TABLE}: {n_fc:,} rows (one per unique_id after notebook 02's dedup rules)")
print(f"\nRows with a non-India / non-IN country value (flagged, kept): {facilities_clean.filter('non_india_country').count():,}")
display(facilities_clean.groupBy("pin_source").count().orderBy(F.desc("count")))
print(f"Facility coordinates usable (in India bounds, after reconciliation): {facilities_clean.filter('fac_lat IS NOT NULL').count():,}")
print(f"Rows where the coordinates string disagrees with the lat/lon doubles: {facilities_clean.filter('coords_disagree').count():,}")
print(f"Rows with likely-swapped facility coordinates (repaired + flagged): {facilities_clean.filter('fac_likely_swapped').count():,}")
print(f"Rows flagged possible_merged_record: {facilities_clean.filter('possible_merged_record').count():,}")

# COMMAND ----------

# DBTITLE 1,Step 5 — Tiered join
# MAGIC %md
# MAGIC ## Step 5 — Join to `pin_lookup` (match tiers, best first)
# MAGIC
# MAGIC | Tier | `match_method` | Condition |
# MAGIC |---|---|---|
# MAGIC | 1 | `tier1_pin_city_state` | PIN valid, city matches one of the PIN's districts or office names, state matches |
# MAGIC | 2 | `tier2_pin_state` | PIN valid, state matches (city missing or not matched) |
# MAGIC | 3 | `tier3_pin_state_conflict` | PIN valid but state conflicts (PIN kept, `conflict_state = true`) |
# MAGIC | 4 | `tier4_city_state_place` | No valid PIN; city + state found in `place_lookup` |
# MAGIC | 5 | `tier5_no_match` | Nothing matched |
# MAGIC
# MAGIC A format-valid PIN absent from the directory cannot support tiers 1–3 and falls through to tier 4/5. When a name matches several place rows, district rows are preferred over office rows, then the smallest radius. Assertions: one row per `unique_id` and row count = `facilities_clean`.

# COMMAND ----------

# DBTITLE 1,Step 5 — Tiered join
facilities_clean_df = spark.table(FACILITIES_CLEAN_TABLE)
n_fc = facilities_clean_df.count()

pin_lk = spark.table(PIN_LOOKUP_TABLE).select(
    "pincode_str", "primary_district", "primary_state", "district_share",
    "n_districts", "is_ambiguous", "centroid_lat", "centroid_lon", "coord_spread_km",
    "districts_set", "office_names")

# tier-4 candidate: city + state found in place_lookup (prefer district rows, then smallest radius)
place_match = (
    facilities_clean_df.select("unique_id", "city_std", "state_std")
    .join(spark.table(PLACE_LOOKUP_TABLE),
          (F.col("city_std") == F.col("place_name")) & (F.col("state_std") == F.col("primary_state")),
          "inner")
)
w_place = (Window.partitionBy("unique_id")
           .orderBy(F.desc(F.col("place_type") == "district"), F.asc("radius_km"), F.asc("place_name")))
place_match_one = (
    place_match.withColumn("rn", F.row_number().over(w_place)).filter("rn = 1")
    .select("unique_id",
            F.col("place_name").alias("pl_place_name"), F.col("place_type").alias("pl_place_type"),
            F.col("district").alias("pl_district"),
            F.col("centroid_lat").alias("pl_centroid_lat"), F.col("centroid_lon").alias("pl_centroid_lon"),
            F.col("radius_km").alias("pl_radius_km"))
)

fj = (
    facilities_clean_df
    .join(place_match_one, "unique_id", "left")
    .join(pin_lk, F.col("clean_pin") == F.col("pincode_str"), "left")
)

pin_matched = F.col("pincode_str").isNotNull()  # PIN actually found in the directory

located = (
    fj
    .withColumn("city_match", F.expr(
        "COALESCE(city_std IS NOT NULL AND "
        "(array_contains(districts_set, city_std) OR array_contains(office_names, city_std)), false)"))
    .withColumn("state_match", F.coalesce(F.col("state_std") == F.col("primary_state"), F.lit(False)))
    .withColumn("state_conflict",
                F.col("state_std").isNotNull() & F.col("primary_state").isNotNull()
                & (F.col("state_std") != F.col("primary_state")))
    .withColumn("match_method",
                F.when(pin_matched & F.col("city_match") & F.col("state_match"), F.lit("tier1_pin_city_state"))
                 .when(pin_matched & ~F.col("state_conflict"), F.lit("tier2_pin_state"))
                 .when(pin_matched, F.lit("tier3_pin_state_conflict"))
                 .when(F.col("pl_place_name").isNotNull(), F.lit("tier4_city_state_place"))
                 .otherwise(F.lit("tier5_no_match")))
    .withColumn("conflict_state", F.col("state_conflict"))
    .select(
        "unique_id", "name",
        "address_line1", "address_line2", "address_line3",
        "address_city", "address_stateOrRegion", "address_zipOrPostcode",
        "clean_pin", "pin_source", "city_std", "state_std",
        "non_india_country", "coords_disagree", "fac_likely_swapped",
        "fac_lat", "fac_lon", "dist_to_pin_km", "possible_merged_record",
        "primary_district", "primary_state", "district_share", "n_districts", "is_ambiguous",
        "centroid_lat", "centroid_lon", "coord_spread_km",
        "match_method", "conflict_state",
        "pl_place_name", "pl_place_type", "pl_district",
        "pl_centroid_lat", "pl_centroid_lon", "pl_radius_km")
)

# ---- assertions: one row per unique_id, row count preserved ----
n_out = located.count()
n_ids = located.select("unique_id").distinct().count()
assert n_out == n_fc, f"Row count changed: facilities_clean {n_fc} -> {n_out}"
assert n_ids == n_fc, f"Not one row per unique_id: {n_ids} distinct ids for {n_out} rows"
print(f"Assertions passed: {n_out:,} rows, {n_ids:,} distinct unique_ids (== facilities_clean {n_fc:,}).")

display(located.groupBy("match_method").count().orderBy(F.desc("count")))

# COMMAND ----------

# DBTITLE 1,Step 6 — Final location and error
# MAGIC %md
# MAGIC ## Step 6 — Final location and error estimate
# MAGIC
# MAGIC `final_lat`/`final_lon`: the facility's own coordinates if valid; otherwise the PIN centroid; otherwise the `place_lookup` centroid. `coord_source` records which.
# MAGIC
# MAGIC `location_error_km` is a **rough, reviewable estimate** of how far off the final point could be:
# MAGIC
# MAGIC | Situation | Error |
# MAGIC |---|---|
# MAGIC | Facility coords agree with the PIN (dist ≤ spread + 5 km) | 2 km |
# MAGIC | Facility coords disagree with the PIN | `dist_to_pin_km` |
# MAGIC | Facility coords, no PIN to cross-check | unknown (NULL) |
# MAGIC | PIN centroid | `coord_spread_km`, minimum 5 km |
# MAGIC | Place centroid | `radius_km` |
# MAGIC | Nothing | NULL |
# MAGIC
# MAGIC > **These values are assumptions to review, not measurements.** The 2 km and 5 km floors in particular are judgment calls.

# COMMAND ----------

# DBTITLE 1,Step 6 — Final location and error
located = (
    located
    .withColumn("coord_source",
                F.when(F.col("fac_lat").isNotNull() & F.col("fac_lon").isNotNull(), F.lit("facility"))
                 .when(F.col("centroid_lat").isNotNull() & F.col("centroid_lon").isNotNull(), F.lit("pin_centroid"))
                 .when(F.col("pl_centroid_lat").isNotNull() & F.col("pl_centroid_lon").isNotNull(), F.lit("place_centroid"))
                 .otherwise(F.lit("none")))
    .withColumn("final_lat", F.coalesce("fac_lat", "centroid_lat", "pl_centroid_lat"))
    .withColumn("final_lon", F.coalesce("fac_lon", "centroid_lon", "pl_centroid_lon"))
    .withColumn("location_error_km",
                F.when(F.col("fac_lat").isNotNull() & F.col("centroid_lat").isNotNull(),
                       F.when(F.col("dist_to_pin_km") <= F.coalesce(F.col("coord_spread_km"), F.lit(0.0)) + 5,
                              F.lit(2.0))
                        .otherwise(F.col("dist_to_pin_km")))
                 .when(F.col("coord_source") == "pin_centroid",
                       F.greatest(F.coalesce(F.col("coord_spread_km"), F.lit(5.0)), F.lit(5.0)))
                 .when(F.col("coord_source") == "place_centroid", F.col("pl_radius_km"))
                 .otherwise(F.lit(None).cast("double")))
)

display(located.groupBy("coord_source").count().orderBy(F.desc("count")))
print("\nlocation_error_km — summary:")
display(located.select("location_error_km").summary("count", "min", "10%", "25%", "50%", "75%", "90%", "max"))
print("location_error_km — distribution:")
display(
    located
    .withColumn("error_bucket",
                F.when(F.col("location_error_km").isNull(), F.lit("unknown"))
                 .when(F.col("location_error_km") < 5, F.lit("< 5 km"))
                 .when(F.col("location_error_km") < 10, F.lit("5-10 km"))
                 .when(F.col("location_error_km") < 25, F.lit("10-25 km"))
                 .when(F.col("location_error_km") < 50, F.lit("25-50 km"))
                 .when(F.col("location_error_km") < 100, F.lit("50-100 km"))
                 .otherwise(F.lit("100+ km")))
    .groupBy("error_bucket")
    .agg(F.count("*").alias("n_facilities"), F.min("location_error_km").alias("min_err"))
    .orderBy(F.col("min_err").asc_nulls_last())
)

# COMMAND ----------

# DBTITLE 1,Step 7 — Uncertainty score
# MAGIC %md
# MAGIC ## Step 7 — Uncertainty score (0 = confident, 1 = unknown)
# MAGIC
# MAGIC `location_uncertainty = 0.30·match_penalty + 0.15·district_ambiguity + 0.40·error_component + 0.15·merge_penalty`
# MAGIC
# MAGIC * **match_penalty** — tier 1 = 0, tier 2 = 0.2, tier 3 = 0.6, tier 4 = 0.7, tier 5 = 1
# MAGIC * **district_ambiguity** — `1 − district_share` (0 when tier 1 matched a single district; tier 4 matched a district exactly = 0; tier 5 district unknown = 1)
# MAGIC * **error_component** — `min(location_error_km / 50, 1)`; 1 when the error is unknown
# MAGIC * **merge_penalty** — 1 when `possible_merged_record`, else 0
# MAGIC
# MAGIC `uncertainty_tier`: low (< 0.2), medium (0.2–0.5), high (> 0.5). All component columns are kept so the score can be explained.

# COMMAND ----------

# DBTITLE 1,Step 7 — Uncertainty score
match_penalty = (
    F.when(F.col("match_method") == "tier1_pin_city_state", F.lit(0.0))
     .when(F.col("match_method") == "tier2_pin_state", F.lit(0.2))
     .when(F.col("match_method") == "tier3_pin_state_conflict", F.lit(0.6))
     .when(F.col("match_method") == "tier4_city_state_place", F.lit(0.7))
     .otherwise(F.lit(1.0))
)
district_ambiguity = (
    F.when((F.col("match_method") == "tier1_pin_city_state")
           & ~F.coalesce(F.col("is_ambiguous"), F.lit(False)), F.lit(0.0))
     .when(F.col("district_share").isNotNull(), F.lit(1.0) - F.col("district_share"))
     .when(F.col("match_method") == "tier4_city_state_place", F.lit(0.0))
     .otherwise(F.lit(1.0))
)
error_component = (
    F.when(F.col("location_error_km").isNotNull(),
           F.least(F.col("location_error_km") / F.lit(50.0), F.lit(1.0)))
     .otherwise(F.lit(1.0))
)
merge_penalty = F.when(F.col("possible_merged_record"), F.lit(1.0)).otherwise(F.lit(0.0))

located = (
    located
    .withColumn("match_penalty", match_penalty)
    .withColumn("district_ambiguity", district_ambiguity)
    .withColumn("error_component", error_component)
    .withColumn("merge_penalty", merge_penalty)
    .withColumn("location_uncertainty",
                0.30 * F.col("match_penalty") + 0.15 * F.col("district_ambiguity")
                + 0.40 * F.col("error_component") + 0.15 * F.col("merge_penalty"))
    .withColumn("uncertainty_tier",
                F.when(F.col("location_uncertainty") < 0.2, F.lit("low"))
                 .when(F.col("location_uncertainty") <= 0.5, F.lit("medium"))
                 .otherwise(F.lit("high")))
)
display(located.groupBy("uncertainty_tier").count().orderBy(F.desc("count")))

# COMMAND ----------

# DBTITLE 1,Step 8 — Plain-language evidence
# MAGIC %md
# MAGIC ## Step 8 — Plain-language evidence
# MAGIC
# MAGIC `location_evidence`: one or two short sentences per facility describing where the location came from and how much to trust it. `location_warnings`: an array of flags — `state_conflict`, `possible_merged_record`, `non_india`, `coords_disagree`, `ambiguous_pin`, `no_location`.

# COMMAND ----------

# DBTITLE 1,Step 8 — Plain-language evidence
km_str = lambda c: F.format_number(F.col(c), 0)

main_sentence = (
    F.when(F.col("coord_source") == "facility",
           F.when(F.col("dist_to_pin_km").isNotNull(),
                  F.concat(F.lit("Location from the facility's own coordinates; "),
                           km_str("dist_to_pin_km"), F.lit(" km from the center of PIN "),
                           F.coalesce(F.col("clean_pin"), F.lit("")),
                           F.lit(" ("), F.coalesce(F.col("primary_district"), F.lit("district unknown")),
                           F.lit(", "), F.coalesce(F.col("primary_state"), F.lit("state unknown")),
                           F.lit(").")))
           .otherwise(F.lit("Location from the facility's own coordinates; no PIN was available to cross-check.")))
     .when(F.col("coord_source") == "pin_centroid",
           F.concat(F.lit("No coordinates on record; placed at the center of PIN "),
                    F.coalesce(F.col("clean_pin"), F.lit("")),
                    F.lit(", which covers about "),
                    F.format_number(F.coalesce(F.col("coord_spread_km"), F.lit(5.0)), 0),
                    F.lit(" km.")))
     .when(F.col("coord_source") == "place_centroid",
           F.concat(F.lit("Placed at the center of "),
                    F.coalesce(F.col("city_std"), F.col("pl_place_name"), F.lit("the matched place")),
                    F.lit(", "), F.coalesce(F.col("state_std"), F.lit("")),
                    F.lit(" from the post-office directory; within about "),
                    F.format_number(F.coalesce(F.col("pl_radius_km"), F.lit(0.0)), 0),
                    F.lit(" km.")))
     .otherwise(F.lit("No reliable location could be determined from the address or the directory."))
)

caveat_sentence = (
    F.when(F.col("conflict_state"),
           F.concat(F.lit("Recorded state ("), F.coalesce(F.col("state_std"), F.lit("?")),
                    F.lit(") conflicts with the PIN directory ("),
                    F.coalesce(F.col("primary_state"), F.lit("?")), F.lit(").")))
     .when(F.col("possible_merged_record"),
           F.lit("The address may combine multiple facilities."))
     .when(F.col("match_method") == "tier4_city_state_place",
           F.lit("Matched by city and state only; no usable PIN."))
     .when(F.coalesce(F.col("is_ambiguous"), F.lit(False)) & (F.col("coord_source") == "pin_centroid"),
           F.concat(F.lit("PIN "), F.coalesce(F.col("clean_pin"), F.lit("")), F.lit(" spans "),
                    F.coalesce(F.col("n_districts").cast("string"), F.lit("?")), F.lit(" districts.")))
     .when(F.col("location_error_km").isNotNull(),
           F.concat(F.lit("Likely within about "), km_str("location_error_km"), F.lit(" km.")))
     .otherwise(F.lit(None).cast("string"))
)

location_warnings = F.filter(
    F.array(
        F.when(F.col("conflict_state"), F.lit("state_conflict")),
        F.when(F.col("possible_merged_record"), F.lit("possible_merged_record")),
        F.when(F.col("non_india_country"), F.lit("non_india")),
        F.when(F.col("coords_disagree"), F.lit("coords_disagree")),
        F.when(F.coalesce(F.col("is_ambiguous"), F.lit(False)) & F.col("clean_pin").isNotNull(), F.lit("ambiguous_pin")),
        F.when(F.col("coord_source") == "none", F.lit("no_location")),
    ),
    lambda x: x.isNotNull(),
)

located = (
    located
    .withColumn("location_evidence", F.concat_ws(" ", main_sentence, caveat_sentence))
    .withColumn("location_warnings", location_warnings)
)

display(located.select("name", "coord_source", "location_evidence", "location_warnings").limit(5))

# COMMAND ----------

# DBTITLE 1,Output
# MAGIC %md
# MAGIC ## Output — `facilities_located`
# MAGIC
# MAGIC One row per facility with the cleaned pin/city/state, matched district, final coordinates, all score components, and the plain-language evidence.

# COMMAND ----------

# DBTITLE 1,Write facilities_located
facilities_located = located.select(
    "unique_id", "name",
    F.col("clean_pin").alias("pin"),
    F.col("city_std").alias("city"),
    F.col("state_std").alias("state"),
    F.coalesce("primary_district", F.col("pl_district"), F.col("pl_place_name")).alias("primary_district"),
    "final_lat", "final_lon", "coord_source", "pin_source", "match_method",
    "conflict_state", "dist_to_pin_km", "location_error_km", "possible_merged_record",
    "match_penalty", "district_ambiguity", "error_component", "merge_penalty",
    "location_uncertainty", "uncertainty_tier",
    "location_evidence", "location_warnings")

write_replace(facilities_located, FACILITIES_LOCATED_TABLE)

located_tbl = spark.table(FACILITIES_LOCATED_TABLE)
n_written = located_tbl.count()
assert n_written == n_fc, f"written {n_written} != facilities_clean {n_fc}"
print(f"Wrote {n_written:,} rows to {FACILITIES_LOCATED_TABLE}")

# COMMAND ----------

# DBTITLE 1,Summary — counts
print("Counts by match_method:")
display(located_tbl.groupBy("match_method").count().orderBy(F.desc("count")))

print("Counts by coord_source:")
display(located_tbl.groupBy("coord_source").count().orderBy(F.desc("count")))

print("Counts by uncertainty_tier:")
display(located_tbl.groupBy("uncertainty_tier").count().orderBy(F.desc("count")))

print(f"Facilities flagged as possible merged records: {located_tbl.filter('possible_merged_record').count():,}")
display(located_tbl.filter("size(location_warnings) > 0")
        .select(F.explode("location_warnings").alias("warning"))
        .groupBy("warning").count().orderBy(F.desc("count")))

# COMMAND ----------

# DBTITLE 1,Histogram — location_error_km
import plotly.express as px

hist_df = located_tbl.select("location_error_km").filter("location_error_km IS NOT NULL").toPandas()
fig = px.histogram(
    hist_df, x="location_error_km", nbins=60,
    title=f"Estimated location error (km) — n={len(hist_df):,} facilities with a location",
    labels={"location_error_km": "km"},
)
fig.show()

# COMMAND ----------

# DBTITLE 1,Top 20 least certain facilities
# 20 least certain facilities, with their raw address fields for manual review
raw_addr = spark.table(FACILITIES_CLEAN_TABLE).select(
    "unique_id", "address_line1", "address_line2", "address_line3",
    "address_city", "address_stateOrRegion", "address_zipOrPostcode", "coordinates")

top20 = (
    located_tbl.orderBy(F.desc("location_uncertainty"), F.asc("name")).limit(20)
    .join(raw_addr, "unique_id", "left")
    .select(
        "name", "location_uncertainty", "uncertainty_tier", "match_method", "coord_source",
        "location_error_km", "location_evidence", "location_warnings",
        "pin", "city", "state", "primary_district",
        "address_line1", "address_line2", "address_line3",
        "address_city", "address_stateOrRegion", "address_zipOrPostcode")
)
display(top20)

# COMMAND ----------

# DBTITLE 1,How location confidence is decided
# MAGIC %md
# MAGIC ## How location confidence is decided
# MAGIC
# MAGIC *(Reusable explanation for the "how did you decide this location?" question in the final demo.)*
# MAGIC
# MAGIC **1. Where the point comes from (`coord_source`).** In order of trust: the facility's own coordinates (`facility`), the centroid of its PIN's post offices (`pin_centroid`), the centroid of the place (district or post office) matched by city + state (`place_centroid`), or nothing (`none`).
# MAGIC
# MAGIC **2. How the address was resolved (`match_method`).**
# MAGIC - **Tier 1 — PIN + city + state:** the PIN is valid, the city appears among that PIN's districts or post offices, and the state matches the directory. Strongest evidence.
# MAGIC - **Tier 2 — PIN + state:** the city is missing or wasn't among the PIN's places, but the state agrees.
# MAGIC - **Tier 3 — state conflict:** the PIN is valid but the recorded state contradicts the directory; the PIN is kept and `conflict_state = true`.
# MAGIC - **Tier 4 — city + state only:** no usable PIN, but the city and state match a place in the post-office directory.
# MAGIC - **Tier 5 — no match:** neither the PIN nor city+state resolved.
# MAGIC
# MAGIC **3. How far off the point might be (`location_error_km`).** A rough, reviewable estimate: facility coordinates that agree with their PIN (within the PIN's spread + 5 km) → **2 km**; facility coordinates that disagree → the distance to the PIN centroid; PIN centroid → the PIN's radius (**at least 5 km**); place centroid → the place's radius; nothing → unknown. *These numbers are assumptions to review, not measurements.*
# MAGIC
# MAGIC **4. The score.** `location_uncertainty = 0.30×match_penalty + 0.15×district_ambiguity + 0.40×error_component + 0.15×merge_penalty`, from 0 (confident) to 1 (unknown):
# MAGIC - match_penalty: tier 1→0, 2→0.2, 3→0.6, 4→0.7, 5→1
# MAGIC - district_ambiguity: 1 − the share of the PIN's offices in the primary district (0 when tier 1 hit a single district)
# MAGIC - error_component: the error in km capped at 50; 1 (worst) when the error is unknown
# MAGIC - merge_penalty: 1 when the record looks like two facilities merged into one
# MAGIC
# MAGIC Tiers: **low** < 0.2, **medium** 0.2–0.5, **high** > 0.5.
# MAGIC
# MAGIC **5. Warnings and evidence.** `location_warnings` flags the specific problems — `state_conflict`, `possible_merged_record`, `non_india`, `coords_disagree`, `ambiguous_pin`, `no_location` — and `location_evidence` states, in plain English, where each facility's location came from and how much to trust it.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'raw' AS source, n_districts, COUNT(*) AS n_pins
# MAGIC FROM (
# MAGIC   SELECT pincode, COUNT(DISTINCT district) AS n_districts
# MAGIC   FROM databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.india_post_pincode_directory
# MAGIC   GROUP BY pincode
# MAGIC ) t
# MAGIC GROUP BY n_districts
# MAGIC
# MAGIC UNION ALL
# MAGIC
# MAGIC SELECT 'clean' AS source, n_districts, COUNT(*) AS n_pins
# MAGIC FROM (
# MAGIC   SELECT pincode_str, COUNT(DISTINCT district_std) AS n_districts
# MAGIC   FROM workspace.datathon_2026.pincode_clean
# MAGIC   GROUP BY pincode_str
# MAGIC ) t
# MAGIC GROUP BY n_districts
# MAGIC
# MAGIC ORDER BY source, n_districts;