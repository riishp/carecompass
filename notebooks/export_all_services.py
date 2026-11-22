# Databricks notebook source
# CareCompass: export every service term listed by any facility into one CSV.
#
# What it does
#   - Reads the list fields (specialties, procedure, equipment, capability) from facilities_silver,
#     keeping only facilities that made it into gold_facility_search (no duplicates / junk rows).
#   - Explodes every list item, normalises the text (lowercase, trimmed, single spaces, no stray punctuation).
#   - Counts how many distinct facilities list each term, per field.
#   - Flags which of the checked services (dialysis, c-section, NICU, ...) each term matches.
#   - Writes one CSV: one row per (term, field).
#
# Run in a notebook on the `datathon` workspace. Source tables are only read.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

CATALOG_SCHEMA = "workspace.datathon_2026"
SILVER = f"{CATALOG_SCHEMA}.facilities_silver"
GOLD = f"{CATALOG_SCHEMA}.gold_facility_search"
VOLUME = "exports"                      # created below if missing
OUT_DIR = f"/Volumes/workspace/datathon_2026/{VOLUME}"
OUT_CSV = f"{OUT_DIR}/all_services.csv"
MIN_FACILITIES = 1                      # raise to 3+ to hide one-off noise

LIST_FIELDS = ["specialties", "procedure", "equipment", "capability"]

# Checked services, with illustrative patterns. Replace them with the core regexes in notebook 04 if they differ.
CHECKED_SERVICES = {
    "dialysis":        r"dialys|haemodialys|hemodialys|\bcapd\b",
    "c_section":       r"c[\s\-]?section|caesarean|cesarean|lscs",
    "nicu":            r"\bnicu\b|neonatal intensive|newborn intensive",
    "cardiac_surgery": r"cardiac surg|heart surg|cabg|bypass surg|open heart|valve replace",
    "angioplasty":     r"angioplast|\bptca\b|stent|cath(eterisation|eterization| lab)",
    "mri":             r"\bmri\b|magnetic resonance",
    "burn_care":       r"\bburns?\b",
    "trauma":          r"trauma|casualty|accident|emergency surg",
}

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS workspace.datathon_2026.{VOLUME}")

gold_ids = spark.table(GOLD).select("unique_id").distinct()
silver = spark.table(SILVER).join(gold_ids, "unique_id", "inner")

present = [c for c in LIST_FIELDS if c in silver.columns]
missing = [c for c in LIST_FIELDS if c not in silver.columns]
print("Fields used:", present, "| missing (skipped):", missing)
assert present, "None of the list fields were found in facilities_silver; check the column names."

def as_array(col_name):
    """Return the column as array<string>, whether it is stored as an array or as JSON text."""
    dtype = silver.schema[col_name].dataType
    if isinstance(dtype, ArrayType):
        return F.col(col_name).cast("array<string>")
    return F.from_json(F.col(col_name).cast("string"), ArrayType(StringType()))

items = None
for c in present:
    part = (silver
            .select("unique_id", F.explode_outer(as_array(c)).alias("raw_term"))
            .where(F.col("raw_term").isNotNull())
            .withColumn("field", F.lit(c)))
    items = part if items is None else items.unionByName(part)

norm = F.lower(F.trim(F.col("raw_term")))
norm = F.regexp_replace(norm, r"[\"'\[\]{}()]", "")
norm = F.regexp_replace(norm, r"\s+", " ")
norm = F.regexp_replace(norm, r"^[\s\-\.,;:]+|[\s\-\.,;:]+$", "")
items = items.withColumn("term", norm).where(F.length("term") > 1)

# COMMAND ----------

summary = (items.groupBy("term", "field")
           .agg(F.countDistinct("unique_id").alias("facility_count"),
                F.first("raw_term").alias("example_raw_text"))
           .where(F.col("facility_count") >= MIN_FACILITIES))

# How many facilities use this term across all fields (one facility counted once)
total = (items.groupBy("term")
         .agg(F.countDistinct("unique_id").alias("facility_count_any_field"),
              F.array_sort(F.collect_set("field")).alias("fields")))
summary = (summary.join(total, "term", "left")
           .withColumn("fields", F.concat_ws("|", "fields")))

match_cols = []
for svc, pattern in CHECKED_SERVICES.items():
    col = f"matches_{svc}"
    summary = summary.withColumn(col, F.col("term").rlike(pattern))
    match_cols.append(col)

summary = summary.withColumn(
    "checked_service",
    F.concat_ws("|", *[F.when(F.col(f"matches_{s}"), F.lit(s)) for s in CHECKED_SERVICES]))
summary = summary.withColumn("in_checked_catalog", F.col("checked_service") != "")

result = (summary
          .select("term", "field", "facility_count", "facility_count_any_field", "fields",
                  "checked_service", "in_checked_catalog", "example_raw_text")
          .orderBy(F.desc("facility_count_any_field"), "term", "field"))

# COMMAND ----------

pdf = result.toPandas()
pdf.to_csv(OUT_CSV, index=False)

print(f"Wrote {len(pdf):,} rows to {OUT_CSV}")
print(f"Distinct terms: {pdf['term'].nunique():,}")
print(pdf.groupby("field")["term"].nunique().rename("distinct_terms"))
print("\nTerms matching a checked service:",
      pdf.loc[pdf["in_checked_catalog"], "term"].nunique())
display(result.limit(200))

# Download to your laptop afterwards with:
#   databricks -p datathon fs cp dbfs:/Volumes/workspace/datathon_2026/exports/all_services.csv .
