# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 09 — Gold: tables shaped for the app and the analysis
# MAGIC
# MAGIC **Input:** `workspace.datathon_2026.facilities_silver` (08) and `nfhs_clean` (06).
# MAGIC **This notebook only selects, renames and counts — no scoring, no conclusions.** Analysis happens on top of these tables.
# MAGIC
# MAGIC | table | grain | serves |
# MAGIC |---|---|---|
# MAGIC | `gold_facility_search` | one row per real facility | **Facility finder** — "I need *service* near *location*" |
# MAGIC | `gold_district_summary` | one row per NFHS district (705) | **Care-desert analysis** — health need (NFHS) next to care supply (facility counts) |
# MAGIC
# MAGIC **Rules**
# MAGIC - Facilities: only `is_primary_record` (one per possible-duplicate group), not `country_suspect`, and with a location (the 4 rows without one are text fragments from column-shifted rows).
# MAGIC - District counts: every NFHS district appears, **including districts with zero facilities** — those are the most important rows for care-desert analysis.
# MAGIC - Counts are given twice: all facilities, and only those with **high/medium location confidence**, because a low-confidence facility may be in the wrong district.
# MAGIC - NFHS has no population figure, so counts are raw counts, not per-capita.

# COMMAND ----------

# DBTITLE 1,0. Setup
from datetime import datetime, timezone
from pyspark.sql import functions as F

TGT = "workspace.datathon_2026"
SILVER = f"{TGT}.facilities_silver"
NFHS = f"{TGT}.nfhs_clean"
SEARCH = f"{TGT}.gold_facility_search"
DISTRICT = f"{TGT}.gold_district_summary"
AUDIT = f"{TGT}.cleaning_audit"
for t, who in [(SILVER, "08_Silver_Facilities"), (NFHS, "06_NFHS_Cleaning")]:
    assert spark.catalog.tableExists(t), f"{t} is missing — run '{who}' first"

RUN_ID = "gold-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
audit_rows = []
def log(step, rule, n, note=""):
    audit_rows.append((RUN_ID, f"gold-{step}", rule, int(n), note))
    print(f"[{step}] {rule}: {n:,} {note}")

silver = spark.read.table(SILVER)
SILVER_N = silver.count()
SILVER_RUN = silver.select(F.max("silver_run_id")).first()[0]
log(0, "silver rows in", SILVER_N, SILVER_RUN)

# COMMAND ----------

# DBTITLE 1,1. gold_facility_search
# MAGIC %md
# MAGIC ## 1. `gold_facility_search` — what the app reads
# MAGIC - **Service:** `specialty_codes` (filter) / `specialty_labels` (display) / `specialties_other`, plus `procedures`, `equipment`, `capabilities` for richer text search.
# MAGIC - **Location:** `latitude`, `longitude`, `location_confidence`, `location_error_km`, `location_note` (plain-language explanation). Show low-confidence results as approximate.
# MAGIC - **Contact:** `phone` (junk numbers already removed), `email`, `website`.
# MAGIC - `search_text`: lower-case name + city + district + services, for simple keyword search.

# COMMAND ----------

base = silver.where("is_primary_record AND NOT country_suspect")
log(1, "excluded: non-primary duplicate records", silver.where("NOT is_primary_record").count())
log(1, "excluded: country_suspect (column-shifted) records", silver.where("country_suspect AND is_primary_record").count())
# rows with no coordinates, city or state are column-shifted text fragments (e.g. "located in Mandaiveli") — not facilities
no_loc = base.where("latitude IS NULL")
log(1, "excluded: no location at all (text fragments from shifted rows)", no_loc.count(),
    "; ".join(r[0][:40] for r in no_loc.select("name").collect()))
search_base = base.where("latitude IS NOT NULL")

def arr_text(c):
    return F.coalesce(F.concat_ws(" ", c), F.lit(""))

search = search_base.select(
    "unique_id", "name", "facility_type", "operator_type",
    "city", "state", "pincode",
    F.initcap("primary_district").alias("district"), "nfhs_state", "nfhs_district",
    "latitude", "longitude", "location_vote", "location_confidence", F.round("location_error_km", 1).alias("location_error_km"),
    F.col("location_evidence").alias("location_note"), "location_warnings",
    "specialty_codes", "specialty_labels", "specialties_other", "n_specialty_codes",
    F.col("procedure").alias("procedures"), "equipment", F.col("capability").alias("capabilities"),
    F.col("official_phone").alias("phone"), "phone_numbers", "email", F.col("officialWebsite").alias("website"),
    "capacity_beds",
    F.when(F.col("doctor_count_plausible"), F.col("number_doctors")).alias("number_doctors"),
    "year_established", "description",
).withColumn("search_text", F.lower(F.concat_ws(" ",
    F.col("name"), F.col("city"), F.col("district"), F.col("state"), F.col("facility_type"),
    arr_text(F.col("specialty_labels")), arr_text(F.col("specialties_other")),
    arr_text(F.col("procedures")), arr_text(F.col("capabilities"))))
).withColumn("gold_run_id", F.lit(RUN_ID)).withColumn("silver_run_id", F.lit(SILVER_RUN))

search.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SEARCH)
SEARCH_N = spark.read.table(SEARCH).count()
log(1, "gold_facility_search rows written", SEARCH_N, SEARCH)
for r in spark.read.table(SEARCH).groupBy("location_confidence").count().collect():
    log(1, f"search rows with location_confidence = {r['location_confidence']}", r["count"])

# COMMAND ----------

# DBTITLE 1,2. gold_district_summary
# MAGIC %md
# MAGIC ## 2. `gold_district_summary` — need next to supply, per district
# MAGIC **Supply** (from facilities, linked through `nfhs_state` + `nfhs_district`): total, hospitals, clinics, public, and counts for key services.
# MAGIC Each count also exists as `*_confident` (high/medium location confidence only).
# MAGIC **Need** (from NFHS-5, renamed to readable names): insurance, institutional births, vaccination, anaemia, child stunting, blood pressure, blood sugar, sanitation, screening.
# MAGIC `n_small_sample_values` / `n_suppressed_values` show how reliable a district's NFHS numbers are.

# COMMAND ----------

# key services -> standard specialty codes (from 07); a facility counts if it has any of the codes
SERVICES = {
    "emergency": ["emergencyMedicine", "criticalCareMedicine"],
    "maternity": ["gynecologyAndObstetrics", "obstetricsAndMaternityCare", "maternalFetalMedicineOrPerinatology", "gynecology"],
    "pediatrics": ["pediatrics", "neonatologyPerinatalMedicine", "pediatricSurgery", "pediatricCriticalCareMedicine"],
    "cardiology": ["cardiology", "cardiacSurgery", "cardiothoracicSurgery", "pediatricCardiology"],
    "dialysis_kidney": ["nephrology", "renalTransplantationUrology"],
    "cancer": ["medicalOncology", "surgicalOncology", "radiationOncology", "oncology", "gynecologicalOncology", "pediatricHematologyOncology"],
    "mental_health": ["psychiatry", "addictionPsychiatry", "childAndAdolescentPsychiatry", "neuropsychiatry"],
    "eye": ["ophthalmology", "cataractAndAnteriorSegmentSurgery", "retinaAndVitreoretinalOphthalmology"],
    "dental": ["dentistry", "generalDentistry", "pediatricDentistry", "endodontics", "orthodontics"],
    "diabetes": ["endocrinologyAndDiabetesAndMetabolism", "diabetology", "endocrinology"],
}

fac = base.where("nfhs_district IS NOT NULL").withColumn(
    "_conf", F.col("location_confidence").isin("high", "medium"))
aggs = [F.count("*").alias("n_facilities"),
        F.sum(F.col("_conf").cast("int")).alias("n_facilities_confident"),
        F.sum((F.col("facility_type") == "hospital").cast("int")).alias("n_hospitals"),
        F.sum((F.col("facility_type") == "clinic").cast("int")).alias("n_clinics"),
        F.sum((F.col("operator_type") == "public").cast("int")).alias("n_public_facilities"),
        F.sum(F.coalesce("capacity_beds", F.lit(0))).alias("known_beds"),
        F.sum((F.col("location_confidence") == "low").cast("int")).alias("n_facilities_low_location_confidence")]
for svc, codes in SERVICES.items():
    has = F.exists(F.coalesce("specialty_codes", F.array().cast("array<string>")), lambda c: c.isin(codes))
    aggs += [F.sum(has.cast("int")).alias(f"n_{svc}"), F.sum((has & F.col("_conf")).cast("int")).alias(f"n_{svc}_confident")]
supply = fac.groupBy(F.col("nfhs_state").alias("state"), F.col("nfhs_district").alias("district")).agg(*aggs)

NEED = {  # readable name -> nfhs_clean column
    "households_surveyed": "households_surveyed",
    "health_insurance_pct": "hh_member_covered_health_insurance_pct",
    "institutional_births_pct": "institutional_birth_5y_pct",
    "anc_4plus_visits_pct": "mothers_who_had_at_least_4_anc_visits_lb5y_pct",
    "children_fully_vaccinated_pct": "child_12_23m_fully_vaccinated_based_on_information_from_eit_pct",
    "children_stunted_pct": "child_u5_who_are_stunted_height_for_age_18_pct",
    "children_underweight_pct": "child_u5_who_are_underweight_weight_for_age_18_pct",
    "children_anaemic_pct": "child_6_59m_who_are_anaemic_lt_11_0_g_dl_22_pct",
    "women_anaemic_pct": "all_w15_49_who_are_anaemic_pct",
    "women_high_bp_pct": "w15_plus_with_high_bp_sys_gte_140_mmhg_and_or_dia_gte_90_mm_pct",
    "men_high_bp_pct": "m15_plus_with_high_bp_sys_gte_140_mmhg_and_or_dia_gte_90_mm_pct",
    "women_high_blood_sugar_pct": "w15_plus_with_high_or_very_high_gt_140_mg_dl_blood_sugar_or_pct",
    "men_high_blood_sugar_pct": "m15_plus_with_high_or_very_high_gt_140_mg_dl_blood_sugar_or_pct",
    "women_cervical_screened_pct": "women_age_30_49_years_ever_undergone_a_cervical_screen_pct",
    "improved_sanitation_pct": "hh_use_improved_sanitation_pct",
    "improved_water_pct": "hh_improved_water_pct",
    "women_literate_pct": "women_age_15_49_who_are_literate_pct",
    "out_of_pocket_per_public_delivery_inr": "average_out_of_pocket_expenditure_per_delivery_in_a_public_fac",
}
nf = spark.read.table(NFHS)
missing_cols = [c for c in NEED.values() if c not in nf.columns]
assert not missing_cols, f"NFHS columns not found: {missing_cols}"
need = nf.select("state", "district", "n_small_sample_values", "n_suppressed_values",
                 *[F.col(src).alias(dst) for dst, src in NEED.items()])

count_cols = [c for c in supply.columns if c.startswith(("n_", "known_"))]
district = (need.join(supply, ["state", "district"], "left")
                .fillna(0, subset=count_cols)
                .withColumn("has_no_facilities", F.col("n_facilities") == 0)
                .withColumn("gold_run_id", F.lit(RUN_ID)))
front = ["state", "district", "n_facilities", "n_facilities_confident", "has_no_facilities"]
district = district.select(*front, *[c for c in district.columns if c not in front])
district.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(DISTRICT)
DIST_N = spark.read.table(DISTRICT).count()
log(2, "gold_district_summary rows written", DIST_N, DISTRICT)
log(2, "districts with zero facilities in the dataset", spark.read.table(DISTRICT).where("has_no_facilities").count())
log(2, "facilities without an NFHS district (not counted)", base.where("nfhs_district IS NULL").count())

(spark.createDataFrame(audit_rows, "run_id string, step string, rule string, rows_affected long, note string")
      .withColumn("logged_at", F.current_timestamp())
      .write.mode("append").option("mergeSchema", "true").saveAsTable(AUDIT))

# COMMAND ----------

# DBTITLE 1,3. Validation
s = spark.read.table(SEARCH)
d = spark.read.table(DISTRICT)
checks = []
def check(name, ok, detail=""):
    checks.append((name, "PASS" if ok else "FAIL", str(detail)))

expected = search_base.count()
check("Search rows = Silver primary, non-suspect rows with a location", SEARCH_N == expected, f"{SEARCH_N} vs {expected}")
check("unique_id unique in search", s.select("unique_id").distinct().count() == SEARCH_N)
bad = s.where("phone IS NOT NULL AND unique_id IN (SELECT unique_id FROM workspace.datathon_2026.facilities_silver WHERE phone_suspect)").count()
check("No suspect phones exposed", bad == 0, f"{bad} rows")
bad = s.where("latitude IS NULL").count()
check("Every searchable facility has coordinates", bad == 0, f"{bad} without")
bad = s.where("search_text IS NULL OR length(search_text) < 3").count()
check("Every row has search_text", bad == 0, f"{bad} rows")
nfhs_n = nf.count()
check("District table has every NFHS district", DIST_N == nfhs_n, f"{DIST_N} vs {nfhs_n}")
check("One row per (state, district)", d.groupBy("state", "district").count().where("count > 1").count() == 0)
counted = d.agg(F.sum("n_facilities")).first()[0]
linked = base.where("nfhs_district IS NOT NULL").count()
check("District counts add up to linked facilities", counted == linked, f"{counted} vs {linked}")
bad = d.where("n_facilities_confident > n_facilities").count()
check("Confident counts never exceed total counts", bad == 0, f"{bad} rows")

result = spark.createDataFrame(checks, "check string, result string, detail string")
display(result)
failed = [x for x in checks if x[1] == "FAIL"]
if failed:
    raise AssertionError(f"{len(failed)} Gold validation check(s) FAILED: " + "; ".join(x[0] for x in failed))
print(f"All {len(checks)} Gold validation checks PASSED.")
