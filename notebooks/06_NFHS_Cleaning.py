# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 06 — NFHS-5 Cleaning (district health indicators)
# MAGIC
# MAGIC **What NFHS is:** India's National Family Health Survey, round 5 (2019–21). One row per district, ~110 health indicators
# MAGIC (insurance coverage, anaemia, vaccination, institutional births, blood pressure, blood sugar…). In CareCompass it measures **health need**; the facilities table measures **care supply**.
# MAGIC
# MAGIC **Input (read-only):** `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.nfhs_5_district_health_indicators`
# MAGIC **Reference (from `03_location_cleaning_and_uncertainty`):** `workspace.datathon_2026.pincode_clean` — India Post district names, the names facilities use.
# MAGIC
# MAGIC **Outputs (CREATE OR REPLACE — safe to re-run):**
# MAGIC | table | contents |
# MAGIC |---|---|
# MAGIC | `nfhs_clean` | one row per district; canonical state, trimmed district, every indicator as a number, reliability counts, and the matching India Post district |
# MAGIC | `nfhs_value_flags` | one row per indicator value NFHS marked as unreliable: `(x)` = based on 25–49 cases, `*` = suppressed (<25 cases) |
# MAGIC | `nfhs_district_crosswalk` | NFHS district → India Post district, with how each match was made |
# MAGIC | `nfhs_quarantine` | rows removed (with reason) |
# MAGIC | `cleaning_audit` | one log row per rule (shared team audit table) |
# MAGIC
# MAGIC **Rules** (from `03_Data_Quality_Audit` §2–4):
# MAGIC 1. Trim whitespace (704 of 706 district names have trailing spaces; `" Lakshadweep "`).
# MAGIC 2. Map states to the 36 canonical names (`Maharastra` → Maharashtra, `NCT of Delhi` → Delhi, `&` → `and`).
# MAGIC 3. Remove the Mizoram "Chandel" row — an exact copy of Manipur's Chandel (Chandel is in Manipur).
# MAGIC 4. Convert text indicators to numbers: `(56.5)` → 56.5 flagged *small sample*; `*` → NULL flagged *suppressed*.
# MAGIC 5. Match every district to India Post names: exact → spelling variant (fuzzy) → hand-written renames (Gurgaon → Gurugram, Allahabad → Prayagraj…).
# MAGIC
# MAGIC **Cleaning only — no analysis.** Section 6 runs validation checks; all must PASS.

# COMMAND ----------

# DBTITLE 1,0. Setup
import re, difflib
from datetime import datetime, timezone
import pandas as pd
from pyspark.sql import functions as F

SRC = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
NFHS_SRC = f"{SRC}.nfhs_5_district_health_indicators"
TGT = "workspace.datathon_2026"
PIN_CLEAN = f"{TGT}.pincode_clean"          # produced by 03_location_cleaning_and_uncertainty
NFHS_CLEAN = f"{TGT}.nfhs_clean"
NFHS_FLAGS = f"{TGT}.nfhs_value_flags"
NFHS_XWALK = f"{TGT}.nfhs_district_crosswalk"
NFHS_QUAR = f"{TGT}.nfhs_quarantine"
AUDIT = f"{TGT}.cleaning_audit"

RUN_ID = "nfhs-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
audit_rows = []
def log(step, rule, n, note=""):
    audit_rows.append((RUN_ID, f"nfhs-{step}", rule, int(n), note))
    print(f"[{step}] {rule}: {n:,} {note}")

assert spark.catalog.tableExists(PIN_CLEAN), f"{PIN_CLEAN} is missing — run 03_location_cleaning_and_uncertainty first"

raw = spark.read.table(NFHS_SRC)  # read-only
RAW_COUNT = raw.count()
print(f"NFHS raw rows: {RAW_COUNT}, columns: {len(raw.columns)}")

# COMMAND ----------

# DBTITLE 1,1. Trim text and map states
# MAGIC %md
# MAGIC ## 1. Trim text and map states to the 36 canonical names

# COMMAND ----------

CANON_STATES = [
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
    "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
]
def skey(s):
    s = re.sub(r"^\s*the\s+", "", str(s).lower())   # "The Dangs" -> "dangs" (but "Theni" stays "theni")
    return re.sub(r"[^a-z]", "", s.replace("&", "and"))
STATE_KEYS = {skey(s): s for s in CANON_STATES}
STATE_KEYS.update({"maharastra": "Maharashtra", "nctofdelhi": "Delhi", "orissa": "Odisha", "pondicherry": "Puducherry"})

state_vals = [r[0] for r in raw.select("state_ut").distinct().collect()]
state_map = {v: STATE_KEYS.get(skey(v)) for v in state_vals}
unmapped = [v for v, c in state_map.items() if c is None]
assert not unmapped, f"Unmapped NFHS state names: {unmapped}"
smap = F.create_map([F.lit(x) for kv in state_map.items() for x in kv])

df = (raw.withColumn("state_raw", F.col("state_ut"))
         .withColumn("district_raw", F.col("district_name"))
         .withColumn("state", smap[F.col("state_ut")])
         .withColumn("district", F.regexp_replace(F.trim(F.col("district_name")), r"\s+", " "))
         .drop("state_ut", "district_name"))

log(1, "district names trimmed", raw.filter(F.col("district_name") != F.trim("district_name")).count())
log(1, "state values changed to canonical name", raw.filter(~F.col("state_ut").isin(CANON_STATES)).count(),
    ", ".join(sorted({f"{k.strip()}→{v}" for k, v in state_map.items() if k != v})))

# COMMAND ----------

# DBTITLE 1,2. Remove duplicate / mislabeled rows
# MAGIC %md
# MAGIC ## 2. Remove duplicate and mislabeled rows
# MAGIC A row is quarantined when every indicator is identical to a row in a **different state** with the same district name — the known case is Mizoram "Chandel" (a copy of Manipur's Chandel; Chandel is in Manipur, and Mizoram's Saiha district is absent from this file).
# MAGIC We keep the copy whose state actually contains that district according to India Post.

# COMMAND ----------

INDICATORS = [c for c in df.columns if c not in ("state", "district", "state_raw", "district_raw")]
pin_districts = (spark.read.table(PIN_CLEAN)
                 .select(F.col("state_std").alias("pst"), F.col("district_std").alias("pdist"))
                 .where("pdist IS NOT NULL").distinct())

sig = F.sha2(F.concat_ws("|", *[F.coalesce(F.col(c).cast("string"), F.lit("∅")) for c in INDICATORS]), 256)
df = df.withColumn("_sig", sig).withColumn("_dkey", F.lower(F.regexp_replace("district", r"[^A-Za-z]", "")))
dupe_groups = df.groupBy("_sig", "_dkey").agg(F.countDistinct("state").alias("n_states")).where("n_states > 1")
cand = df.join(dupe_groups, ["_sig", "_dkey"])
# does India Post list this district name in this state?
pin_keys = pin_districts.select("pst", F.lower(F.regexp_replace("pdist", r"[^A-Za-z]", "")).alias("_pkey")).distinct()
cand = (cand.join(pin_keys, (F.lower(F.col("state")) == F.col("pst")) & (F.col("_dkey") == F.col("_pkey")), "left")
            .withColumn("_in_state", F.col("_pkey").isNotNull()).drop("pst", "_pkey"))
to_quarantine = cand.where("NOT _in_state").select("state", "district")
quarantine = (df.join(to_quarantine, ["state", "district"], "inner")
                .withColumn("quarantine_reason", F.lit("exact copy of the same district in another state; district not in this state per India Post")))
df = df.join(to_quarantine, ["state", "district"], "left_anti").drop("_sig", "_dkey")
for r in quarantine.select("state", "district").collect():
    log(2, "row quarantined (duplicate in wrong state)", 1, f"{r.state} / {r.district}")

# COMMAND ----------

# DBTITLE 1,3. Indicators to numbers with reliability flags
# MAGIC %md
# MAGIC ## 3. Indicators → numbers, with NFHS reliability markers kept as flags
# MAGIC | raw | meaning (NFHS convention) | clean value | flag |
# MAGIC |---|---|---|---|
# MAGIC | `56.5` | normal estimate | 56.5 | — |
# MAGIC | `(56.5)` | based on 25–49 unweighted cases — less reliable | 56.5 | `small_sample` |
# MAGIC | `*` | fewer than 25 cases — NFHS suppresses it | NULL | `suppressed` |
# MAGIC
# MAGIC Every flagged cell is listed in `nfhs_value_flags`; each district also gets `n_small_sample_values` and `n_suppressed_values`.

# COMMAND ----------

text_ind = [c for c, t in df.dtypes if c in INDICATORS and t == "string"]
flag_cols = []
for c in text_ind:
    s = F.trim(F.col(c))
    df = (df.withColumn(f"_flag_{c}",
                        F.when(s == "*", "suppressed").when(s.rlike(r"^\(\s*-?[0-9.]+\s*\)$"), "small_sample"))
            .withColumn(f"_raw_{c}", F.col(c))
            .withColumn(c, F.expr(f"try_cast(regexp_replace(trim(`{c}`), '[()\\\\s,]', '') as double)")))
    flag_cols.append(c)

# any text value that is neither a number, (number) nor * would silently become NULL — surface it
unexpected = df.select([F.sum((F.col(f"_raw_{c}").isNotNull() & F.col(c).isNull() & F.col(f"_flag_{c}").isNull()).cast("int")).alias(c)
                        for c in flag_cols]).first().asDict()
unexpected = {c: n for c, n in unexpected.items() if n}
assert not unexpected, f"Unrecognised indicator text values: {unexpected}"

flags_long = df.select("state", "district", F.explode(F.array(*[
    F.struct(F.lit(c).alias("indicator"), F.col(f"_raw_{c}").alias("raw_value"), F.col(c).alias("value"), F.col(f"_flag_{c}").alias("reliability"))
    for c in flag_cols])).alias("x")).select("state", "district", "x.*").where("reliability IS NOT NULL")

df = (df.withColumn("n_small_sample_values", sum(F.when(F.col(f"_flag_{c}") == "small_sample", 1).otherwise(0) for c in flag_cols))
        .withColumn("n_suppressed_values", sum(F.when(F.col(f"_flag_{c}") == "suppressed", 1).otherwise(0) for c in flag_cols))
        .drop(*[f"_flag_{c}" for c in flag_cols], *[f"_raw_{c}" for c in flag_cols]))

for r in flags_long.groupBy("reliability").count().collect():
    log(3, f"indicator values flagged {r['reliability']}", r["count"])
log(3, "text indicator columns converted to numbers", len(flag_cols))

# COMMAND ----------

# DBTITLE 1,4. Match districts to India Post names
# MAGIC %md
# MAGIC ## 4. District crosswalk: NFHS name → India Post district (the name facilities carry)
# MAGIC Order of matching, first hit wins:
# MAGIC 1. **manual** — hand-checked renames/spellings below (fuzzy matching gets these wrong, e.g. *Faizabad* would match *Firozabad*; it is now *Ayodhya*)
# MAGIC 2. **exact** — same letters after lower-casing and removing spaces/punctuation
# MAGIC 3. **fuzzy** — ≥ 75% similar spelling within the same state (e.g. *Baramula* → *baramulla*)
# MAGIC 4. **none** — no India Post district; kept, but can't be joined to facilities

# COMMAND ----------

# (canonical state, NFHS district) -> India Post district_std (lowercase, as in pincode_clean)
MANUAL = {
    ("Andhra Pradesh", "Sri Potti Sriramulu Nello"): "spsr nellore",
    ("Chhattisgarh", "Koriya"): "korea",
    ("Chhattisgarh", "Uttar Bastar Kanker"): "kanker",
    ("Gujarat", "The Dangs"): "dang",
    ("Haryana", "Gurgaon"): "gurugram",
    ("Haryana", "Mewat"): "nuh",
    ("Jammu and Kashmir", "Punch"): "poonch",
    ("Jharkhand", "Purbi Singhbhum"): "east singhbum",
    ("Jharkhand", "Pashchimi Singhbhum"): "west singhbhum",
    ("Karnataka", "Belgaum"): "belagavi",
    ("Karnataka", "Bijapur"): "vijayapura",
    ("Karnataka", "Bellary"): "ballari",
    ("Karnataka", "Bangalore"): "bengaluru urban",
    ("Karnataka", "Mysore"): "mysuru",
    ("Karnataka", "Gulbarga"): "kalaburagi",
    ("Lakshadweep", "Lakshadweep"): "lakshadweep district",
    ("Madhya Pradesh", "Khargone (West Nimar)"): "khargone",
    ("Madhya Pradesh", "Khandwa (East Nimar)"): "east nimar",
    ("Maharashtra", "Bid"): "beed",
    ("Odisha", "Subarnapur"): "sonepur",
    ("Punjab", "Muktsar"): "sri muktsar sahib",
    ("Punjab", "Sahibzada Ajit Singh Nagar"): "s.a.s nagar",
    ("Tamil Nadu", "Thoothukkudi"): "tuticorin",
    ("Uttar Pradesh", "Jyotiba Phule Nagar"): "amroha",
    ("Uttar Pradesh", "Mahamaya Nagar"): "hathras",
    ("Uttar Pradesh", "Allahabad"): "prayagraj",
    ("Uttar Pradesh", "Sant Ravidas Nagar (Bhadohi)"): "bhadohi",
    ("Uttar Pradesh", "Faizabad"): "ayodhya",
    ("Uttar Pradesh", "Kanshiram Nagar"): "kasganj",
    ("Uttarakhand", "Garhwal"): "pauri garhwal",
    ("West Bengal", "Koch Bihar"): "coochbehar",
    ("West Bengal", "Uttar Dinajpur"): "dinajpur uttar",
    ("West Bengal", "Dakshin Dinajpur"): "dinajpur dakshin",
    ("West Bengal", "North Twenty Four Pargana"): "24 paraganas north",
    ("West Bengal", "South Twenty Four Pargana"): "24 paraganas south",
    ("West Bengal", "Hugli"): "hooghly",
    ("West Bengal", "Haora"): "howrah",
    ("West Bengal", "Paschim Medinipur"): "medinipur west",
    ("West Bengal", "Purba Medinipur"): "medinipur east",
}

pin_d = pin_districts.toPandas()
pin_d["state"] = pin_d.pst.map(lambda s: STATE_KEYS.get(skey(s)))
pin_d["key"] = pin_d.pdist.map(lambda s: re.sub(r"[^a-z]", "", s.lower()))
by_state = {st: g for st, g in pin_d.groupby("state")}

def match(state, district):
    cands = by_state.get(state)
    if (state, district) in MANUAL:
        tgt = MANUAL[(state, district)]
        ok = cands is not None and tgt in set(cands.pdist)
        return (tgt if ok else None), ("manual" if ok else "manual-target-missing")
    if cands is None:
        return None, "none"
    k = re.sub(r"[^a-z]", "", re.sub(r"^\s*the\s+", "", district.lower()))
    hit = cands[cands.key == k]
    if len(hit):
        return hit.pdist.iloc[0], "exact"
    m = difflib.get_close_matches(k, list(cands.key), n=1, cutoff=0.75)
    if m:
        return cands[cands.key == m[0]].pdist.iloc[0], "fuzzy"
    return None, "none"

nd = df.select("state", "district").toPandas()
nd[["pin_district", "match_type"]] = nd.apply(lambda r: pd.Series(match(r.state, r.district)), axis=1)
nd["pin_state"] = nd.state.str.lower()
# Two NFHS districts mapping to one India Post district (e.g. Warangal Rural + Urban -> warangal) is allowed but flagged
nd["shares_pin_district"] = nd.duplicated(["state", "pin_district"], keep=False) & nd.pin_district.notna()

xwalk = spark.createDataFrame(nd[["state", "district", "pin_state", "pin_district", "match_type", "shares_pin_district"]])
xwalk.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(NFHS_XWALK)
for mt, n in nd.match_type.value_counts().items():
    log(4, f"district match: {mt}", n)
log(4, "NFHS districts sharing one India Post district", int(nd.shares_pin_district.sum()),
    ", ".join(sorted(set(nd[nd.shares_pin_district].district))))
display(nd[nd.match_type.isin(["fuzzy", "manual", "none", "manual-target-missing"])].sort_values(["match_type", "state"]))

# COMMAND ----------

# DBTITLE 1,5. Write tables
# MAGIC %md
# MAGIC ## 5. Write `nfhs_clean`, `nfhs_value_flags`, `nfhs_quarantine`, audit

# COMMAND ----------

nfhs_clean = (df.join(xwalk.select("state", "district", "pin_state", "pin_district", F.col("match_type").alias("district_match_type")),
                      ["state", "district"], "left")
                .withColumn("cleaned_run_id", F.lit(RUN_ID)))
front = ["state", "district", "pin_state", "pin_district", "district_match_type",
         "n_small_sample_values", "n_suppressed_values", "state_raw", "district_raw"]
nfhs_clean = nfhs_clean.select(*front, *[c for c in nfhs_clean.columns if c not in front])
nfhs_clean.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(NFHS_CLEAN)
flags_long.withColumn("cleaned_run_id", F.lit(RUN_ID)).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(NFHS_FLAGS)
(quarantine.drop("_sig", "_dkey").withColumn("cleaned_run_id", F.lit(RUN_ID))
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(NFHS_QUAR))

CLEAN_N, QUAR_N = spark.read.table(NFHS_CLEAN).count(), spark.read.table(NFHS_QUAR).count()
log(5, "nfhs_clean rows written", CLEAN_N, NFHS_CLEAN)
log(5, "nfhs_quarantine rows written", QUAR_N, NFHS_QUAR)

(spark.createDataFrame(audit_rows, "run_id string, step string, rule string, rows_affected long, note string")
      .withColumn("logged_at", F.current_timestamp())
      .write.mode("append").option("mergeSchema", "true").saveAsTable(AUDIT))

# COMMAND ----------

# DBTITLE 1,6. Validation
# MAGIC %md
# MAGIC ## 6. Validation — all checks must PASS (the cell raises an error otherwise)

# COMMAND ----------

c = spark.read.table(NFHS_CLEAN)
checks = []
def check(name, ok, detail=""):
    checks.append((name, "PASS" if ok else "FAIL", str(detail)))

check("Row reconciliation: raw = clean + quarantine", RAW_COUNT == CLEAN_N + QUAR_N, f"{RAW_COUNT} = {CLEAN_N} + {QUAR_N}")
dup = c.groupBy("state", "district").count().where("count > 1").count()
check("One row per (state, district)", dup == 0, f"{dup} duplicates")
bad_state = c.where(~F.col("state").isin(CANON_STATES) | F.col("state").isNull()).count()
check("Every state is one of the 36 canonical names", bad_state == 0, f"{bad_state} invalid")
untrim = c.where((F.col("district") != F.trim("district")) | F.col("district").rlike(r"\s{2,}")).count()
check("No leading/trailing/double spaces in district", untrim == 0, f"{untrim} rows")
non_num = [col for col, t in c.dtypes if col in INDICATORS and t not in ("double", "int", "bigint")]
check("Every indicator column is numeric", not non_num, ", ".join(non_num) or "all numeric")
pct_cols = [col for col in INDICATORS if col.endswith("_pct")]
out_of_range = c.select([F.sum(((F.col(col) < 0) | (F.col(col) > 100)).cast("int")).alias(col) for col in pct_cols]).first().asDict()
out_of_range = {k: v for k, v in out_of_range.items() if v}
check("Every % indicator is within 0–100", not out_of_range, str(out_of_range) or "ok")
nomatch = c.where("pin_district IS NULL").select("state", "district").collect()
check("Every district matched to an India Post district", not nomatch, [f"{r.state}/{r.district}" for r in nomatch][:10])
check("Mizoram has no 'Chandel' row", c.where("state = 'Mizoram' AND district = 'Chandel'").count() == 0)

result = spark.createDataFrame(checks, "check string, result string, detail string")
display(result)
failed = [x for x in checks if x[1] == "FAIL"]
if failed:
    raise AssertionError(f"{len(failed)} NFHS validation check(s) FAILED: " + "; ".join(x[0] for x in failed))
print(f"All {len(checks)} NFHS validation checks PASSED.")
