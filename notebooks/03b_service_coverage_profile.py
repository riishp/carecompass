# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 03b — Service Coverage Profile
# MAGIC
# MAGIC **Purpose:** before building service-evidence scoring (notebook 04), measure how complete and how usable the service-related fields are. This is a **read-only analysis** of the data — source tables are never modified.
# MAGIC
# MAGIC **Inputs (read-only):**
# MAGIC - `workspace.datathon_2026.facilities_clean` (key: `unique_id`) — for identity and state.
# MAGIC - The five service fields (`specialties`, `procedure`, `equipment`, `capability`, `description`) and `facilityTypeId` are **not** in `facilities_clean`, so per the spec we fall back to the shared table `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities` (deduped per `unique_id` exactly as in notebook 03, Step 4a — keep the most complete record).
# MAGIC - Step 5 joins `workspace.datathon_2026.facilities_located`.
# MAGIC
# MAGIC **Outputs (CREATE OR REPLACE, safe to re-run):** `workspace.datathon_2026.service_fields_parsed`, `.service_mention_summary`, `.service_coverage_by_state`.
# MAGIC
# MAGIC > **Guiding principle:** a missing field means "unknown", not "service absent" — a facility with an empty `specialties` list may still offer a service; we just have no textual evidence of it.

# COMMAND ----------

# DBTITLE 1,Config and helpers
from pyspark.sql import functions as F, Window

# ---- read-only inputs (never modified) ----
RAW_FACILITIES = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities"
FACILITIES_CLEAN = "workspace.datathon_2026.facilities_clean"
FACILITIES_LOCATED = "workspace.datathon_2026.facilities_located"

# ---- outputs (CREATE OR REPLACE, safe to re-run) ----
OUT = "workspace.datathon_2026"
SERVICE_FIELDS_TABLE = f"{OUT}.service_fields_parsed"
MENTION_SUMMARY_TABLE = f"{OUT}.service_mention_summary"
COVERAGE_TABLE = f"{OUT}.service_coverage_by_state"

LIST_FIELDS = ["specialties", "procedure", "equipment", "capability"]  # description = free text
ALL_FIELDS = LIST_FIELDS + ["description"]

# NULL, '', '[]', 'null', 'none', 'n/a' (plus the other junk tokens seen in notebook 01) count as empty
MISSING_TOKENS = ["", "null", "nan", "none", "n/a", "na", "-", "[]"]

SERVICES = {
    "dialysis":        r"dialysis|haemodialysis|hemodialysis|nephrolog",
    "c_section":       r"caesarean|cesarean|c-section|\blscs\b",
    "nicu":            r"\bnicu\b|\bsncu\b|neonatal intensive|neonatolog",
    "cardiac_surgery": r"cardiac surg|cardiothoracic|\bcabg\b|bypass surg|open heart",
    "mri":             r"\bmri\b|magnetic resonance",
    "burn_care":       r"burn unit|burns? ward|burn care|burns? (unit|centre|center)",
}


def write_replace(df, table):
    """CREATE OR REPLACE semantics — the notebook can be re-run safely."""
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table)


def is_real(col):
    """A real value: not NULL and not one of the empty markers ('', '[]', 'null', 'none', 'n/a', ...)."""
    s = F.lower(F.trim(F.coalesce(col, F.lit(""))))
    return ~s.isin(MISSING_TOKENS)


def parse_list(col):
    """Parse a list-like string (JSON array stored as a string, or a ';'/','/'|' delimited
    list) into a cleaned array<string>: brackets and per-item quotes stripped, lowercased,
    trimmed, empty/missing items removed."""
    s = F.trim(F.coalesce(col, F.lit("")))
    s = F.regexp_replace(s, r"^\s*\[", "")   # JSON array: drop opening bracket ...
    s = F.regexp_replace(s, r"\]\s*$", "")   # ... and closing bracket
    arr = F.split(s, r"[;|,]")               # JSON commas and ; | delimiters
    arr = F.transform(arr, lambda x: F.trim(
        F.regexp_replace(F.regexp_replace(F.trim(x), r'^"', ""), r'"$', "")))
    arr = F.transform(arr, lambda x: F.lower(x))
    arr = F.filter(arr, lambda x: ~x.isin(MISSING_TOKENS))
    return arr

# COMMAND ----------

# DBTITLE 1,Step 1 — Detect the format
# MAGIC %md
# MAGIC ## Step 1 — Detect the format of the five fields
# MAGIC
# MAGIC Show 10 non-null sample values per field, then classify each field as JSON-array, delimited list, or free text, and parse the list-like fields into `array<string>`. Saved as `service_fields_parsed` (one row per facility).

# COMMAND ----------

# DBTITLE 1,Step 1a — Samples and format detection
# The service fields are missing from facilities_clean -> fall back to the shared facilities
# table, deduped per unique_id the same way as notebook 03 Step 4a (keep the most complete record).
needed = ["unique_id", "name", "facilityTypeId"] + ALL_FIELDS
raw_all = spark.table(RAW_FACILITIES).select(needed)

n_fill = F.lit(0)
for c in needed[1:]:
    n_fill = n_fill + F.when(F.col(c).isNotNull() & (F.trim(F.col(c)) != ""), 1).otherwise(0)
raw_dedup = (
    raw_all.withColumn("n_fill", n_fill)
    .withColumn("rn", F.row_number().over(
        Window.partitionBy("unique_id").orderBy(F.desc("n_fill"), F.sha2(F.concat_ws("|", *needed), 0))))
    .filter("rn = 1").drop("rn", "n_fill")
)
print(f"facilities (shared table, unique_id-deduped, read-only): {raw_dedup.count():,}")

# ---- 10 non-null sample values per field ----
for f in ALL_FIELDS:
    print(f"\n----- {f}: 10 non-null samples -----")
    display(raw_dedup.where(is_real(F.col(f))).select(F.col(f).alias("value")).limit(10))

# ---- format detection: JSON array vs delimited vs free text ----
json_like = lambda col: F.trim(col).rlike(r"^\s*\[.*\]\s*$")
fmt_exprs = []
for f in ALL_FIELDS:
    real = is_real(F.col(f))
    fmt_exprs += [
        F.count(F.when(real, 1)).alias(f"{f}_filled"),
        F.count(F.when(real & json_like(F.col(f)), 1)).alias(f"{f}_json_array"),
        F.count(F.when(real & ~json_like(F.col(f)) & F.col(f).rlike(r"[;|]") , 1)).alias(f"{f}_semicolon_pipe"),
        F.count(F.when(real & ~json_like(F.col(f)) & F.col(f).rlike(r"[;,|]"), 1)).alias(f"{f}_any_delimiter"),
    ]
fmt = raw_dedup.agg(*fmt_exprs).collect()[0].asDict()
for f in ALL_FIELDS:
    filled, n_json, n_semi, n_delim = fmt[f"{f}_filled"], fmt[f"{f}_json_array"], fmt[f"{f}_semicolon_pipe"], fmt[f"{f}_any_delimiter"]
    kind = ("JSON arrays stored as strings" if n_json >= 0.5 * filled
            else "delimited lists" if n_delim >= 0.5 * filled
            else "free text (no dominant structure)")
    print(f"{f:12s} filled {filled:6,} | JSON-like {n_json:6,} | ; or | {n_semi:5,} | any delimiter {n_delim:6,} -> {kind}")

# COMMAND ----------

# DBTITLE 1,Step 1b — Parse and save service_fields_parsed
# ---- parse list fields into array<string>, keep description as text; one row per facility ----
base = (
    spark.table(FACILITIES_CLEAN).select("unique_id", "state_std", "city_std")
    .join(raw_dedup, "unique_id", "left")
)
n_fc = spark.table(FACILITIES_CLEAN).count()
assert base.count() == n_fc, "join changed the row count"

svc = base
for f in LIST_FIELDS:
    svc = (svc.withColumn(f"{f}_items", parse_list(F.col(f)))
              .withColumn(f"{f}_n", F.size(f"{f}_items")))
svc = (
    svc.withColumn("description_text", F.trim(F.coalesce(F.col("description"), F.lit(""))))
       .withColumn("description_words",
                  F.when(is_real(F.col("description_text")),
                         F.size(F.split(F.lower(F.col("description_text")), r"\s+"))))
)
service_fields_parsed = svc.select(
    "unique_id", "name", "state_std", "city_std", "facilityTypeId",
    *[f"{f}_items" for f in LIST_FIELDS], *[f"{f}_n" for f in LIST_FIELDS],
    "description_text", "description_words")

write_replace(service_fields_parsed, SERVICE_FIELDS_TABLE)
print(f"Wrote {SERVICE_FIELDS_TABLE}: {service_fields_parsed.count():,} rows (one per facility)")
print("\nSanity check — a few parsed rows:")
display(spark.table(SERVICE_FIELDS_TABLE).filter("specialties_n > 0 OR procedure_n > 0")
        .select("name", "specialties_n", F.slice("specialties_items", 1, 5).alias("specialties_first5"),
                "procedure_n", F.slice("procedure_items", 1, 5).alias("procedure_first5"),
                "equipment_n", "capability_n", "description_words")
        .limit(8))

# COMMAND ----------

# DBTITLE 1,Step 2 — Fill rates
# MAGIC %md
# MAGIC ## Step 2 — How often each field is filled in
# MAGIC
# MAGIC Real value = not NULL / '' / '[]' / 'null' / 'none' / 'n/a'. Percentages overall, plus breakdowns by state and by `facilityTypeId` to see whether some regions or facility types are much less documented. % with **all five** fields empty, and % with **only description** filled.

# COMMAND ----------

# DBTITLE 1,Step 2 — Fill rates by field, state, facility type
p = spark.table(SERVICE_FIELDS_TABLE)

q = p.select("unique_id", "state_std", "city_std", "facilityTypeId",
             *[f"{f}_n" for f in LIST_FIELDS], "description_words")
for f in LIST_FIELDS:
    q = q.withColumn(f"f_{f}", F.col(f"{f}_n") > 0)
q = q.withColumn("f_description", F.col("description_words").isNotNull())

n_fill_expr = F.lit(0)
for f in ALL_FIELDS:
    n_fill_expr = n_fill_expr + F.col(f"f_{f}").cast("int")
q = q.withColumn("n_filled", n_fill_expr)

n = q.count()
print(f"Facilities: {n:,}\n")
print("---- % of facilities with a real value, per field ----")
row = q.agg(*[F.round(100 * F.avg(F.col(f"f_{f}").cast("double")), 1).alias(f) for f in ALL_FIELDS]).collect()[0]
for f in ALL_FIELDS:
    print(f"  {f:12s} {row[f]:5.1f}%")

all_empty = q.filter("n_filled = 0").count()
only_desc = q.filter("f_description AND n_filled = 1").count()
print(f"\nAll five fields empty:        {all_empty:5,} ({100 * all_empty / n:.1f}%)")
print(f"Only description filled:     {only_desc:5,} ({100 * only_desc / n:.1f}%)")

overall_doc = F.round(100 * F.avg(F.col("n_filled")) / 5.0, 1).alias("overall_doc_pct")
field_rates = [F.round(100 * F.avg(F.col(f"f_{f}").cast("double")), 1).alias(f) for f in ALL_FIELDS]

print("\n---- documentation by state (15 least documented) ----")
display(q.groupBy(F.coalesce(F.col("state_std"), F.lit("(unknown state)")).alias("state"))
        .agg(F.count("*").alias("n_facilities"), overall_doc, *field_rates)
        .orderBy(F.asc("overall_doc_pct")).limit(15))

print("\n---- documentation by facilityTypeId (all types, least documented first) ----")
display(q.groupBy(F.coalesce(F.col("facilityTypeId"), F.lit("(none)")).alias("facilityTypeId"))
        .agg(F.count("*").alias("n_facilities"), overall_doc, *field_rates)
        .orderBy(F.asc("overall_doc_pct")))

# COMMAND ----------

# DBTITLE 1,Step 3 — Content volume
# MAGIC %md
# MAGIC ## Step 3 — How much each field contains
# MAGIC
# MAGIC For list fields: item-count distributions (min, median, p90, max, over facilities with ≥1 item) and histograms. For description: word-count distribution. Facilities in the **top 5% of specialty count** are flagged `possible_boilerplate` (saved back into `service_fields_parsed`), with examples.

# COMMAND ----------

# DBTITLE 1,Step 3 — Item counts, word counts, boilerplate
p = spark.table(SERVICE_FIELDS_TABLE)

# ---- list fields: item counts ----
for f in LIST_FIELDS:
    s = (p.filter(F.col(f"{f}_n") > 0)
         .agg(F.count("*").alias("n_with_items"),
              F.min(f"{f}_n").alias("mn"),
              F.expr(f"percentile({f}_n, 0.5)").alias("median"),
              F.expr(f"percentile({f}_n, 0.9)").alias("p90"),
              F.max(f"{f}_n").alias("mx")).collect()[0])
    print(f"{f:12s} (facilities with >=1 item: {s['n_with_items']:5,}):  min {s['mn']:.0f}  median {s['median']:.0f}  p90 {s['p90']:.0f}  max {s['mx']:.0f}")

def n_bucket(n):
    return (F.when(n == 0, F.lit("0")).when(n == 1, F.lit("1")).when(n == 2, F.lit("2"))
            .when(n == 3, F.lit("3")).when(n <= 5, F.lit("4-5"))
            .when(n <= 10, F.lit("6-10")).otherwise(F.lit("11+")))

hist = None
for f in LIST_FIELDS:
    h = (p.select(F.lit(f).alias("field"), n_bucket(F.col(f"{f}_n")).alias("bucket"), F.col(f"{f}_n").alias("n"))
         .groupBy("field", "bucket").agg(F.count("*").alias("n_facilities"), F.min("n").alias("sort_key")))
    hist = h if hist is None else hist.unionByName(h)
print("\nItem-count histograms (all facilities, empty = 0):")
display(hist.orderBy("field", "sort_key"))

# ---- description: word counts ----
d = p.filter("description_words IS NOT NULL")
ds = d.agg(F.count("*").alias("n"), F.min("description_words").alias("mn"),
           F.expr("percentile(description_words, 0.5)").alias("median"),
           F.expr("percentile(description_words, 0.9)").alias("p90"),
           F.max("description_words").alias("mx")).collect()[0]
print(f"\ndescription (facilities with a real value: {ds['n']:,}):  min {ds['mn']:.0f} words  median {ds['median']:.0f}  p90 {ds['p90']:.0f}  max {ds['mx']:.0f}")
w = F.col("description_words")
print("Word-count distribution:")
display(d.groupBy(F.when(w < 10, F.lit("<10")).when(w <= 25, F.lit("10-25")).when(w <= 50, F.lit("26-50"))
                 .when(w <= 100, F.lit("51-100")).when(w <= 200, F.lit("101-200")).otherwise(F.lit("201+"))
                 .alias("words_bucket")).agg(F.count("*").alias("n_facilities"), F.min("description_words").alias("sort_key"))
        .orderBy("sort_key"))

# ---- top 5% of specialty count -> possible_boilerplate (saved into service_fields_parsed) ----
p95 = p.filter("specialties_n > 0").agg(F.expr("percentile(specialties_n, 0.95)").alias("p95")).collect()[0]["p95"]
print(f"\nTop-5% threshold for specialties item count: {p95:.0f} items")
p3 = p.withColumn("possible_boilerplate", F.col("specialties_n") >= F.lit(float(p95)))
write_replace(p3, SERVICE_FIELDS_TABLE)
boiler = spark.table(SERVICE_FIELDS_TABLE).filter("possible_boilerplate")
print(f"Facilities flagged possible_boilerplate: {boiler.count():,}")
display(boiler.orderBy(F.desc("specialties_n")).limit(10)
        .select("name", "specialties_n",
                F.array_join(F.slice("specialties_items", 1, 8), " | ").alias("first_8_specialties")))

# COMMAND ----------

# DBTITLE 1,Step 4 — Service mentions
# MAGIC %md
# MAGIC ## Step 4 — Service mentions
# MAGIC
# MAGIC Case-insensitive regex search over lowercased text from all five fields, for dialysis, c-section, NICU, cardiac surgery, MRI, and burn care. Reported per service: facilities with any match, facilities matching in **2+ different fields** (corroborated), which fields the matches come from, and 5 random example snippets (~60 characters of context each side). Saved as `service_mention_summary`.

# COMMAND ----------

# DBTITLE 1,Step 4 — Service mentions and corroboration
p = spark.table(SERVICE_FIELDS_TABLE)

# searchable text per field (list fields joined to one string; description lowercased)
t = p.select("unique_id", "name",
            *[F.array_join(F.col(f"{f}_items"), " ").alias(f"tx_{f}") for f in LIST_FIELDS],
            F.lower(F.col("description_text")).alias("tx_description"))
t = t.withColumn("all_text", F.concat_ws(" | ", *[f"tx_{f}" for f in ALL_FIELDS]))

# per (service, field) match flags, and the number of distinct fields matching per service
flags = t
for svc_name, pat in SERVICES.items():
    for f in ALL_FIELDS:
        flags = flags.withColumn(f"m_{svc_name}_{f}",
                                F.when(F.col(f"tx_{f}").rlike(pat), F.lit(1)).otherwise(F.lit(0)))
    total = F.lit(0)
    for f in ALL_FIELDS:
        total = total + F.col(f"m_{svc_name}_{f}")
    flags = flags.withColumn(f"nfields_{svc_name}", total)

agg_exprs = []
for svc_name in SERVICES:
    agg_exprs.append(F.sum(F.when(F.col(f"nfields_{svc_name}") > 0, 1).otherwise(0)).alias(f"{svc_name}__any"))
    agg_exprs.append(F.sum(F.when(F.col(f"nfields_{svc_name}") >= 2, 1).otherwise(0)).alias(f"{svc_name}__corroborated"))
    for f in ALL_FIELDS:
        agg_exprs.append(F.sum(F.col(f"m_{svc_name}_{f}")).alias(f"{svc_name}__{f}"))
res = flags.agg(*agg_exprs).collect()[0].asDict()

# 5 random example snippets per service (~60 chars of context on each side)
rows = []
for svc_name, pat in SERVICES.items():
    ctx = f"(?s)(.{{0,60}}(?:{pat}).{{0,60}})"
    snips = (flags.filter(F.col(f"nfields_{svc_name}") > 0)
             .select("name", F.regexp_extract(F.col("all_text"), ctx, 1).alias("snippet"))
             .orderBy(F.rand(42)).limit(5).collect())
    rows.append((svc_name, res[f"{svc_name}__any"], res[f"{svc_name}__corroborated"],
                 *[res[f"{svc_name}__{f}"] for f in ALL_FIELDS],
                 [r["snippet"] for r in snips]))

summary = spark.createDataFrame(
    rows,
    "service string, n_facilities_with_match long, n_corroborated_2plus_fields long, "
    "n_specialties long, n_procedure long, n_equipment long, n_capability long, "
    "n_description long, example_snippets array<string>")
write_replace(summary, MENTION_SUMMARY_TABLE)

print(f"Wrote {MENTION_SUMMARY_TABLE}")
display(summary.select("service", "n_facilities_with_match", "n_corroborated_2plus_fields",
                       "n_specialties", "n_procedure", "n_equipment", "n_capability", "n_description"))

for r in rows:
    print(f"\n----- {r[0]}: 5 random matching snippets -----")
    for s in r[-1]:
        print(f"  ...{s}...")

# COMMAND ----------

# DBTITLE 1,Step 5 — Coverage near people
# MAGIC %md
# MAGIC ## Step 5 — Coverage near people
# MAGIC
# MAGIC Join the mention flags to `facilities_located` and, using only facilities whose `uncertainty_tier` is **low or medium** (i.e. where we trust the location enough to say the service is near that state's population), count matched and corroborated facilities per state. Then list the states with **zero corroborated** facilities for each service. Saved as `service_coverage_by_state`.

# COMMAND ----------

# DBTITLE 1,Step 5 — Coverage by state and zero-coverage states
loc = spark.table(FACILITIES_LOCATED).select("unique_id", F.col("state").alias("state"), "uncertainty_tier")
svc_flags = flags.select("unique_id", *[f"nfields_{s}" for s in SERVICES])
cov_base = svc_flags.join(loc, "unique_id", "inner")
pop = cov_base.filter("uncertainty_tier IN ('low', 'medium')")
print(f"Facilities with low/medium location uncertainty: {pop.count():,} of {cov_base.count():,}")

parts = []
for svc_name in SERVICES:
    s = (pop.groupBy(F.coalesce(F.col("state"), F.lit("(unknown state)")).alias("state"))
         .agg(F.sum(F.when(F.col(f"nfields_{svc_name}") > 0, 1).otherwise(0)).alias("n_matched"),
              F.sum(F.when(F.col(f"nfields_{svc_name}") >= 2, 1).otherwise(0)).alias("n_corroborated"))
         .withColumn("service", F.lit(svc_name)))
    parts.append(s)
coverage = parts[0]
for s in parts[1:]:
    coverage = coverage.unionByName(s)
coverage = coverage.select("service", "state", "n_matched", "n_corroborated")
write_replace(coverage, COVERAGE_TABLE)
print(f"Wrote {COVERAGE_TABLE}: {coverage.count():,} rows (state x service)")

# ---- states with ZERO corroborated facilities, per service ----
states_pop = {r["state"] for r in pop.select(
    F.coalesce(F.col("state"), F.lit("(unknown state)")).alias("state")).distinct().collect()}
cov_tbl = spark.table(COVERAGE_TABLE)
print(f"\nStates in the low/medium population: {len(states_pop)}")
for svc_name in SERVICES:
    corr_states = {r["state"] for r in cov_tbl.filter(f"service = '{svc_name}' AND n_corroborated > 0")
                   .select("state").collect()}
    zero = sorted(states_pop - corr_states)
    print(f"\n{svc_name}: {len(corr_states)} states with corroborated facilities, "
          f"{len(zero)} with zero corroborated:")
    if zero:
        print("  " + ", ".join(zero))

display(coverage.orderBy("service", F.desc("n_matched")).limit(40))

# COMMAND ----------

# DBTITLE 1,What this means for scoring
# MAGIC %md
# MAGIC ## What this means for scoring
# MAGIC
# MAGIC **1. Which fields are reliable enough to count as evidence?**
# MAGIC
# MAGIC | Field | Filled | Format | Verdict |
# MAGIC |---|---|---|---|
# MAGIC | `specialties` | 98.8% | JSON array | Reliable evidence source (but see boilerplate below) |
# MAGIC | `capability` | 98.5% | JSON array | Reliable evidence source |
# MAGIC | `procedure` | 87.7% | JSON array | Reliable, and the *only* place c-section is ever mentioned |
# MAGIC | `equipment` | 70.6% | JSON array | Sparse — use when present, never require it (clinics 63.8%, dentists 65.6%, doctors 38.1%) |
# MAGIC | `description` | 99.2% | free text | Thin (median 16 words, often generic) — supplementary only |
# MAGIC
# MAGIC Only 80 facilities (0.8%) have all five fields empty and 35 (0.3%) only description, so an explicit "no textual evidence" tier is warranted but small.
# MAGIC
# MAGIC **2. Per service, is evidence plentiful, moderate, or sparse?** (facilities with any match / corroborated in 2+ fields)
# MAGIC
# MAGIC | Service | Any | Corroborated | Corroborated % | Verdict |
# MAGIC |---|---|---|---|---|
# MAGIC | dialysis | 2,483 | 1,456 | 59% | plentiful |
# MAGIC | nicu | 1,570 | 830 | 53% | plentiful |
# MAGIC | mri | 950 | 538 | 57% | plentiful |
# MAGIC | cardiac_surgery | 708 | 275 | 39% | moderate |
# MAGIC | c_section | 442 | 95 | 21% | sparse |
# MAGIC | burn_care | 64 | 26 | 41% | very sparse |
# MAGIC
# MAGIC **3. Is a "2+ fields" bar realistic?**
# MAGIC
# MAGIC For dialysis, nicu and mri, yes — roughly half of matches are corroborated. For c_section it is **too strict**: 97% of its matches come from `procedure` alone (`specialties` never mentions caesarean sections), so requiring 2 fields keeps only 95 of 442 facilities. Burn care is similar (64 matches total). Recommendation for notebook 04:
# MAGIC
# MAGIC * corroborated (2+ fields) → **strong evidence**;
# MAGIC * single-field match, especially in a structured list field (`procedure`, `capability`, `specialties`) → **medium evidence**;
# MAGIC * no match → **unknown, not absent** — `equipment` is missing for ~30% of facilities, so an unlisted MRI is not a disproof.
# MAGIC * Field-combination checks must be service-specific: `specialties` vocabulary contains NICU-adjacent terms (neonatology) but never c-section or MRI terms.
# MAGIC
# MAGIC **4. Where is documentation missing?**
# MAGIC
# MAGIC * `equipment` is the weak field for every facility type (~30% missing overall; ~36% for dentists and clinics, 62% for doctors).
# MAGIC * The state we joined on (`facilities_located.state`, from `address_stateOrRegion`) is **noisy**: 189 distinct values including city names, abbreviations ("m.p.", "dl", "ts"), and coordinate strings. Scoring by state should use the PIN-directory state from notebook 03 (`primary_state`), not the raw address state.
# MAGIC * `facilityTypeId` also contains junk values (URLs, coordinates, hashes); the usable mass is hospital (5,626), clinic (3,782), dentist (489).
# MAGIC * **2,873 facilities (28%) are flagged `possible_boilerplate`** (50+ specialty items with heavy repetition, e.g. `internalmedicine` repeated many times) — deduplicate item arrays before matching, or repeats will inflate confidence.
# MAGIC * Canonical-state view (36 PIN-directory states, low/medium location uncertainty): burn_care has zero corroborated facilities in **22 states** (incl. Delhi, West Bengal, AP, Telangana); c_section in **19** (incl. Madhya Pradesh, Uttarakhand, AP); cardiac_surgery in **11**; mri in **8**; dialysis and nicu in **6** (mostly small or remote: A&N Islands, Ladakh, Lakshadweep, Sikkim, Manipur, Tripura, ...). Some are genuinely remote states; for the rest this is a documentation gap — the recommender must report "no evidence found in our data", never "this service does not exist here".

# COMMAND ----------

