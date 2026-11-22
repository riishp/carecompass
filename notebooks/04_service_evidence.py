# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 04 — Service evidence scoring
# MAGIC
# MAGIC **Purpose.** For each facility × each of 7 services (dialysis, c_section, nicu, cardiac_surgery, angioplasty, mri, burn_care), score how strongly the record supports the claim that the facility offers that service, and keep the exact text that supports the score. The recommender app quotes this text to answer **"WHY this facility?"**.
# MAGIC
# MAGIC **Conservative principle.** All evidence is self-reported (the facility's own web presence). A missing field means *unknown*, not *service absent*: tiers `none` / `no_data` assert "not evidenced in our data", never "does not offer".
# MAGIC
# MAGIC **Inputs (read-only).**
# MAGIC * `workspace.datathon_2026.gold_facility_search` — the cleaned gold table, one row per `unique_id`, with case-preserved `procedures` / `equipment` / `capabilities` / `specialty_labels` / `specialties_other` / `description`, camelCase `specialty_codes`, cleaned `facility_type`, canonical PIN-derived `state` (never a raw address state), `number_doctors`, `name`.
# MAGIC * `workspace.datathon_2026.service_fields_parsed` (notebook 03b output) — joined **only** for the `possible_boilerplate` flag (join is complete, 0 unmatched).
# MAGIC
# MAGIC **Outputs (new tables only, overwrite semantics).** `facility_items`, `service_requirements`, `facility_service_evidence`, `benchmark_evidence_labels`.
# MAGIC
# MAGIC No source table is ever modified.

# COMMAND ----------

# DBTITLE 1,Config — services, regexes, helpers
from pyspark.sql import functions as F, Window
from functools import reduce

SCHEMA = "workspace.datathon_2026"
GOLD_TABLE = f"{SCHEMA}.gold_facility_search"
SFP_TABLE = f"{SCHEMA}.service_fields_parsed"        # 03b output — used ONLY for possible_boilerplate
M03B_SUMMARY = f"{SCHEMA}.service_mention_summary"   # read-only, for the comparison summary

GOLD = spark.table(GOLD_TABLE).cache()
N_GOLD = GOLD.count()
print(f"gold_facility_search: {N_GOLD:,} facilities (one row per unique_id)")

# ---------- classification regexes (all matched against item_key = lower(trim(item))) ----------
MEDICAL_RE = r"dialysis|nephrolog|caesarean|cesarean|\blscs\b|\bnicu\b|\bsncu\b|neonat|cardiac|cath ?lab|angioplasty|stent|\bcabg\b|\bmri\b|ct ?scan|burn|\bicu\b|picu|intensive care|ventilator|operation theat|surgery|surgical|blood bank|emergency|trauma|casualty|ambulance|x-?ray|ultrasound|deliver|labou?r room|maternity|obstetric|oncolog|chemo|endoscop|laparoscop|transplant|ana?esth"
TRUST_RE = r"nabh|nabl|\bjci\b|\biso\b|accredit|certified"
UNINFORMATIVE_RULES = [  # (drop_reason, regex) — applied in this order
    ("coordinates",         r'"coordinates"|\{"coordinates"|type" ?:"point"|-?\d{1,2}\.\d{4,}, ?-?\d{1,3}\.\d{4,}'),
    ("address_or_location", r"located (in|at|near)|situated|branch location|pin ?code|address"),
    ("directory_listing",   r"is listed|listed (in|under|among|as)|justdial|practo|medindia|lybrate|hospital director"),
    ("amenities",           r"wheelchair|parking|restroom|toilet|wifi|cafeteria|\batm\b|\blift\b|elevator|waiting (room|area)|lobby|canteen"),
    ("photo_observation",   r"photo|image|visible"),
    ("opening_hours",       r"24 ?x ?7|24/7|open(ing)? (hours|daily|days)|timings|a\.?m\.? (to|-) ?p\.?m\.?|monday|sunday|saturday"),
    ("contact",             r"\bphone\b|mobile|email|contact|www\.|http|\+91|\b\d{10}\b|whatsapp"),
    ("doctor_name",         r"\bdr\.? |^dr\b|doctors? (on staff|available|visiting)"),
    ("vague_marketing",     r"state[- ]of[- ]the[- ]art|world[- ]class|advanced|modern|latest|hi-tech|high-tech|premium|renowned|leading|\bbest\b|top[- ]rated|trusted|years of experience|employees|cutting[- ]edge|excellent|quality care"),
    ("numeric",             r"^[\d\s.,\-]+$"),  # pure numbers are uninformative (extension)
]
SPECIFICITY_RE = r"\d+ ?(bed|machine|unit|station|slice|tesla)|\d(\.\d)? ?t\b|siemens|philips|\bge\b|fresenius|nipro|b\.? ?braun|toshiba|canon"

# ---------- service requirements map (codes verified against gold.specialty_codes in Step 2) ----------
SERVICES = {
    "dialysis": {
        "core": r"dialysis|ha?emodialysis|peritoneal dialysis|\bcapd\b",
        "supporting": r"nephrolog|kidney care|ro plant|av fistula",
        "specialty_codes": ["nephrology"],
        "supporting_codes": [],
    },
    "c_section": {
        "core": r"caesarean|cesarean|c-section|\blscs\b",
        "supporting": r"obstetric|labou?r room|delivery|maternity|operation theat|blood bank|24.?7|ana?esth",
        "specialty_codes": ["gynecologyAndObstetrics"],
        "supporting_codes": ["gynecology", "obstetricsAndMaternityCare"],
    },
    "nicu": {
        "core": r"\bnicu\b|\bsncu\b|neonatal intensive|neonatal icu",
        "supporting": r"neonatolog|incubator|phototherapy|radiant warmer|\bcpap\b|neonatal ventilator",
        "specialty_codes": ["neonatologyPerinatalMedicine"],
        "supporting_codes": [],
    },
    "cardiac_surgery": {
        "core": r"cardiac surg|cardiothoracic|\bcabg\b|bypass surg|open heart|valve replacement",
        "supporting": r"cath ?lab|angioplasty|cardiac icu|\bccu\b",
        "specialty_codes": ["cardiothoracicSurgery", "cardiacSurgery"],
        "supporting_codes": ["cardiology"],
    },
    "angioplasty": {  # interventionalCardiology code is ABSENT from gold -> no primary specialty, cardiology supporting only (spec fallback)
        "core": r"angioplasty|\bptca\b|stent|cath ?lab|cardiac catheteri",
        "supporting": r"cardiolog|\bccu\b|cardiac icu",
        "specialty_codes": [],
        "supporting_codes": ["cardiology"],
    },
    "mri": {
        "core": r"\bmri\b|magnetic resonance",
        "supporting": r"radiolog|ct ?scan",
        "specialty_codes": [],
        "supporting_codes": ["radiology", "diagnosticRadiology", "interventionalRadiology"],
    },
    "burn_care": {
        "core": r"burn unit|burns? ward|burn care|burns? (icu|unit|centre|center)|treat\w* burns?",
        "supporting": r"plastic surg|skin graft",
        "specialty_codes": [],
        "supporting_codes": ["plasticSurgery", "reconstructivePlasticSurgery", "burnAndTraumaPlasticSurgery"],
    },
}
CONTRADICTION_SERVICES = {"nicu", "cardiac_surgery", "angioplasty", "burn_care"}
SVC_DISPLAY = {"dialysis": "dialysis", "c_section": "C-section", "nicu": "NICU",
               "cardiac_surgery": "cardiac surgery", "angioplasty": "angioplasty",
               "mri": "MRI", "burn_care": "burn care"}
SVC_SPECIALTY_PHRASE = {"dialysis": "nephrology is a listed specialty",
                        "c_section": "obstetrics & gynecology is a listed specialty",
                        "nicu": "neonatology is a listed specialty",
                        "cardiac_surgery": "cardiothoracic surgery is a listed specialty",
                        "angioplasty": "interventional cardiology is a listed specialty"}  # mri/burn_care: no primary specialty
SVC_WEAK_LABEL = {"cardiac_surgery": "cardiology", "angioplasty": "cardiology", "mri": "radiology",
                  "burn_care": "plastic surgery", "c_section": "gynecology or obstetrics & maternity care"}

def write_table(df, name):
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.{name}")
    print(f"wrote {SCHEMA}.{name}: {spark.table(f'{SCHEMA}.{name}').count():,} rows")

# COMMAND ----------

# DBTITLE 1,Step 1 — item table (facility_items)
# STEP 1 — item table -> facility_items.
# posexplode procedures / equipment / capabilities (+ specialty_labels AND specialties_other -> 'specialties'),
# description as ONE item (field='description', pos=0); dedup per (unique_id, field) on lower(trim(item)) keeping first pos.

def explode_field(colname, fieldname):
    return (GOLD.select("unique_id", F.posexplode(colname).alias("pos", "item"))
                  .withColumn("field", F.lit(fieldname)))

list_items = (explode_field("procedures", "procedure")
    .unionByName(explode_field("equipment", "equipment"))
    .unionByName(explode_field("capabilities", "capability"))
    .unionByName(explode_field("specialty_labels", "specialties"))
    .unionByName(explode_field("specialties_other", "specialties")))
desc_items = (GOLD.select("unique_id", F.lit(0).alias("pos"),
                          F.coalesce(F.col("description"), F.lit("")).alias("item"))
                 .withColumn("field", F.lit("description")))
raw_items = list_items.unionByName(desc_items).withColumn("item", F.coalesce(F.col("item"), F.lit("")))
N_RAW_ITEMS = raw_items.count()

w = Window.partitionBy("unique_id", "field", "item_key").orderBy("pos")
items = (raw_items
    .withColumn("item_key", F.lower(F.trim(F.col("item"))))
    .withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") == 1).drop("rn"))

# ---- classify (rules in this exact order) ----
cls = (F.when(F.col("field") == "specialties", F.lit("informative"))
       .when(F.length(F.trim(F.col("item"))) < 4, F.lit("empty"))
       .when(F.col("item_key").rlike(MEDICAL_RE), F.lit("informative"))
       .when(F.col("item_key").rlike(TRUST_RE), F.lit("trust_signal")))
for _reason, _pat in UNINFORMATIVE_RULES:
    cls = cls.when(F.col("item_key").rlike(_pat), F.lit("uninformative"))
cls = cls.otherwise(F.lit("unclear"))
items = items.withColumn("item_class", cls)

dr = F.when(F.col("item_class") == "empty", F.lit("too_short"))
for _reason, _pat in UNINFORMATIVE_RULES:
    dr = dr.when((F.col("item_class") == "uninformative") & F.col("item_key").rlike(_pat), F.lit(_reason))
items = items.withColumn("drop_reason", dr)

items = items.select("unique_id", "field", "pos", "item", "item_key", "item_class", "drop_reason")
print(f"items before dedup: {N_RAW_ITEMS:,} -> after dedup per (unique_id, field, item_key): {items.count():,}")
write_table(items, "facility_items")

print("\nCounts by field x item_class:")
display(spark.table(f"{SCHEMA}.facility_items").groupBy("field", "item_class").count().orderBy("field", "item_class"))
print("\nCounts by item_class (total):")
display(spark.table(f"{SCHEMA}.facility_items").groupBy("item_class").count().orderBy("item_class"))

# COMMAND ----------

# DBTITLE 1,Step 1 — examples per class
# STEP 1 (cont.) — 20 random examples per item_class.
ITEMS = spark.table(f"{SCHEMA}.facility_items")
for c in ["empty", "informative", "trust_signal", "uninformative", "unclear"]:
    print(f"\n----- 20 random '{c}' items -----")
    display(ITEMS.filter(F.col("item_class") == c).select("field", "item", "drop_reason").orderBy(F.rand(4)).limit(20))

# COMMAND ----------

# DBTITLE 1,Step 2 — requirements map (service_requirements)
# STEP 2 — requirements map -> service_requirements.
# First, the real specialty labels (human-readable) for grounding the code mappings.
labels = (GOLD.select(F.explode("specialty_labels").alias("label")).groupBy("label").count().orderBy(F.desc("count")))
print(f"distinct specialty_labels: {labels.count():,} (top 60 shown)")
display(labels.limit(60))

codes = {r["c"] for r in GOLD.select(F.explode(F.coalesce(F.col("specialty_codes"), F.array().cast("array<string>"))).alias("c")).distinct().collect()}
print(f"distinct specialty_codes in gold: {len(codes):,}")
for sid, cfg in SERVICES.items():
    for code in cfg["specialty_codes"] + cfg["supporting_codes"]:
        assert code in codes, f"{sid}: mapped specialty code '{code}' not present in gold.specialty_codes"
assert "interventionalCardiology" not in codes, "interventionalCardiology unexpectedly present — update the angioplasty mapping"
print("All mapped specialty codes verified present in gold.")

req_rows = [("(any)", "specificity", SPECIFICITY_RE)]
for sid, cfg in SERVICES.items():
    req_rows.append((sid, "core", cfg["core"]))
    req_rows.append((sid, "supporting", cfg["supporting"]))
    req_rows.extend((sid, "specialty", code) for code in cfg["specialty_codes"])
    req_rows.extend((sid, "specialty_supporting", code) for code in cfg["supporting_codes"])
service_requirements = spark.createDataFrame(req_rows, "service_id string, term_type string, pattern string")
write_table(service_requirements, "service_requirements")
display(service_requirements.orderBy("service_id", "term_type"))

# COMMAND ----------

# DBTITLE 1,Facility dimensions (boilerplate, type, doctors)
# Facility dimensions used by matching: boilerplate flag (03b join), eye/dental heuristics, number_doctors.
sfp = spark.table(SFP_TABLE).select("unique_id", F.col("possible_boilerplate").alias("boilerplate"))
unmatched = GOLD.select("unique_id").join(sfp, "unique_id", "left").filter(F.col("boilerplate").isNull()).count()
print(f"service_fields_parsed join: unmatched unique_ids = {unmatched} (expected 0)")

fac = (GOLD.select("unique_id", "name", "facility_type", "number_doctors",
                   F.coalesce(F.col("specialty_codes"), F.array().cast("array<string>")).alias("specialty_codes"))
       .join(sfp, "unique_id", "left")
       .withColumn("boilerplate", F.coalesce(F.col("boilerplate"), F.lit(False)))
       .withColumn("is_dental", (F.col("facility_type") == "dentist") | F.lower(F.col("name")).rlike(r"\bdent"))
       .withColumn("is_eye", F.lower(F.col("name")).rlike(r"\bey(e|es)\b|eye care|ophthalm"))
       .cache())
print(f"facilities: {fac.count():,} | dentist-or-dental-named: {fac.filter('is_dental').count():,} | "
      f"eye-named: {fac.filter('is_eye').count():,} | boilerplate: {fac.filter('boilerplate').count():,} | "
      f"number_doctors<=2 (known): {fac.filter('number_doctors IS NOT NULL AND number_doctors <= 2').count():,}")

# COMMAND ----------

# DBTITLE 1,Step 3 — match per facility × service
# STEP 3 — per facility x service match (native Spark expressions only).
# Scope: specialties + description always; procedure/equipment/capability only if informative or unclear.
scope = (spark.table(f"{SCHEMA}.facility_items")
         .filter(F.col("field").isin("specialties", "description") | F.col("item_class").isin("informative", "unclear"))
         .select("unique_id", "field", "pos", "item", "item_key").cache())
print(f"match-scope items: {scope.count():,}")

PEC = ["procedure", "equipment", "capability"]
EMPTY_ARR = F.array().cast("array<string>")
parts = []
for sid, cfg in SERVICES.items():
    spec_arr = F.array(*[F.lit(c) for c in cfg["specialty_codes"]]).cast("array<string>") if cfg["specialty_codes"] else EMPTY_ARR
    supp_arr = F.array(*[F.lit(c) for c in cfg["supporting_codes"]]).cast("array<string>") if cfg["supporting_codes"] else EMPTY_ARR
    m = (scope
        .withColumn("is_core", F.col("field").isin(*PEC) & F.col("item_key").rlike(cfg["core"]))
        .withColumn("is_supp", F.col("field").isin(*PEC) & F.col("item_key").rlike(cfg["supporting"]))
        .withColumn("desc_core", (F.col("field") == "description") & F.col("item_key").rlike(cfg["core"]))
        .withColumn("matched_item", F.col("is_core") | F.col("is_supp") | F.col("desc_core"))
        .withColumn("spec_hit", F.col("matched_item") & F.col("item_key").rlike(SPECIFICITY_RE)))
    per_fac = (m.groupBy("unique_id").agg(
        F.collect_set(F.when(F.col("is_core"), F.col("field"))).alias("core_fields"),
        F.collect_set(F.when(F.col("is_supp"), F.col("field"))).alias("supp_fields_pec"),
        F.size(F.collect_set(F.when(F.col("is_supp"), F.col("item_key")))).alias("n_supp_terms_pec"),
        F.max(F.when(F.col("desc_core"), 1).otherwise(0)).alias("description_mention"),
        F.max(F.when(F.col("spec_hit"), 1).otherwise(0)).alias("specificity_hit"),
        F.collect_list(
            F.when(
                F.col("matched_item"),
                F.struct(
                    F.when(F.col("is_core"), 0).otherwise(1).alias("k1"),
                    F.when(F.col("field") == "procedure", 0).when(F.col("field") == "equipment", 1)
                         .when(F.col("field") == "capability", 2).otherwise(3).alias("k2"),
                    F.col("pos").alias("k3"),
                    F.col("field").alias("f"),
                    F.col("item").alias("t"),
                ),
            ),
        ).alias("matched"),
    ))
    row = (per_fac.join(fac, "unique_id", "right")  # every facility gets a row per service
        .withColumn("core_fields", F.coalesce(F.col("core_fields"), EMPTY_ARR))
        .withColumn("supp_fields_pec", F.coalesce(F.col("supp_fields_pec"), EMPTY_ARR))
        .withColumn("n_supp_terms_pec", F.coalesce(F.col("n_supp_terms_pec"), F.lit(0)))
        .withColumn("description_mention", F.coalesce(F.col("description_mention"), F.lit(0)) > 0)
        .withColumn("specificity_hit", F.coalesce(F.col("specificity_hit"), F.lit(0)) > 0)
        .withColumn("specialty_support", F.size(F.array_intersect(F.col("specialty_codes"), spec_arr)) > 0)
        .withColumn("has_supp_code", F.size(F.array_intersect(F.col("specialty_codes"), supp_arr)) > 0)
        .withColumn("supporting_fields",
             F.when(F.col("has_supp_code"), F.array_union(F.col("supp_fields_pec"), F.array(F.lit("specialties"))))
              .otherwise(F.col("supp_fields_pec")))
        .withColumn("n_supporting_terms", F.col("n_supp_terms_pec") + F.when(F.col("has_supp_code"), 1).otherwise(0))
        .withColumn("evidence_snippets",
             F.transform(F.slice(F.array_sort(F.col("matched")), 1, 3), lambda x: F.concat(x["f"], F.lit(": "), x["t"])))
        .withColumn("service_id", F.lit(sid))
        .withColumn("contradiction", F.lit(sid in CONTRADICTION_SERVICES) &
                    (F.col("is_dental") | F.col("is_eye") |
                     (F.col("number_doctors").isNotNull() & (F.col("number_doctors") <= 2))))
        .select("unique_id", "service_id", "core_fields", "supporting_fields", "specialty_support",
                "description_mention", "specificity_hit", "n_supporting_terms", "boilerplate",
                "contradiction", "evidence_snippets"))
    parts.append(row)
evidence = reduce(lambda a, b: a.unionByName(b), parts).cache()
print(f"facility x service rows: {evidence.count():,} (expected {N_GOLD * 7:,})")

# COMMAND ----------

# DBTITLE 1,Steps 4–5 — tiers, caps, score
# STEPS 4-5 — tiers (rules first, then caps) and the 0-1 ordering score.
has_informative = (spark.table(f"{SCHEMA}.facility_items").groupBy("unique_id")
    .agg(F.max(F.when(F.col("item_class") == "informative", 1).otherwise(0)).alias("has_informative")))

ev = (evidence.join(has_informative, "unique_id", "left")
      .withColumn("has_informative", F.coalesce(F.col("has_informative"), F.lit(0))))

core_n = F.size(F.col("core_fields"))
union_n = F.size(F.array_distinct(F.array_union(F.col("core_fields"), F.col("supporting_fields"))))
strong_base = (core_n >= 1) & ((union_n >= 2) | F.col("specificity_hit"))
strong = strong_base | ((F.col("service_id") == "c_section") & (core_n >= 1) & (F.col("n_supporting_terms") >= 2))

tier = (F.when(strong, F.lit("strong"))
    .when(core_n >= 1, F.lit("moderate"))
    .when(F.col("specialty_support") | F.col("description_mention") | (F.size(F.col("supporting_fields")) > 0), F.lit("weak"))
    .when(F.col("has_informative") == 1, F.lit("none"))
    .otherwise(F.lit("no_data")))
# caps: boilerplate can't be strong without a specificity hit; contradiction caps the tier at weak.
tier = F.when((tier == "strong") & F.col("boilerplate") & ~F.col("specificity_hit"), F.lit("moderate")).otherwise(tier)
tier = F.when(tier.isin("strong", "moderate") & F.col("contradiction"), F.lit("weak")).otherwise(tier)

score = (F.lit(0.40) * F.when(core_n >= 1, 1.0).otherwise(0.0)
       + F.lit(0.20) * F.when(union_n >= 2, 1.0).otherwise(0.0)
       + F.lit(0.15) * F.when(F.col("specificity_hit"), 1.0).otherwise(0.0)
       + F.lit(0.10) * F.when(F.col("specialty_support"), 1.0).otherwise(0.0)
       + F.lit(0.05) * F.when(F.col("description_mention"), 1.0).otherwise(0.0)
       - F.lit(0.15) * F.when(F.col("boilerplate"), 1.0).otherwise(0.0)
       - F.lit(0.30) * F.when(F.col("contradiction"), 1.0).otherwise(0.0))
score = F.least(F.greatest(score, F.lit(0.0)), F.lit(1.0))

ev = ev.withColumn("evidence_tier", tier).withColumn("evidence_score", score)
print("Tier counts (facility x service pairs):")
display(ev.groupBy("evidence_tier").count().orderBy("evidence_tier"))

# COMMAND ----------

# DBTITLE 1,Step 6 — plain-language evidence text
# STEP 6 — plain-language evidence text (1-2 short sentences, quoting snippets word for word).
svc_info = spark.createDataFrame(
    [(sid, SVC_DISPLAY[sid], SVC_SPECIALTY_PHRASE.get(sid), SVC_WEAK_LABEL.get(sid)) for sid in SERVICES],
    "service_id string, svc_display string, specialty_phrase string, weak_label string")
ev = ev.join(svc_info, "service_id")

def piece(s):  # "procedure: text" -> "Procedure lists 'text'"
    fld = F.regexp_extract(s, r"^([a-z]+): ", 1)
    return F.concat(F.upper(F.substring(fld, 1, 1)), F.substring(fld, 2, 12),
                    F.lit(" lists '"), F.regexp_extract(s, r": (.*)$", 1), F.lit("'"))
joined = F.concat_ws("; ", F.transform(F.col("evidence_snippets"), piece))

sm_text = F.concat(joined,
    F.when(F.col("specialty_support") & F.col("specialty_phrase").isNotNull(),
           F.concat(F.lit("; "), F.col("specialty_phrase"))).otherwise(F.lit("")),
    F.lit("."))
weak_text = (F.when(F.size(F.col("evidence_snippets")) > 0,
              F.concat(F.lit("Only supporting evidence: "), joined, F.lit("; no explicit mention of "), F.col("svc_display"), F.lit(".")))
            .when(F.array_contains(F.col("supporting_fields"), "specialties"),
              F.concat(F.lit("Only a related specialty ("), F.coalesce(F.col("weak_label"), F.lit("a related specialty")),
                       F.lit(") is listed; no direct mention of "), F.col("svc_display"), F.lit(".")))
            .when(F.col("specialty_support"),
              F.concat(F.lit("Only the listed specialty points to "), F.col("svc_display"),
                       F.lit("; no procedure, equipment, or capability items mention it.")))
            .otherwise(F.concat(F.lit("Only the free-text description mentions "), F.col("svc_display"),
                       F.lit("; no structured field lists it."))))
none_text = F.concat(F.lit("No mention of "), F.col("svc_display"),
    F.lit(" anywhere in this record; other services are documented here, so this reads as not evidenced rather than absent."))
nodata_text = F.concat(F.lit("No usable service details on record; whether this facility offers "),
    F.col("svc_display"), F.lit(" is unknown."))

tier_text = (F.when(F.col("evidence_tier").isin("strong", "moderate"), sm_text)
             .when(F.col("evidence_tier") == "weak", weak_text)
             .when(F.col("evidence_tier") == "none", none_text)
             .otherwise(nodata_text))
tier_text = F.when((F.col("evidence_tier").isin("strong", "moderate")) & F.col("boilerplate"),
    F.concat(tier_text, F.lit(" Note: record lists an unusual number of specialties (possible boilerplate)."))).otherwise(tier_text)
tier_text = F.when((F.col("evidence_tier").isin("strong", "moderate", "weak")) & F.col("contradiction"),
    F.concat(tier_text, F.lit(" Claim is doubtful: dentist/eye facility or at most 2 doctors."))).otherwise(tier_text)

ev = ev.withColumn("evidence_text", tier_text)
display(ev.select("service_id", "evidence_tier", "evidence_snippets", "evidence_text").limit(5))

# COMMAND ----------

# DBTITLE 1,Output — facility_service_evidence + assertions
# OUTPUT — facility_service_evidence + assertions.
fse = ev.select("unique_id", "service_id", "evidence_tier", "evidence_score", "core_fields", "supporting_fields",
                "specialty_support", "specificity_hit", "boilerplate", "contradiction",
                "evidence_snippets", "evidence_text")
write_table(fse, "facility_service_evidence")
FSE = spark.table(f"{SCHEMA}.facility_service_evidence")

n_rows, n_pairs = FSE.count(), FSE.select("unique_id", "service_id").distinct().count()
assert N_GOLD == 9969, f"gold facility count changed: {N_GOLD}"
assert n_rows == N_GOLD * 7, f"expected {N_GOLD * 7} rows, got {n_rows}"
assert n_pairs == n_rows, f"duplicate (unique_id, service_id) pairs: {n_rows - n_pairs}"
assert FSE.groupBy("unique_id").count().filter(F.col("count") != 7).count() == 0, "some facility lacks all 7 service rows"
tiers = {r["evidence_tier"] for r in FSE.select("evidence_tier").distinct().collect()}
assert tiers <= {"strong", "moderate", "weak", "none", "no_data"}, f"unexpected tiers: {tiers}"
rng = FSE.agg(F.min("evidence_score").alias("mn"), F.max("evidence_score").alias("mx")).collect()[0]
assert rng["mn"] >= 0 and rng["mx"] <= 1, f"score out of [0,1]: {rng}"
print(f"ALL ASSERTIONS PASSED — {n_rows:,} rows = {N_GOLD:,} facilities x 7 services, one row per pair; "
      f"tiers {sorted(tiers)}; score range [{rng['mn']}, {rng['mx']}]")

# COMMAND ----------

# DBTITLE 1,Output — benchmark_evidence_labels
# OUTPUT — benchmark_evidence_labels: ~13 pairs per service, spread across tiers, human_label left empty
# (values to use when labelling: offers / does_not / unclear).
FSE2 = spark.table(f"{SCHEMA}.facility_service_evidence")
seed = 20261121
parts = []
for sid in SERVICES:
    counts = {r["evidence_tier"]: r["count"] for r in FSE2.filter(F.col("service_id") == sid).groupBy("evidence_tier").count().collect()}
    present = [t for t in ["strong", "moderate", "weak", "none", "no_data"] if counts.get(t, 0) > 0]
    alloc = {t: 1 for t in present}
    remaining = 13 - len(present)
    while remaining > 0:  # fill proportionally to tier size (largest capacity first)
        best = max(present, key=lambda t: counts[t] - alloc.get(t, 0))
        if counts[best] - alloc.get(best, 0) <= 0:
            break
        alloc[best] += 1
        remaining -= 1
    print(f"{sid}: tier counts {counts} -> sample {alloc} (total {sum(alloc.values())})")
    for t, k in alloc.items():
        parts.append(FSE2.filter((F.col("service_id") == sid) & (F.col("evidence_tier") == t))
                     .orderBy(F.rand(seed)).limit(k))
bench = reduce(lambda a, b: a.unionByName(b), parts)
bench = (bench.join(GOLD.select("unique_id", "facility_type", "state", "number_doctors", "specialty_labels",
                                 "procedures", "equipment", "capabilities", "description"), "unique_id")
         .select("unique_id", "service_id", "evidence_tier", "evidence_score", "evidence_snippets", "evidence_text",
                 "facility_type", "state", "number_doctors", "specialty_labels", "procedures", "equipment",
                 "capabilities", "description", F.lit("").alias("human_label")))
write_table(bench, "benchmark_evidence_labels")
display(spark.table(f"{SCHEMA}.benchmark_evidence_labels").groupBy("service_id", "evidence_tier").count().orderBy("service_id", "evidence_tier"))

# COMMAND ----------

# DBTITLE 1,Summary — tiers, boilerplate share, 03b comparison
# SUMMARY — service x tier; boilerplate share among strong+moderate; comparison vs 03b 'any match'.
FSE3 = spark.table(f"{SCHEMA}.facility_service_evidence")
print("Counts by service x evidence tier:")
display(FSE3.groupBy("service_id").pivot("evidence_tier", ["strong", "moderate", "weak", "none", "no_data"]).count().orderBy("service_id"))

print("\nBoilerplate share among strong+moderate:")
sm = (FSE3.filter(F.col("evidence_tier").isin("strong", "moderate"))
      .groupBy("service_id").agg(F.count("*").alias("n_sm"), F.sum(F.col("boilerplate").cast("int")).alias("n_boilerplate")))
sm = sm.withColumn("boilerplate_share", F.when(F.col("n_sm") > 0, F.col("n_boilerplate") / F.col("n_sm")))
overall = (FSE3.filter(F.col("evidence_tier").isin("strong", "moderate"))
           .agg(F.count("*").alias("n_sm"), F.sum(F.col("boilerplate").cast("int")).alias("n_boilerplate"))
           .withColumn("service_id", F.lit("(overall)"))
           .withColumn("boilerplate_share", F.when(F.col("n_sm") > 0, F.col("n_boilerplate") / F.col("n_sm"))))
display(sm.unionByName(overall).select("service_id", "n_sm", "n_boilerplate", "boilerplate_share"))

print("\nComparison vs 03b 'any match' (caveats: 03b universe is 10,076 facilities incl. 107 rows gold dropped; "
      "03b counted matches in items this notebook classifies uninformative; angioplasty not profiled in 03b -> null):")
m03b = spark.table(M03B_SUMMARY).withColumnRenamed("service", "service_id").select("service_id", "n_facilities_with_match")
n04 = (FSE3.groupBy("service_id")
       .agg(F.sum(F.when(F.col("evidence_tier").isin("strong", "moderate", "weak"), 1).otherwise(0)).alias("n_04_with_evidence")))
display(n04.join(m03b, "service_id", "left").select("service_id", "n_04_with_evidence", "n_facilities_with_match").orderBy("service_id"))

# COMMAND ----------

# DBTITLE 1,Examples — 5 evidence_text per tier
# 5 example evidence_text values per tier.
for t in ["strong", "moderate", "weak", "none", "no_data"]:
    print(f"\n----- {t} -----")
    display(spark.table(f"{SCHEMA}.facility_service_evidence").filter(F.col("evidence_tier") == t)
            .select("service_id", "evidence_text").orderBy(F.rand(7)).limit(5))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT service_id, evidence_tier, COUNT(*) AS n
# MAGIC FROM workspace.datathon_2026.facility_service_evidence
# MAGIC GROUP BY service_id, evidence_tier
# MAGIC ORDER BY service_id, evidence_tier;

# COMMAND ----------

# DBTITLE 1,How service evidence is decided
# MAGIC %md
# MAGIC ## How service evidence is decided
# MAGIC
# MAGIC All evidence is **self-reported** (scraped from each facility's own web presence) — nothing here is independently verified.
# MAGIC
# MAGIC **Evidence sources.** Items are the individual entries of the facility's `procedures`, `equipment`, `capabilities` and `specialties` lists plus its free-text `description`, deduplicated per facility and field. Only items classified `informative` or `unclear` (plus specialties and description, always kept) can support a claim; directory listings, addresses, amenities, contact details, opening hours, doctor names, and vague marketing are classified `uninformative` and ignored.
# MAGIC
# MAGIC **Term matching.** Each of the 7 services has a **core** regex (the service by name), a **supporting** regex (adjacent capability, e.g. "av fistula" for dialysis), mapped **specialty codes** (count as `specialty_support`), and related-specialty codes that count only as supporting evidence (`specialty_supporting`). A **specificity** pattern (counts, brand names like Siemens / 1.5 Tesla) applies to any service.
# MAGIC
# MAGIC **Tiers, in order:**
# MAGIC * **strong** — a core term in `procedure`/`equipment`/`capability` **and** either a core-or-supporting match in a second distinct field **or** a specificity hit. For c_section only: one core match + at least 2 distinct supporting terms is also strong.
# MAGIC * **moderate** — a core term in one of the three structured fields, nothing more.
# MAGIC * **weak** — no core match, but a supporting term, a mapped specialty label, or a core mention in the description only.
# MAGIC * **none** — nothing matched, but the facility documents other services (reads as "not evidenced", not "absent").
# MAGIC * **no_data** — the facility has no informative items at all (unknown).
# MAGIC
# MAGIC **Caps.** A facility flagged `possible_boilerplate` in notebook 03b (50+ repeated specialty labels) cannot be **strong** unless it has a specificity hit. A nicu / cardiac_surgery / angioplasty / burn_care claim at a dentist or eye-named facility, or at a facility with ≤2 doctors, is a **contradiction** and is capped at **weak**.
# MAGIC
# MAGIC **Score** (ordering within a tier, clipped to [0, 1]): 0.40 first core field + 0.20 second distinct core/supporting field + 0.15 specificity hit + 0.10 specialty support + 0.05 description mention − 0.15 boilerplate − 0.30 contradiction.
# MAGIC
# MAGIC **Why text.** `evidence_text` quotes the exact supporting items (word for word, original casing) so the recommender can answer "WHY this facility?" without re-reading the raw record.