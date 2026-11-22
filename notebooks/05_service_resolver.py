# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Service resolver
# MAGIC
# MAGIC **Purpose.** Users of the app type a short word or phrase for the healthcare service they need — "dialysis", "bypass", "baby icu", "dialisis". This notebook turns that phrase into one of our 7 checked services, asks for clarification when it is ambiguous, and falls back to an AI-generated search mapping when the service is not one we have defined. Every result records **HOW** it was resolved, so the app can explain itself.
# MAGIC
# MAGIC **All logic lives in Python functions** (`resolve_service` and its helpers) that the app can import later; they read their configuration from the tables below, not from notebook state.
# MAGIC
# MAGIC **Inputs (read-only).**
# MAGIC * `workspace.datathon_2026.gold_facility_search` — the only valid specialty labels are the distinct camelCase values of `specialty_codes`.
# MAGIC * `workspace.datathon_2026.facility_items` — item-level text with `item_class` (notebook 04).
# MAGIC * `workspace.datathon_2026.service_requirements` — patterns per service (notebook 04).
# MAGIC
# MAGIC **Outputs.**
# MAGIC * `service_catalog`, `service_synonyms`, `benchmark_service_queries` — new tables, CREATE OR REPLACE on rerun.
# MAGIC * `generated_mappings`, `query_log_service` — new tables; after creation they are **appended to only** (the query log grows; generated mappings grow via a merge that bumps `times_requested`).
# MAGIC
# MAGIC No existing table is ever modified. LLM calls go through `ai_query` against `LLM_ENDPOINT` (a Unity Gateway model service) and are each wrapped in try/except — if the LLM fails, the resolver continues as if `use_llm=False`.

# COMMAND ----------

# DBTITLE 1,Config — schema, endpoint, helpers
import difflib, json, re
from datetime import datetime
from pyspark.sql import functions as F

SCHEMA = "workspace.datathon_2026"
LLM_ENDPOINT = "system.ai.meta-llama-3-3-70b-instruct"   # Unity Gateway model service, used by ai_query

def write_table(df, name):
    """CREATE-OR-REPLACE style write, only ever used on this notebook's own new tables."""
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{SCHEMA}.{name}")
    print(f"wrote {SCHEMA}.{name}: {spark.table(f'{SCHEMA}.{name}').count():,} rows")

def _table_exists(name):
    # Spark Connect resolves spark.table() lazily (a missing table does NOT raise), so use the catalog API.
    try:
        return spark.catalog.tableExists(f"{SCHEMA}.{name}")
    except Exception:
        try:
            return not spark.table(f"{SCHEMA}.{name}").isEmpty()
        except Exception:
            return False

print("resolver config ready — LLM endpoint:", LLM_ENDPOINT)

# COMMAND ----------

# DBTITLE 1,Step 1 — service_catalog
# STEP 1 — service_catalog: the 7 checked services with urgency, max_km and a one-line description.
CATALOG_ROWS = [
    ("dialysis",        "Dialysis",                     "recurring", 100, "Regular blood filtering for kidney failure"),
    ("c_section",       "Emergency C-section",          "emergency",   60, "Surgical delivery of a baby, available urgently"),
    ("nicu",            "Newborn intensive care (NICU)", "emergency", 100, "Intensive care for premature or sick newborns"),
    ("cardiac_surgery", "Heart surgery",                "planned",   300, "Bypass, valve and open-heart surgery"),
    ("angioplasty",     "Angioplasty / cath lab",       "emergency", 150, "Opening blocked heart arteries with a catheter or stent"),
    ("mri",             "MRI scan",                     "planned",   200, "Magnetic resonance imaging"),
    ("burn_care",       "Burn care",                    "emergency", 150, "Specialized treatment of serious burns"),
]
catalog = spark.createDataFrame(
    CATALOG_ROWS,
    "service_id string, display_name string, urgency string, max_km int, description string")
write_table(catalog, "service_catalog")

# every catalog service must have matching patterns from notebook 04
req_ids = {r["service_id"] for r in
           spark.table(f"{SCHEMA}.service_requirements").select("service_id").distinct().collect()
           if r["service_id"] != "(any)"}
cat_ids = {row[0] for row in CATALOG_ROWS}
assert cat_ids == req_ids, (
    f"catalog and service_requirements disagree: "
    f"catalog-only={cat_ids - req_ids}, requirements-only={req_ids - cat_ids}")
print("All 7 catalog service_ids verified against service_requirements.")
display(spark.table(f"{SCHEMA}.service_catalog"))

# COMMAND ----------

# DBTITLE 1,Step 2 — service_synonyms
# STEP 2 — service_synonyms: phrase -> service_id (a phrase may map to several services).
SYNONYM_ROWS = [  # (phrase, service_id)
    # dialysis
    ("dialysis", "dialysis"), ("haemodialysis", "dialysis"), ("hemodialysis", "dialysis"),
    ("kidney dialysis", "dialysis"), ("kidney machine", "dialysis"), ("renal dialysis", "dialysis"),
    ("peritoneal dialysis", "dialysis"), ("capd", "dialysis"), ("kidney failure", "dialysis"),
    # c_section
    ("c-section", "c_section"), ("c section", "c_section"), ("csection", "c_section"),
    ("caesarean", "c_section"), ("cesarean", "c_section"), ("caesarean section", "c_section"),
    ("lscs", "c_section"), ("emergency c-section", "c_section"), ("operative delivery", "c_section"),
    ("surgical delivery", "c_section"),
    # nicu
    ("nicu", "nicu"), ("sncu", "nicu"), ("baby icu", "nicu"), ("newborn icu", "nicu"),
    ("neonatal icu", "nicu"), ("neonatal intensive care", "nicu"), ("premature baby", "nicu"),
    ("preterm baby", "nicu"), ("newborn intensive care", "nicu"),
    # cardiac_surgery
    ("heart surgery", "cardiac_surgery"), ("cardiac surgery", "cardiac_surgery"),
    ("bypass", "cardiac_surgery"), ("bypass surgery", "cardiac_surgery"), ("cabg", "cardiac_surgery"),
    ("open heart surgery", "cardiac_surgery"), ("valve replacement", "cardiac_surgery"),
    ("valve surgery", "cardiac_surgery"), ("cardiothoracic surgery", "cardiac_surgery"),
    # angioplasty
    ("angioplasty", "angioplasty"), ("stent", "angioplasty"), ("ptca", "angioplasty"),
    ("cath lab", "angioplasty"), ("heart attack", "angioplasty"), ("cardiac catheterization", "angioplasty"),
    # mri
    ("mri", "mri"), ("mri scan", "mri"), ("magnetic resonance", "mri"),
    ("brain mri", "mri"), ("spine mri", "mri"),
    # burn_care
    ("burn", "burn_care"), ("burns", "burn_care"), ("burn care", "burn_care"),
    ("burn unit", "burn_care"), ("burn ward", "burn_care"), ("burn injury", "burn_care"),
    ("fire injury", "burn_care"),
    # ambiguous: both heart services
    ("heart", "cardiac_surgery"), ("heart", "angioplasty"),
    ("heart treatment", "cardiac_surgery"), ("heart treatment", "angioplasty"),
    ("cardiac", "cardiac_surgery"), ("cardiac", "angioplasty"),
    ("heart problem", "cardiac_surgery"), ("heart problem", "angioplasty"),
]
syn = spark.createDataFrame(
    [(p, s, " ".join(p.lower().split())) for p, s in SYNONYM_ROWS],
    "phrase string, service_id string, phrase_norm string")
write_table(syn, "service_synonyms")

n_phrases = syn.select("phrase_norm").distinct().count()
n_amb = syn.groupBy("phrase_norm").count().filter("count > 1").count()
print(
    f"{len(SYNONYM_ROWS)} synonym rows, {n_phrases} distinct phrases, "
    f"{n_amb} ambiguous phrases (heart, heart treatment, cardiac, heart problem)")

# COMMAND ----------

# DBTITLE 1,Step 3a — loaders
# STEP 3a — resolver building blocks. Everything loads from the tables so the app can import these functions.
_CACHE = {}

def _catalog():
    """service_id -> {display_name, description, urgency, max_km}; loaded once per session."""
    if "catalog" not in _CACHE:
        rows = {}
        for r in spark.table(f"{SCHEMA}.service_catalog").collect():
            rows[r["service_id"]] = {
                "display_name": r["display_name"], "description": r["description"],
                "urgency": r["urgency"], "max_km": r["max_km"]}
        _CACHE["catalog"] = rows
    return _CACHE["catalog"]

def _synonyms():
    """(sorted phrase list, phrase_norm -> ordered service_ids); loaded once per session."""
    if "synonyms" not in _CACHE:
        m = {}
        for r in spark.table(f"{SCHEMA}.service_synonyms").select("phrase_norm", "service_id").collect():
            m.setdefault(r["phrase_norm"], [])
            if r["service_id"] not in m[r["phrase_norm"]]:
                m[r["phrase_norm"]].append(r["service_id"])
        _CACHE["synonyms"] = (sorted(m), m)
    return _CACHE["synonyms"]

def _n_facilities():
    """Distinct facilities in facility_items — denominator for the 30% breadth rule."""
    if "n_fac" not in _CACHE:
        _CACHE["n_fac"] = spark.table(f"{SCHEMA}.facility_items").select("unique_id").distinct().count()
    return _CACHE["n_fac"]

def _real_specialty_labels():
    """The only allowed specialty labels: distinct camelCase codes from gold_facility_search."""
    if "labels" not in _CACHE:
        codes = (
            spark.table(f"{SCHEMA}.gold_facility_search")
            .select(F.explode(F.coalesce(F.col("specialty_codes"), F.array().cast("array<string>"))).alias("c"))
            .distinct()
            .collect())
        _CACHE["labels"] = {r["c"] for r in codes}
    return _CACHE["labels"]

def _options_for(service_ids):
    cat = _catalog()
    return [
        {"service_id": s, "display_name": cat[s]["display_name"], "description": cat[s]["description"]}
        for s in service_ids
    ]

def _normalize(text):
    return " ".join(str(text).lower().split())

def _clean_list(values, lowercase=False):
    """Trim, optionally lowercase, and de-dup a list of terms; keeps camelCase labels intact."""
    if not isinstance(values, list):
        return []
    seen, out = set(), []
    for v in values:
        s = str(v).strip()
        if lowercase:
            s = s.lower()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out

_phrases, _syn = _synonyms()
print(
    f"resolver ready: {len(_catalog())} services, {len(_syn)} phrases "
    f"({sum(1 for v in _syn.values() if len(v) > 1)} ambiguous), "
    f"{len(_real_specialty_labels())} real specialty labels, "
    f"{_n_facilities():,} facilities in facility_items")

# COMMAND ----------

# DBTITLE 1,Step 3b — LLM, cache, log helpers
# STEP 3b — LLM helpers (ai_query), the generated_mappings cache, and the query log (STEP 4).
# NOTE: ai_query's responseFormat allows exactly ONE top-level field and it must be a STRUCT, so each
# response schema below wraps the fields in a single wrapper struct. With responseFormat, ai_query
# returns a JSON string; the payload holds the INNER struct's fields (verified against this endpoint).

CATALOG_PICK_FORMAT = "STRUCT<answer STRUCT<service_id STRING>>"
MAPPING_FORMAT = (
    "STRUCT<mapping STRUCT<"
    "core_terms ARRAY<STRING>, supporting_terms ARRAY<STRING>, specialty_labels ARRAY<STRING>>>")

def _ai_query(prompt, response_format):
    """One ai_query call; returns the parsed JSON. Raises on any failure — callers must catch."""
    q = prompt.replace("'", "''")
    sql = (
        f"SELECT ai_query('{LLM_ENDPOINT}', '{q}', "
        f"responseFormat => '{response_format}', "
        f"modelParameters => named_struct('temperature', 0)) AS ans")
    raw = spark.sql(sql).collect()[0][0]
    if isinstance(raw, str):
        return json.loads(raw)
    return raw.asDict() if hasattr(raw, "asDict") else raw

def _unwrap(parsed, wrapper):
    """Some endpoints keep the wrapper field in the JSON; most return the inner fields directly."""
    if isinstance(parsed, dict) and isinstance(parsed.get(wrapper), dict):
        return parsed[wrapper]
    return parsed

def _llm_catalog_pick(norm):
    """Ask the model for exactly one catalog service_id, or NONE."""
    cat = _catalog()
    lines = "\n".join(
        f"- {sid}: {info['display_name']} — {info['description']}"
        for sid, info in sorted(cat.items()))
    prompt = (
        "A user typed a healthcare service into a search box. "
        "Decide which of our checked services it refers to.\n"
        f"Checked services:\n{lines}\n\n"
        f"User query: {norm}\n\n"
        "If the query clearly refers to one of these services, answer with that service_id exactly as written. "
        "If it does not clearly match any checked service, answer NONE.\n"
        "Answer with the service_id only.")
    parsed = _unwrap(_ai_query(prompt, CATALOG_PICK_FORMAT), "answer")
    sid = str((parsed or {}).get("service_id") or "").strip()
    return sid or None

def _llm_generate_mapping(norm):
    """Generate core/supporting search terms + real specialty labels for a non-catalog service.
    Returns a validated mapping dict, or None if no core terms survive validation."""
    labels_sorted = sorted(_real_specialty_labels())
    prompt = (
        "A user typed a healthcare service need into a search box. "
        "Our facility search matches records using:\n"
        "- core_terms: short phrases that must appear in a facility's procedures/equipment/capabilities "
        "for it to be a candidate (e.g. 'hemodialysis', 'burn unit').\n"
        "- supporting_terms: adjacent capability phrases that make a candidate more credible "
        "(e.g. 'nephrology', 'skin graft').\n"
        "- specialty_labels: the facility's specialty, chosen ONLY from the allowed list below "
        "(or an empty list if none fit).\n\n"
        f"User query: {norm}\n\n"
        f"Allowed specialty_labels (exact strings): {', '.join(labels_sorted)}\n\n"
        "Rules: 3-5 core terms (1-3 words each, lowercase, simple spellings a facility would actually write); "
        "2-4 supporting terms; 0-4 specialty labels from the allowed list only. "
        "If the query is not a real healthcare service or need (e.g. gibberish), return empty lists.\n"
        "Return only the JSON.")
    parsed = _unwrap(_ai_query(prompt, MAPPING_FORMAT), "mapping")
    parsed = parsed if isinstance(parsed, dict) else {}
    core = _clean_list(parsed.get("core_terms"), lowercase=True)
    supp = _clean_list(parsed.get("supporting_terms"), lowercase=True)
    labels = _clean_list(parsed.get("specialty_labels"))
    # validation 1: specialty labels must be real camelCase codes from gold_facility_search
    real = _real_specialty_labels()
    dropped_labels = [t for t in labels if t not in real]
    labels = [t for t in labels if t in real]
    # validation 2: a core term matching >30% of facilities is too broad to be a useful search term
    n_fac = _n_facilities()
    kept_core, dropped_core = [], []
    for t in core:
        n_match = (
            spark.table(f"{SCHEMA}.facility_items")
            .filter(F.col("item_class").isin("informative", "unclear"))
            .filter(F.col("item_key").rlike(re.escape(t)))
            .select("unique_id")
            .distinct()
            .count())
        if n_match / n_fac > 0.30:
            dropped_core.append(t)
        else:
            kept_core.append(t)
    if not kept_core:
        return None
    return {
        "core_terms": kept_core,
        "supporting_terms": supp,
        "specialty_labels": labels,
        "dropped_terms": dropped_core + dropped_labels,
    }

def _ensure_generated_mappings():
    if not _table_exists("generated_mappings"):
        spark.sql(
            f"CREATE OR REPLACE TABLE {SCHEMA}.generated_mappings ("
            "query_norm STRING, core_terms ARRAY<STRING>, supporting_terms ARRAY<STRING>, "
            "specialty_labels ARRAY<STRING>, dropped_terms ARRAY<STRING>, "
            "created_at TIMESTAMP, times_requested INT)")

def _get_cached_mapping(norm):
    _ensure_generated_mappings()
    rows = spark.table(f"{SCHEMA}.generated_mappings").filter(F.col("query_norm") == norm).limit(1).collect()
    if not rows:
        return None
    r = rows[0]
    return {
        "core_terms": list(r["core_terms"] or []),
        "supporting_terms": list(r["supporting_terms"] or []),
        "specialty_labels": list(r["specialty_labels"] or []),
        "dropped_terms": list(r["dropped_terms"] or []),
    }

def _save_generated_mapping(norm, mapping):
    """Insert a new generated mapping; on an existing one just bump times_requested (merge, never replace)."""
    _ensure_generated_mappings()
    new = spark.createDataFrame(
        [(norm, mapping["core_terms"], mapping["supporting_terms"], mapping["specialty_labels"],
          mapping.get("dropped_terms", []), datetime.now(), 1)],
        "query_norm string, core_terms array<string>, supporting_terms array<string>, "
        "specialty_labels array<string>, dropped_terms array<string>, "
        "created_at timestamp, times_requested int")
    new.createOrReplaceTempView("_gm_new")
    spark.sql(
        f"MERGE INTO {SCHEMA}.generated_mappings t "
        "USING _gm_new s ON t.query_norm = s.query_norm "
        "WHEN MATCHED THEN UPDATE SET t.times_requested = t.times_requested + 1 "
        "WHEN NOT MATCHED THEN INSERT *")

def _log_call(result):
    """STEP 4 — append every resolve_service call to query_log_service (created once, then append-only)."""
    row = spark.createDataFrame(
        [(datetime.now(), result["input"], result["normalized"], result["status"],
          result["method"], result["service_id"])],
        "timestamp timestamp, input string, normalized string, status string, "
        "method string, service_id string")
    if _table_exists("query_log_service"):
        row.write.mode("append").saveAsTable(f"{SCHEMA}.query_log_service")
    else:
        row.write.mode("overwrite").saveAsTable(f"{SCHEMA}.query_log_service")  # creation; later writes append

print("LLM + cache + log helpers defined.")

# COMMAND ----------

# DBTITLE 1,Step 3c — resolve_service
# STEP 3c — resolve_service(text, use_llm=True) -> dict with always the same keys (plus STEP 4 logging).
def _result(text, norm):
    return {
        "input": str(text),
        "normalized": norm,
        "status": None,
        "method": None,
        "service_id": None,
        "options": [],
        "matched_phrase": None,
        "mapping": None,
        "confidence_cap": None,
        "note": None,
    }

def resolve_service(text, use_llm=True):
    """Resolve a typed service phrase to a checked service, a clarification request,
    an AI-generated search mapping, or none — always with the same return keys."""
    norm = _normalize(text)
    res = _result(text, norm)
    cat = _catalog()
    phrases, syn = _synonyms()

    # (a) exact synonym match
    if norm in syn:
        sids = syn[norm]
        res["matched_phrase"] = norm
        if len(sids) == 1:
            res["status"], res["method"], res["service_id"] = "matched", "synonym", sids[0]
            res["note"] = f"Interpreted '{text}' as {cat[sids[0]]['display_name']} (synonym match)."
        else:
            res["status"], res["method"] = "ambiguous", "synonym"
            res["options"] = _options_for(sids)
            names = " or ".join(o["display_name"] for o in res["options"])
            res["note"] = f"'{text}' could mean {names} — please pick one."
        _log_call(res)
        return res

    # (b) close match (typo tolerance), difflib cutoff 0.85
    close = difflib.get_close_matches(norm, phrases, n=3, cutoff=0.85)
    if close:
        sids = []
        for p in close:
            for s in syn[p]:
                if s not in sids:
                    sids.append(s)
        res["matched_phrase"] = close[0]
        if len(sids) == 1:
            res["status"], res["method"], res["service_id"] = "matched", "fuzzy_synonym", sids[0]
            res["note"] = (
                f"Interpreted '{text}' as {cat[sids[0]]['display_name']} (close to '{close[0]}').")
        else:
            res["status"], res["method"] = "ambiguous", "fuzzy_synonym"
            res["options"] = _options_for(sids)
            names = " or ".join(o["display_name"] for o in res["options"])
            res["note"] = f"'{text}' could mean {names} — please pick one."
        _log_call(res)
        return res

    # LLM steps — every call wrapped; on failure continue as if use_llm=False
    llm_failed = False
    if use_llm:
        try:
            picked = _llm_catalog_pick(norm)
            if picked in cat:
                res["status"], res["method"], res["service_id"] = "matched", "llm_catalog_pick", picked
                res["note"] = f"Interpreted '{text}' as {cat[picked]['display_name']} (AI catalog pick)."
                _log_call(res)
                return res
        except Exception as e:
            llm_failed = True
            print(f"LLM catalog pick failed for {norm!r}: {e}")

    if use_llm and not llm_failed:
        try:
            cached = _get_cached_mapping(norm)
            if cached is not None:
                res["status"], res["method"] = "generated", "llm_generated"
                res["mapping"] = cached
                res["confidence_cap"] = "moderate"
                res["note"] = (
                    f"No checked service matched; reusing a previously generated search for '{text}'.")
                _save_generated_mapping(norm, cached)   # bump times_requested
                _log_call(res)
                return res
            generated = _llm_generate_mapping(norm)
            if generated is not None:
                res["status"], res["method"] = "generated", "llm_generated"
                res["mapping"] = generated
                res["confidence_cap"] = "moderate"
                _save_generated_mapping(norm, generated)
                res["note"] = (
                    f"No checked service matched; generated a search for '{text}' (moderate confidence).")
                _log_call(res)
                return res
        except Exception as e:
            llm_failed = True
            print(f"LLM mapping generation failed for {norm!r}: {e}")

    # (e) nothing matched
    res["status"] = "none"
    if use_llm and llm_failed:
        res["note"] = f"Could not resolve '{text}'; the AI fallback was unavailable."
    else:
        res["note"] = f"Could not resolve '{text}' to any checked service or search."
    _log_call(res)
    return res

print("resolve_service defined.")

# COMMAND ----------

# DBTITLE 1,Demos — resolve_service examples
# Quick demonstrations (these also seed the query log and the generated_mappings cache).
DEMO_QUERIES = [
    ("Dialysis", False),        # exact synonym (case-insensitive)
    ("kidney machine", False),  # exact synonym
    ("heart", False),           # ambiguous -> both heart services
    ("dialisis", False),        # typo -> close match
    ("heart problem", False),   # ambiguous phrase
    ("asdf", False),            # nothing
    ("brain scan", True),       # everyday phrase -> AI catalog pick
    ("physiotherapy", True),    # out-of-catalog service -> AI-generated search
]
for q, llm in DEMO_QUERIES:
    print(f"--- resolve_service({q!r}, use_llm={llm}) ---")
    print(json.dumps(resolve_service(q, use_llm=llm), indent=2))

# COMMAND ----------

# DBTITLE 1,Step 5a — benchmark queries
# STEP 5a — benchmark_service_queries (~40 rows): what each typed phrase SHOULD resolve to.
BENCH_ROWS = [
    # exact synonyms
    ("dialysis", "matched", "dialysis"), ("haemodialysis", "matched", "dialysis"),
    ("capd", "matched", "dialysis"), ("lscs", "matched", "c_section"),
    ("emergency c-section", "matched", "c_section"), ("baby icu", "matched", "nicu"),
    ("sncu", "matched", "nicu"), ("premature baby", "matched", "nicu"),
    ("bypass", "matched", "cardiac_surgery"), ("cabg", "matched", "cardiac_surgery"),
    ("valve replacement", "matched", "cardiac_surgery"), ("open heart surgery", "matched", "cardiac_surgery"),
    ("stent", "matched", "angioplasty"), ("cath lab", "matched", "angioplasty"),
    ("heart attack", "matched", "angioplasty"), ("mri scan", "matched", "mri"),
    ("spine mri", "matched", "mri"), ("burn unit", "matched", "burn_care"),
    ("burns", "matched", "burn_care"), ("fire injury", "matched", "burn_care"),
    # typos
    ("dialisis", "matched", "dialysis"), ("cardiac surgry", "matched", "cardiac_surgery"),
    ("cesarian", "matched", "c_section"), ("neonatl icu", "matched", "nicu"),
    ("kidny machine", "matched", "dialysis"), ("mri scn", "matched", "mri"),
    # everyday phrases (need the LLM catalog pick when use_llm=True)
    ("kidney failure treatment", "matched", "dialysis"),
    ("newborn breathing problems", "matched", "nicu"),
    ("heart bypass operation", "matched", "cardiac_surgery"),
    ("brain scan", "matched", "mri"),
    ("burn treatment", "matched", "burn_care"),
    ("baby in intensive care", "matched", "nicu"),
    ("stent surgery", "matched", "angioplasty"),
    ("dialysis machine", "matched", "dialysis"),
    # ambiguous (both heart services)
    ("heart", "ambiguous", None), ("cardiac", "ambiguous", None),
    ("heart treatment", "ambiguous", None),
    # out-of-catalog services (AI-generated search when use_llm=True)
    ("physiotherapy", "generated", None), ("chemotherapy", "generated", None),
    ("snakebite", "generated", None),
    # nonsense
    ("asdf", "none", None),
]
CATEGORY_SIZES = [("exact", 20), ("typo", 6), ("everyday", 8), ("ambiguous", 3),
                  ("out_of_catalog", 3), ("nonsense", 1)]
bench_rows = []
i = 0
for cat, n_cat in CATEGORY_SIZES:
    chunk = BENCH_ROWS[i:i + n_cat]
    assert len(chunk) == n_cat, f"category {cat} expected {n_cat} rows, got {len(chunk)}"
    for q, st, sid in chunk:
        bench_rows.append((q, st, sid, cat))
    i += n_cat
assert len(bench_rows) == len(BENCH_ROWS), "category sizes must cover all benchmark rows"
write_table(
    spark.createDataFrame(
        bench_rows,
        "query string, expected_status string, expected_service_id string, category string"),
    "benchmark_service_queries")
display(spark.table(f"{SCHEMA}.benchmark_service_queries"))

# COMMAND ----------

# DBTITLE 1,Step 5b — benchmark, use_llm=False
# STEP 5b — benchmark pass 1: use_llm=False (pure synonym + fuzzy matching).
bench_rows = [r.asDict() for r in spark.table(f"{SCHEMA}.benchmark_service_queries").collect()]
RES_SCHEMA = ("query string, category string, expected_status string, expected_service_id string, "
              "returned_status string, returned_method string, returned_service_id string, correct boolean")

def _run_benchmark(use_llm):
    recs = []
    for r in bench_rows:
        out = resolve_service(r["query"], use_llm=use_llm)
        ok = (out["status"] == r["expected_status"]) and (
            r["expected_service_id"] is None or out["service_id"] == r["expected_service_id"])
        recs.append((r["query"], r["category"], r["expected_status"], r["expected_service_id"],
                     out["status"], out["method"], out["service_id"], bool(ok)))
    return spark.createDataFrame(recs, RES_SCHEMA)

res_no_llm = _run_benchmark(False).cache()
n = res_no_llm.count()
nc = res_no_llm.filter("correct").count()
print(f"accuracy, use_llm=False: {nc}/{n} = {nc / n:.0%}")
display(res_no_llm.filter(~F.col("correct")).orderBy("category", "query"))

# COMMAND ----------

# DBTITLE 1,Step 5b — benchmark, use_llm=True
# STEP 5b (cont.) — benchmark pass 2: use_llm=True (catalog picks + generated mappings).
res_llm = _run_benchmark(True).cache()
n = res_llm.count()
nc = res_llm.filter("correct").count()
print(f"accuracy, use_llm=True: {nc}/{n} = {nc / n:.0%}")

print("\naccuracy by category, use_llm=True:")
display(
    res_llm.groupBy("category")
    .agg(F.count("*").alias("n"), F.sum(F.col("correct").cast("int")).alias("correct"))
    .withColumn("accuracy", F.col("correct") / F.col("n"))
    .orderBy("category"))

print("\nfailures, use_llm=True (expected vs returned):")
display(res_llm.filter(~F.col("correct")).orderBy("category", "query"))

print("\ngenerated mappings so far:")
display(spark.table(f"{SCHEMA}.generated_mappings"))

# COMMAND ----------

# DBTITLE 1,How a typed service is understood
# MAGIC %md
# MAGIC ## How a typed service is understood
# MAGIC
# MAGIC **The four steps, in plain English.** When a user types a phrase:
# MAGIC
# MAGIC 1. **Exact synonym.** The phrase is lowercased, trimmed and whitespace-collapsed, then looked up in `service_synonyms`. One service → matched. More than one → we ask for clarification (e.g. "heart" → Heart surgery or Angioplasty).
# MAGIC 2. **Close match.** Typos are handled with `difflib.get_close_matches` (cutoff 0.85) against all 59 phrases, so "dialisis" is interpreted as "dialysis".
# MAGIC 3. **AI catalog pick.** If it is still unresolved and the LLM is available, the model sees the 7 catalog descriptions and answers with exactly one `service_id` or NONE — so "brain scan" → mri, "kidney failure treatment" → dialysis.
# MAGIC 4. **AI-generated search.** If it is none of our services but still a real healthcare need (e.g. "physiotherapy", "snakebite"), the model generates core terms, supporting terms and specialty labels for a plain-text search. The result is validated — specialty labels must be real camelCase codes from `gold_facility_search`, and a core term matching more than 30% of facilities is too broad and dropped — then cached in `generated_mappings` for reuse.
# MAGIC
# MAGIC If everything fails — or the LLM is unavailable — the result is `none`: "could not resolve", never a wrong guess.
# MAGIC
# MAGIC **Why generated mappings are capped at moderate confidence.** A checked service (steps 1–3) is matched against curated synonyms and notebook 04's validated requirements. A generated mapping is a one-shot LLM suggestion: even after validation it has never been human-reviewed, the model can miss the Indian-English phrasings a facility actually writes, and its terms match text that only sounds adjacent to the service. The resolver therefore labels these results `confidence_cap = 'moderate'` so the app shows them as best-effort search results rather than checked services.
# MAGIC
# MAGIC **Benchmark** (41 queries: 20 exact synonyms, 6 typos, 8 everyday phrases, 3 ambiguous, 3 out-of-catalog, 1 nonsense):
# MAGIC
# MAGIC * `use_llm=False` (synonyms + fuzzy matching only): **30/41 = 73%** — all exact, typo, ambiguous and nonsense queries correct; the 11 misses are the 8 everyday phrases and 3 out-of-catalog services, which need the LLM.
# MAGIC * `use_llm=True`: **41/41 = 100%** — everyday phrases resolved via catalog picks, out-of-catalog services via validated generated mappings, and gibberish ("asdf") correctly still resolves to none.

# COMMAND ----------

