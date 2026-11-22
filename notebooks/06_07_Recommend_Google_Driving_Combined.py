# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 06 + 07 — Facility recommendations with real driving routes
# MAGIC
# MAGIC This is a **standalone Databricks notebook** combining the existing recommendation
# MAGIC engine (`06_recommend`) with Google Routes API road-distance calculations (`07`).
# MAGIC
# MAGIC **What changes:** Haversine distance still filters nearby candidates. The final
# MAGIC shortlisted candidates are routed over roads, and their **driving minutes** are used
# MAGIC for the access component in the ranking. Results show **road kilometers + minutes**, never
# MAGIC straight-line kilometers disguised as driving distance. No malaria friction raster is needed
# MAGIC for car routing.
# MAGIC
# MAGIC **Already-existing tables required:** `workspace.datathon_2026` tables produced by
# MAGIC earlier notebooks, including `service_catalog`, `service_synonyms`, `facility_items`,
# MAGIC `gold_facility_search`, `facility_service_evidence`, `facilities_located`, `place_lookup`,
# MAGIC `pincode_clean`, and `pin_lookup`. This notebook does **not** recreate them.
# MAGIC
# MAGIC **IMPORTANT:** Databricks Free Edition restricts outbound connections. If it blocks
# MAGIC `routes.googleapis.com`, the notebook reports `routing_unavailable` and **will not fake
# MAGIC a driving distance**. This cannot be bypassed by changing a Python setting.
# MAGIC
# MAGIC **Location caveat:** `Darbhanga` resolves to a **district centroid** from your postal data,
# MAGIC not necessarily downtown Darbhanga. Enter `latitude, longitude` for a precisely
# MAGIC comparable route. Google routes are traffic-unaware, not live arrival-time guarantees.

# COMMAND ----------

# CONFIGURATION: Paste your Google Maps API key into the quotes below.
# WARNING: This notebook cell may be visible to anyone who can view the notebook.
# Do not save/share a real key in a team notebook; preferably use a Databricks secret.

import os

GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

# Try Databricks secret if no environment variable exists
if not GOOGLE_MAPS_API_KEY:
    try:
        GOOGLE_MAPS_API_KEY = dbutils.secrets.get(
            scope="datathon-secrets",
            key="google-maps-key"
        )
    except Exception:
        pass

if not GOOGLE_MAPS_API_KEY:
    raise ValueError(
        "Google Maps API key missing. "
        "Configure GOOGLE_MAPS_API_KEY or a Databricks secret."
    )

# Google's free demo key has a limited request quota. Keep this small while testing.
MAX_GOOGLE_ROUTES = 5
REQUEST_TIMEOUT_SECONDS = 20
print("Google key provided:", bool(os.environ.get("GOOGLE_MAPS_API_KEY")))
print("Maximum Google route requests per search:", MAX_GOOGLE_ROUTES)

# COMMAND ----------

# Shared config — schema, LLM endpoint, table helpers. New tables only, via CREATE OR REPLACE;
# query_log_recommend is created once (empty) in STEP 6 and appended to afterwards.

import json, re, math, difflib
from datetime import datetime

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

SCHEMA = "workspace.datathon_2026"
LLM_ENDPOINT = "system.ai.meta-llama-3-3-70b-instruct"

ENABLE_QUERY_LOGGING = False  # set True only if you want the original notebook 05 query logs
_CACHE = {}   # session-level caches: catalog, synonyms, places, district radii, office names


def _table_exists(name):
    # Spark Connect resolves spark.table() lazily (a missing table does NOT raise), so use the catalog API.
    try:
        return spark.catalog.tableExists(f"{SCHEMA}.{name}")
    except Exception:
        try:
            return not spark.table(f"{SCHEMA}.{name}").isEmpty()
        except Exception:
            return False



def _title(s):
    """'darbhanga' -> 'Darbhanga' (labels only; all matching is on normalized lowercase)."""
    return " ".join(w.capitalize() for w in str(s or "").split())


def _arr(v):
    """Spark array column -> clean python list (handles None / NaN)."""
    if v is None:
        return []
    try:
        return [x for x in list(v) if x is not None]
    except TypeError:
        return []


def _hav_np(lat, lon, lats, lons):
    """Haversine km from one point to arrays of points."""
    la, lo = math.radians(lat), math.radians(lon)
    fla, flo = np.radians(lats), np.radians(lons)
    return (
        6371.0 * 2.0 * np.arcsin(np.sqrt(
            np.sin((fla - la) / 2.0) ** 2
            + math.cos(la) * np.cos(fla) * np.sin((flo - lo) / 2.0) ** 2)))


print("Recommendation tables:", SCHEMA)

# COMMAND ----------

# resolve_service — copied from notebook 05 (cells 3a/3b/3c) so this notebook is importable on its
# own; keep in sync with 05. Only difference: _ai_query also escapes backslashes, because Spark SQL
# treats a backslash as an escape character inside string literals (see the review comment on 05).

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


CATALOG_PICK_FORMAT = "STRUCT<answer STRUCT<service_id STRING>>"
MAPPING_FORMAT = (
    "STRUCT<mapping STRUCT<"
    "core_terms ARRAY<STRING>, supporting_terms ARRAY<STRING>, specialty_labels ARRAY<STRING>>>")


def _ai_query(prompt, response_format):
    """One ai_query call; returns the parsed JSON. Raises on any failure — callers must catch."""
    q = prompt.replace("\\", "\\\\").replace("'", "''")
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
    """Append every resolve_service call to query_log_service (created once, then append-only)."""
    if not ENABLE_QUERY_LOGGING:
        return
    row = spark.createDataFrame(
        [(datetime.now(), result["input"], result["normalized"], result["status"],
          result["method"], result["service_id"])],
        "timestamp timestamp, input string, normalized string, status string, "
        "method string, service_id string")
    if _table_exists("query_log_service"):
        row.write.mode("append").saveAsTable(f"{SCHEMA}.query_log_service")
    else:
        row.write.mode("overwrite").saveAsTable(f"{SCHEMA}.query_log_service")


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


_phrases, _syn = _synonyms()
print(
    f"resolver ready: {len(_catalog())} services, {len(_syn)} phrases "
    f"({sum(1 for v in _syn.values() if len(v) > 1)} ambiguous), "
    f"{len(_real_specialty_labels())} real specialty labels, "
    f"{_n_facilities():,} facilities in facility_items")

# COMMAND ----------

# STEP 1 — resolve_location(text) -> dict. Always the same keys:
#   input, status ('resolved'|'ambiguous'|'none'), method, lat, lon, radius_km, label, options, note.
# Order: (a) coordinates, (b) 6-digit PIN in pin_lookup, (c) place name (districts, then offices,
# exact then difflib close match at 0.85), (d) none. All matching is on normalized lowercase names.

INDIA_LAT = (6.0, 37.5)
INDIA_LON = (66.0, 98.0)
PIN_RE = re.compile(r"[1-8]\d{5}")
COORD_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)$")


def _places():
    """place_lookup as a pandas frame, cached once per session."""
    if "places_pdf" not in _CACHE:
        _CACHE["places_pdf"] = (
            spark.table(f"{SCHEMA}.place_lookup")
            .select(
                "place_name", "place_type", "district", "primary_state", "states_set",
                "centroid_lat", "centroid_lon", "radius_km", "n_post_offices", "multi_state")
            .toPandas())
    return _CACHE["places_pdf"]


def _district_geo(name):
    """Robust district center + radius. place_lookup district centroids/radii are unreliable:
    post-office geocodes collide with namesake offices elsewhere in India (e.g. ~16% of
    'darbhanga' office rows sit ~1000 km away), which drags the naive mean centroid up to
    161 km off and the MAX radius to 1203 km. Fix: center = coordinate MEDIAN of the
    district's offices (robust to that pollution); radius = MEDIAN distance from that
    center to the offices (the median is robust to stray coordinates), capped at 25 km."""
    key = f"district_geo::{name}"
    if key not in _CACHE:
        po = (
            spark.table(f"{SCHEMA}.pincode_clean")
            .filter(F.col("district_std") == name)
            .filter(F.col("pin_lat").isNotNull() & F.col("pin_lon").isNotNull())
            .select("pin_lat", "pin_lon")
            .toPandas())
        out = None
        if len(po) >= 5:
            mla, mlo = float(po["pin_lat"].median()), float(po["pin_lon"].median())
            d = _hav_np(
                mla, mlo, po["pin_lat"].to_numpy(), po["pin_lon"].to_numpy())
            out = (mla, mlo, float(min(np.median(d), 25.0)))
        if out is None:
            row = _places()[
                (_places()["place_name"] == name)
                & (_places()["place_type"] == "district")]
            if len(row) and not pd.isna(row.iloc[0]["centroid_lat"]):
                out = (
                    float(row.iloc[0]["centroid_lat"]),
                    float(row.iloc[0]["centroid_lon"]),
                    25.0)
            else:
                out = (25.0, 79.0, 25.0)
        _CACHE[key] = out
    return _CACHE[key]


def _place_radius(name, district, clat, clon):
    """Median distance from the place's centroid to its post offices (robust to stray
    coordinates), capped at 25 km. place_lookup.radius_km is a MAX and is inflated by strays."""
    key = f"place_radius::{name}::{district}"
    if key not in _CACHE:
        po = (
            spark.table(f"{SCHEMA}.pincode_clean")
            .filter((F.col("officename_std") == name) & (F.col("district_std") == district))
            .filter(F.col("pin_lat").isNotNull() & F.col("pin_lon").isNotNull())
            .select("pin_lat", "pin_lon")
            .toPandas())
        if len(po) >= 1:
            d = _hav_np(
                float(clat), float(clon),
                po["pin_lat"].to_numpy(), po["pin_lon"].to_numpy())
            _CACHE[key] = float(min(np.median(d), 25.0))
        else:
            _CACHE[key] = 2.0
    return _CACHE[key]


def _loc_result(text):
    return {
        "input": str(text if text is not None else ""),
        "status": "none",
        "method": None,
        "lat": None,
        "lon": None,
        "radius_km": None,
        "label": None,
        "options": [],
        "note": None,
    }


def _fill_resolved(res, method, lat, lon, radius_km, label, note):
    res.update({
        "status": "resolved", "method": method, "lat": float(lat), "lon": float(lon),
        "radius_km": float(radius_km), "label": label, "note": note})
    return res


def _resolve_place_name(s):
    """Exact-name place resolution against districts first, then offices.
    Returns a result dict, or None if the name is not in place_lookup."""
    pdf = _places()

    # ---- districts (one row per district name) ----
    d = pdf[(pdf["place_type"] == "district") & (pdf["place_name"] == s)]
    if len(d):
        states = sorted({st for v in d["states_set"] for st in _arr(v)})
        if len(states) > 1:
            res = _loc_result(s)
            res["status"], res["method"] = "ambiguous", "district"
            res["options"] = [
                {"name": _title(s), "district": _title(s), "state": _title(st)}
                for st in states]
            res["note"] = (
                f"{_title(s)} exists in more than one state "
                f"({', '.join(_title(st) for st in states)}) — please pick one.")
            return res
        r = d.iloc[0]
        if pd.isna(r["centroid_lat"]):
            return None
        state = _title(states[0] if states else (r["primary_state"] or ""))
        gla, glo, grad = _district_geo(s)
        return _fill_resolved(
            _loc_result(s), "district", gla, glo, grad,
            f"{_title(s)} district, {state}",
            f"Centered on {_title(s)} district, {state}; distances measured from the center "
            f"of the district, ±{grad:.0f} km.")

    # ---- offices (one row per office name + district) ----
    o = pdf[(pdf["place_type"] == "office") & (pdf["place_name"] == s)]
    if len(o):
        states = sorted({str(r["primary_state"]) for _, r in o.iterrows() if r["primary_state"]})
        if len(states) > 1:
            res = _loc_result(s)
            res["status"], res["method"] = "ambiguous", "place"
            seen, opts = set(), []
            for _, r in o.sort_values("n_post_offices", ascending=False).iterrows():
                key = (r["place_name"], r["district"], r["primary_state"])
                if key not in seen:
                    seen.add(key)
                    opts.append({
                        "name": _title(key[0]), "district": _title(key[1]),
                        "state": _title(key[2])})
                if len(opts) >= 8:
                    break
            res["options"] = opts
            res["note"] = (
                f"'{_title(s)}' is a place name in more than one state "
                f"({', '.join(_title(x) for x in states)}) — please pick one.")
            return res
        og = o[o["centroid_lat"].notna()]
        if len(og) == 0:
            return None
        r = og.sort_values(["n_post_offices", "district"], ascending=[False, True]).iloc[0]
        rad = _place_radius(s, str(r["district"]), r["centroid_lat"], r["centroid_lon"])
        extra = ""
        if len(o) > 1:
            extra = (f" {len(o)} places share this name in this state; "
                     f"using the one with the most post offices.")
        return _fill_resolved(
            _loc_result(s), "place", r["centroid_lat"], r["centroid_lon"],
            rad,
            f"{_title(s)}, {_title(r['district'])} district, {_title(r['primary_state'])}",
            f"Matched the place {_title(s)} in {_title(r['district'])} district, "
            f"{_title(r['primary_state'])}; distances measured from {_title(s)}, "
            f"±{rad:.0f} km." + extra)
    return None


def resolve_location(text):
    """STEP 1 — free-text location -> resolved point + honest radius, or a clarification request."""
    s = _normalize(text)
    res = _loc_result(text)

    # (a) coordinates like "25.6, 85.1", within India's bounds
    m = COORD_RE.match(s)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if INDIA_LAT[0] <= lat <= INDIA_LAT[1] and INDIA_LON[0] <= lon <= INDIA_LON[1]:
            return _fill_resolved(
                res, "coordinates", lat, lon, 0.0,
                f"{lat:.4f}, {lon:.4f}",
                "Using the coordinates you provided.")
        res["note"] = f"'{text}' does not look like a location in India."
        return res

    # (b) 6-digit PIN found in pin_lookup
    if PIN_RE.fullmatch(s):
        rows = (
            spark.table(f"{SCHEMA}.pin_lookup")
            .filter(F.col("pincode_str") == s)
            .limit(1)
            .collect())
        if rows:
            r = rows[0]
            districts = _arr(r["districts_set"])
            n_d = int(r["n_districts"] or 1)
            spread = max(float(r["coord_spread_km"] or 0.0), 2.0)
            if n_d > 1 and len(districts) > 1:
                note = (
                    f"PIN {s} spans {n_d} districts "
                    f"({', '.join(_title(x) for x in sorted(districts))}); "
                    f"centered on {_title(r['primary_district'])} district, "
                    f"{_title(r['primary_state'])}; distances measured from the PIN's center, "
                    f"±{spread:.0f} km.")
            else:
                note = (
                    f"Centered on PIN {s} "
                    f"({_title(r['primary_district'])} district, {_title(r['primary_state'])}); "
                    f"distances measured from the PIN's center, ±{spread:.0f} km.")
            return _fill_resolved(
                res, "pin", r["centroid_lat"], r["centroid_lon"],
                spread,
                f"PIN {s} — {_title(r['primary_district'])} district, {_title(r['primary_state'])}",
                note)
        res["note"] = f"PIN {s} is not in the India Post directory."
        return res

    # (c) place name: exact, then difflib close match (cutoff 0.85); districts, then offices
    out = _resolve_place_name(s)
    if out is None:
        pdf = _places()
        if "district_names" not in _CACHE:
            _CACHE["district_names"] = sorted(
                set(pdf[pdf["place_type"] == "district"]["place_name"]))
        close_d = difflib.get_close_matches(s, _CACHE["district_names"], n=1, cutoff=0.85)
        if close_d:
            out = _resolve_place_name(close_d[0])
            if out is not None and out["status"] == "resolved":
                out["note"] = f"Interpreted '{text}' as {out['label']} (close to '{close_d[0]}')."
            elif out is not None and out["status"] == "ambiguous":
                out["note"] = f"Interpreted '{text}' as '{close_d[0]}' — {out['note']}"
    if out is None:
        pdf = _places()
        if "office_names" not in _CACHE:
            _CACHE["office_names"] = sorted(
                set(pdf[pdf["place_type"] == "office"]["place_name"]))
        close_o = difflib.get_close_matches(s, _CACHE["office_names"], n=1, cutoff=0.85)
        if close_o:
            out = _resolve_place_name(close_o[0])
            if out is not None and out["status"] == "resolved":
                out["note"] = f"Interpreted '{text}' as {out['label']} (close to '{close_o[0]}')."
            elif out is not None and out["status"] == "ambiguous":
                out["note"] = f"Interpreted '{text}' as '{close_o[0]}' — {out['note']}"
    if out is not None:
        out["input"] = str(text if text is not None else "")
        return out

    # (d) nothing matched
    res["note"] = f"Could not recognize '{text}' as a place, PIN, or coordinates."
    return res

# COMMAND ----------

# STEP 2 — get_candidates(service_resolution, location): facilities with strong/moderate evidence
# within max_km of the resolved point, with widening (1.5x, 3x, then weak) if fewer than 3 options.
# Matched services read facility_service_evidence (notebook 04); generated services first restrict
# facilities to the radius, then match the mapping's core/supporting terms against facility_items
# (informative/unclear) with notebook-04 tier logic, capped at 'moderate'.

TIER_BASE = {"strong": 0.7, "moderate": 0.45, "weak": 0.15}


def _dist_col(lat, lon):
    """Spark column: haversine km from the user point to final_lat/final_lon."""
    la, lo = math.radians(lat), math.radians(lon)
    fla, flo = F.radians(F.col("final_lat")), F.radians(F.col("final_lon"))
    return (
        F.lit(6371.0) * 2.0 * F.asin(F.sqrt(
            F.pow(F.sin((fla - F.lit(la)) / F.lit(2.0)), 2)
            + F.lit(math.cos(la)) * F.cos(fla) * F.pow(F.sin((flo - F.lit(lo)) / F.lit(2.0)), 2))))


def _base_facilities():
    """gold_facility_search + facilities_located, one row per facility. All trust / readiness
    columns used by STEP 3 come from here (full list in the overview cell)."""
    gold = spark.table(f"{SCHEMA}.gold_facility_search").select(
        "unique_id", "name", "facility_type", "number_doctors", "capacity_beds",
        "phone", "phone_numbers", "website", "specialty_codes")
    loc = spark.table(f"{SCHEMA}.facilities_located").select(
        "unique_id", "final_lat", "final_lon", "location_error_km", "uncertainty_tier",
        "location_evidence", "location_warnings", "possible_merged_record")
    return gold.join(loc, "unique_id", "inner")


_CAND_COLS = [
    "unique_id", "name", "facility_type", "number_doctors", "capacity_beds",
    "phone", "phone_numbers", "website", "specialty_codes",
    "final_lat", "final_lon", "location_error_km", "uncertainty_tier",
    "location_evidence", "location_warnings", "possible_merged_record",
    "service_id", "evidence_tier", "evidence_score", "evidence_snippets",
    "evidence_text", "boilerplate", "contradiction", "distance_km"]


def _matched_candidates(service_id, location):
    """Evidence rows for a checked service (strong/moderate/weak), joined to located facilities."""
    d = _dist_col(location["lat"], location["lon"])
    df = (
        spark.table(f"{SCHEMA}.facility_service_evidence")
        .filter(F.col("service_id") == service_id)
        .filter(F.col("evidence_tier").isin("strong", "moderate", "weak"))
        .select(
            "unique_id", "service_id", "evidence_tier", "evidence_score",
            "evidence_snippets", "evidence_text", "boilerplate", "contradiction")
        .join(_base_facilities(), "unique_id", "inner")
        .filter(F.col("final_lat").isNotNull())
        .withColumn("distance_km", d))
    return df.toPandas()[_CAND_COLS]


def _gen_evidence_text(first_items, specialty_support, tier):
    parts = [f"{_title(f)} lists '{str(it)}'" for (f, it) in first_items]
    txt = ""
    if parts:
        txt = "; ".join(parts) + "."
    if specialty_support:
        txt += " A listed specialty matches this need."
    if tier == "weak":
        txt += " Specialty or related terms only."
    return txt if txt else "Matched on related text."


def _generated_candidates(mapping, location, radius_km):
    """Generated services: restrict facilities to the radius first, then match the mapping's core
    and supporting terms against facility_items (informative/unclear) with notebook-04 tier logic,
    capped at 'moderate'. radius_km=None means no distance cap (used for the honest no_results text)."""
    d = _dist_col(location["lat"], location["lon"])
    fac = (
        _base_facilities()
        .filter(F.col("final_lat").isNotNull())
        .withColumn("distance_km", d))
    if radius_km is not None:
        fac = fac.filter(F.col("distance_km") <= F.lit(float(radius_km)))

    core_terms = list(mapping.get("core_terms") or [])
    supp_terms = list(mapping.get("supporting_terms") or [])
    labels = list(mapping.get("specialty_labels") or [])

    items = (
        spark.table(f"{SCHEMA}.facility_items")
        .filter(F.col("item_class").isin("informative", "unclear"))
        .join(fac.select("unique_id"), "unique_id", "inner"))
    if core_terms:
        core_re = "|".join(re.escape(t) for t in core_terms)
        items = items.withColumn("core_hit", F.col("item_key").rlike(core_re))
    else:
        items = items.withColumn("core_hit", F.lit(False))
    if supp_terms:
        supp_re = "|".join(re.escape(t) for t in supp_terms)
        items = items.withColumn("supp_hit", F.col("item_key").rlike(supp_re))
    else:
        items = items.withColumn("supp_hit", F.lit(False))
    items = items.filter(F.col("core_hit") | F.col("supp_hit"))

    per = items.groupBy("unique_id").agg(
        F.collect_set(F.when(F.col("core_hit"), F.col("field"))).alias("core_fields"),
        F.collect_set(F.when(F.col("supp_hit"), F.col("field"))).alias("supporting_fields"),
        F.slice(
            F.collect_list(F.struct(
                F.col("field").alias("field"), F.col("item").alias("item"))),
            1, 3).alias("first_items"))

    if labels:
        spec_col = (
            F.size(F.array_intersect(
                F.coalesce(F.col("specialty_codes"), F.array().cast("array<string>")),
                F.array(*[F.lit(x) for x in labels])))
            > 0)
    else:
        spec_col = F.lit(False)

    boiler = (
        spark.table(f"{SCHEMA}.service_fields_parsed")
        .select("unique_id", F.col("possible_boilerplate").alias("boilerplate_flag")))

    ev = (
        fac
        .join(per, "unique_id", "left")
        .join(boiler, "unique_id", "left")
        .withColumn("specialty_support", spec_col)
        .filter(
            (F.size(F.coalesce(F.col("core_fields"), F.array())) > 0)
            | (F.size(F.coalesce(F.col("supporting_fields"), F.array())) > 0)
            | F.col("specialty_support")))

    pdf = ev.toPandas()
    rows = []
    for _, r in pdf.iterrows():
        core_fields = sorted(set(_arr(r["core_fields"])))
        supp_fields = sorted(set(_arr(r["supporting_fields"])))
        spec = bool(r["specialty_support"])
        # notebook-04 tier logic, capped at 'moderate' for generated services:
        # a core hit (+ more evidence or specialty support) would be strong; any core hit is
        # moderate; supporting/specialty only is weak.
        tier = "moderate" if core_fields else "weak"
        first_items = [(x["field"], x["item"]) for x in _arr(r["first_items"])]
        rows.append({
            "unique_id": r["unique_id"], "name": r["name"],
            "facility_type": r["facility_type"], "number_doctors": r["number_doctors"],
            "capacity_beds": r["capacity_beds"], "phone": r["phone"],
            "phone_numbers": r["phone_numbers"], "website": r["website"],
            "specialty_codes": r["specialty_codes"],
            "final_lat": r["final_lat"], "final_lon": r["final_lon"],
            "location_error_km": r["location_error_km"], "uncertainty_tier": r["uncertainty_tier"],
            "location_evidence": r["location_evidence"], "location_warnings": r["location_warnings"],
            "possible_merged_record": r["possible_merged_record"],
            "service_id": None,
            "evidence_tier": tier,
            "evidence_score": float(min(
                1.0,
                0.4 * len(core_fields) + 0.2 * len(supp_fields) + (0.4 if spec else 0.0))),
            "evidence_snippets": [f"{f}: '{it}'" for (f, it) in first_items],
            "evidence_text": _gen_evidence_text(first_items, spec, tier),
            "boilerplate": (bool(r["boilerplate_flag"]) if not pd.isna(r["boilerplate_flag"]) else False),
            "contradiction": False,
            "distance_km": float(r["distance_km"]),
        })
    return pd.DataFrame(rows, columns=_CAND_COLS) if rows else pd.DataFrame(columns=_CAND_COLS)


def _widen(cand, base_max):
    """STEP 2 widening: 1.5x max_km, then 3x, then also weak evidence — until 3 options are found
    or all steps are used. Returns (kept_df, radius_km, notes, widened_weak)."""
    tiers_core = ["strong", "moderate"]
    base = cand[
        cand["evidence_tier"].isin(tiers_core)
        & (cand["distance_km"] <= base_max)]
    if len(base) >= 3:
        return base, float(base_max), [], False
    notes = [f"No strong or moderate evidence within {base_max:.0f} km."]
    steps = [
        (1.5, tiers_core, "Widened the search to {r:.0f} km"),
        (3.0, tiers_core, "Widened the search to {r:.0f} km"),
        (3.0, tiers_core + ["weak"],
         "Also included weaker evidence (specialty or related terms only)"),
    ]
    radius, best, best_weak = float(base_max), base, False
    for mult, tiers, tpl in steps:
        radius = float(base_max) * mult
        sub = cand[
            cand["evidence_tier"].isin(tiers)
            & (cand["distance_km"] <= radius)]
        action = tpl.format(r=radius)
        if len(sub) >= 3:
            notes.append(action + ".")
            return sub, radius, notes, "weak" in tiers
        notes.append(f"Still fewer than 3 options within {radius:.0f} km.")
        if len(sub) > len(best):
            best, best_weak = sub, "weak" in tiers
    return best, radius, notes, best_weak


def _nearest_evidence(cand, kind, mapping, location):
    """Nearest strong/moderate/weak evidence overall — for the honest no_results message."""
    tiers = ["strong", "moderate", "weak"]
    wide = cand[cand["evidence_tier"].isin(tiers)] if len(cand) else cand
    if kind == "generated" and len(wide) == 0:
        wide = _generated_candidates(mapping, location, radius_km=None)
        if len(wide):
            wide = wide[wide["evidence_tier"].isin(tiers)]
    if len(wide) == 0:
        return None
    i = wide["distance_km"].idxmin()
    r = wide.loc[i]
    return {
        "unique_id": r["unique_id"], "name": r["name"],
        "distance_km": round(float(r["distance_km"]), 1),
        "evidence_tier": r["evidence_tier"]}


def get_candidates(service_resolution, location):
    """STEP 2 — candidates + widening + honest nearest-evidence. Returns a dict."""
    cat = _catalog()
    if service_resolution["status"] == "matched":
        service_id = service_resolution["service_id"]
        base_max = int(cat[service_id]["max_km"])
        cand = _matched_candidates(service_id, location)
        kind = "matched"
    elif service_resolution["status"] == "generated":
        base_max = 150
        cand = _generated_candidates(service_resolution["mapping"], location, radius_km=base_max)
        kind = "generated"
    else:
        raise ValueError("get_candidates needs a matched or generated service resolution")

    kept, radius, notes, weak_used = _widen(cand, base_max)
    return {
        "candidates": kept,
        "service_kind": kind,
        "base_max_km": float(base_max),
        "search_radius_km": radius,
        "search_notes": notes,
        "widened_weak": weak_used,
        "nearest_evidence": (
            None if len(kept) > 0
            else _nearest_evidence(cand, kind, service_resolution.get("mapping"), location)),
    }


print("get_candidates ready.")

# COMMAND ----------

# STEP 3 — score_candidates(cand, urgency): every component 0-1, plus final_score.
# evidence  = tier base (strong .7 / moderate .45 / weak .15) + 0.3 * evidence_score
# access    = exp(-(distance_km + location_error_km) / decay_km) — the user's location radius
#             is NOT part of the score; it is only shown in the location note ("±X km")
#             decay: emergency 30, recurring 80, planned 200
# readiness = facility_type (hospital 1.0, clinic 0.5, other 0.3); doctors and capacity scaled
#             as min(log1p(x)/log1p(50), 1), 0.3 when missing
# trust     = phone, official website, no merged-record flag, location uncertainty tier
#             (low 1, medium 0.6, high 0.2) — averaged. source/recency items don't exist in the
#             data (see the overview cell) and are excluded.

DECAY_KM = {"emergency": 30.0, "recurring": 80.0, "planned": 200.0}
WEIGHTS = {
    "emergency": {"evidence": 0.35, "access": 0.40, "readiness": 0.15, "trust": 0.10},
    "recurring": {"evidence": 0.40, "access": 0.30, "readiness": 0.15, "trust": 0.15},
    "planned": {"evidence": 0.50, "access": 0.15, "readiness": 0.20, "trust": 0.15},
}
LOG50 = math.log1p(50)


def urgency_for(service_resolution):
    """Urgency from service_catalog; generated services use 'planned'."""
    if service_resolution["status"] == "matched":
        return _catalog()[service_resolution["service_id"]]["urgency"]
    return "planned"


def _has_phone(row):
    p = row.get("phone")
    if isinstance(p, str) and p.strip():
        return True
    v = row.get("phone_numbers")
    return v is not None and len(_arr(v)) > 0


def score_candidates(cand, urgency):
    df = cand.copy()
    decay = DECAY_KM[urgency]

    # evidence component
    df["evidence_component"] = (
        df["evidence_tier"].map(TIER_BASE).astype(float)
        + 0.3 * df["evidence_score"].fillna(0.0).astype(float))

    # access component — user's location radius is shown in the location note, not scored
    err = df["location_error_km"].fillna(0.0).astype(float)
    df["access_component"] = np.exp(
        -(df["distance_km"].astype(float) + err) / decay)

    # readiness component
    ft = df["facility_type"].str.lower().map({"hospital": 1.0, "clinic": 0.5}).fillna(0.3)
    docs = np.where(
        df["number_doctors"].isna(),
        0.3,
        np.minimum(np.log1p(df["number_doctors"].fillna(0.0)) / LOG50, 1.0))
    cap = np.where(
        df["capacity_beds"].isna(),
        0.3,
        np.minimum(np.log1p(df["capacity_beds"].fillna(0.0)) / LOG50, 1.0))
    df["readiness_component"] = (ft.astype(float) + docs + cap) / 3.0

    # trust component
    has_phone = df.apply(_has_phone, axis=1)
    web = df["website"].fillna("").astype(str).str.strip().ne("")
    no_merge = ~df["possible_merged_record"].fillna(False).astype(bool)
    ut = (
        df["uncertainty_tier"].str.lower()
        .map({"low": 1.0, "medium": 0.6, "high": 0.2})
        .fillna(0.6))
    df["trust_component"] = (
        has_phone.astype(float) + web.astype(float)
        + no_merge.astype(float) + ut.astype(float)) / 4.0

    # final weighted score
    w = WEIGHTS[urgency]
    df["final_score"] = (
        w["evidence"] * df["evidence_component"]
        + w["access"] * df["access_component"]
        + w["readiness"] * df["readiness_component"]
        + w["trust"] * df["trust_component"])
    return df


print("score_candidates ready.")

# COMMAND ----------

# STEP 4 — pick_top3(scored): Best overall, Closest credible option, Strongest evidence, then
# fill by final_score (skip a facility already chosen). STEP 5 — fixed-template explanations;
# no free-form LLM text is ever used in an explanation.

COMPONENT_PHRASES = {
    "evidence": "its service evidence",
    "access": "being close to you",
    "readiness": "its readiness (staff and capacity)",
    "trust": "verifiable contact and location details",
}


def pick_top3(scored):
    """Returns a list of {'role', 'row'} picks, at most 3, with the STEP 4 roles.
    Facilities flagged possible_merged_record are a last resort: named roles and the fill
    come from clean records whenever any exist, and flagged records are only used to reach
    3 options (always shown with a warning)."""
    if len(scored) == 0:
        return []
    df = scored.sort_values("final_score", ascending=False).reset_index(drop=True)
    flagged = df["possible_merged_record"].fillna(False).astype(bool)
    clean = df[~flagged]
    pool = clean if len(clean) > 0 else df
    picks, used = [], set()

    def take(row, role):
        uid = row["unique_id"]
        if uid in used:
            return
        used.add(uid)
        picks.append({"role": role, "row": row})

    take(pool.iloc[0], "Best overall")
    cred = pool[pool["evidence_tier"].isin(["strong", "moderate"])]
    if len(cred):
        take(cred.sort_values("distance_km").iloc[0], "Closest credible option")
    take(
        pool.sort_values("evidence_component", ascending=False).iloc[0],
        "Strongest evidence")
    for _, row in pool.iterrows():
        if len(picks) >= 3:
            break
        take(row, "Next best option")
    if len(picks) < 3:   # last resort: possibly-merged records, always warned about
        for _, row in df.iterrows():
            if len(picks) >= 3:
                break
            take(row, "Next best option")
    return picks


def explain_result(pick, service_resolution, weights):
    """STEP 5 — the fixed explanation template for one result."""
    r, role = pick["row"], pick["role"]

    warnings = list(_arr(r.get("location_warnings")))
    if bool(r.get("boilerplate")):
        warnings.append(
            "Some listed services may be boilerplate text repeated across many facilities.")
    if r["evidence_tier"] == "weak":
        warnings.append(
            "Matched on specialty or related terms only — not a direct mention of the service.")
    if service_resolution["status"] == "generated":
        warnings.append(
            f"Best-effort search using AI-generated terms for '{service_resolution['input']}' "
            f"— not a checked service.")
    if bool(r.get("contradiction")):
        warnings.append(
            "Some fields hint this service may NOT be offered here — call before travelling.")
    if bool(r.get("possible_merged_record")):
        warnings.append("This record may merge two facilities; details may be unreliable.")

    contribs = {k: weights[k] * float(r[f"{k}_component"]) for k in COMPONENT_PHRASES}
    top2 = sorted(contribs, key=contribs.get, reverse=True)[:2]
    why_ranked = (
        f"Chosen mainly for {COMPONENT_PHRASES[top2[0]]} and {COMPONENT_PHRASES[top2[1]]}.")

    phone = r.get("phone")
    if not (isinstance(phone, str) and phone.strip()):
        nums = _arr(r.get("phone_numbers"))
        phone = str(nums[0]) if nums else ""
    contact = f"Phone: {phone}. " if phone else "No phone number on record. "
    contact += "Call ahead to confirm the service is available today."

    err = r.get("location_error_km")
    err_txt = "" if pd.isna(err) else f" ±{float(err):.0f} km"

    return {
        "unique_id": r["unique_id"],
        "name": r["name"],
        "role": role,
        "distance_km": round(float(r["distance_km"]), 1),
        "final_score": round(float(r["final_score"]), 3),
        "evidence_tier": r["evidence_tier"],
        "lat": float(r["final_lat"]),
        "lon": float(r["final_lon"]),
        "phone": phone,
        "website": (r.get("website") or None),
        "possible_merged_record": bool(r.get("possible_merged_record")),
        "components": {k: round(float(r[f"{k}_component"]), 3) for k in COMPONENT_PHRASES},
        "headline": f"{r['name']} — {float(r['distance_km']):.0f} km{err_txt} — {role}",
        "why_service": f"{r['evidence_text']} (evidence tier: {r['evidence_tier']})",
        "why_location": r.get("location_evidence"),
        "why_ranked": why_ranked,
        "contact": contact,
        "warnings": warnings,
        "caveat": "All service information is self-reported by the facility.",
    }


def how_we_decided(service_resolution, location, urgency, search_notes, final_radius_km):
    """The whole-query explanation: how the service and location were interpreted, the weights
    used and why (urgency), and any widening."""
    svc = service_resolution["note"]
    if service_resolution["status"] == "generated":
        m = service_resolution.get("mapping") or {}
        svc = (
            f"{svc} Search terms used: "
            f"core {', '.join(m.get('core_terms') or []) or '—'}; "
            f"supporting {', '.join(m.get('supporting_terms') or []) or '—'}; "
            f"specialty labels {', '.join(m.get('specialty_labels') or []) or '—'}.")
    w = WEIGHTS[urgency]
    art = "an" if urgency == "emergency" else "a"
    weights_txt = (
        f"Treated as {art} {urgency} need, so the score weights are "
        f"evidence {w['evidence']}, access {w['access']}, "
        f"readiness {w['readiness']}, trust {w['trust']} — "
        + {
            "emergency": "access matters most because hours matter.",
            "recurring": "evidence and access both matter for regular treatment.",
            "planned": "evidence matters most; distance is less critical."}[urgency])
    return {
        "service": svc,
        "location": location["note"],
        "weights": weights_txt,
        "widening": search_notes,
        "urgency": urgency,
        "final_search_radius_km": float(final_radius_km),
    }


print("pick_top3 + explanations ready.")

# COMMAND ----------

# Google Maps Routes API (Compute Routes): real road distance and driving minutes.
# No heuristic multiplier, no backup routing provider, no hallucinated road duration.
import requests
from urllib.parse import urlencode

GOOGLE_ROUTES_URL = "https://routes.googleapis.com/directions/v2:computeRoutes"
DRIVE_DECAY_MINUTES = {"emergency": 45.0, "recurring": 90.0, "planned": 180.0}


def _google_route(origin_lat, origin_lon, dest_lat, dest_lon, *, http=None):
    key = os.environ.get("GOOGLE_MAPS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("No Google API key configured: paste it into the first code cell")

    coords = [origin_lat, origin_lon, dest_lat, dest_lon]
    if not all(math.isfinite(float(x)) for x in coords):
        raise ValueError("Route origin or destination has invalid coordinates")

    body = {
        "origin": {"location": {"latLng": {
            "latitude": float(origin_lat), "longitude": float(origin_lon)}}},
        "destination": {"location": {"latLng": {
            "latitude": float(dest_lat), "longitude": float(dest_lon)}}},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_UNAWARE",
        "computeAlternativeRoutes": False,
        "units": "METRIC",
        "regionCode": "IN",
    }
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": key,
        "X-Goog-FieldMask": "routes.distanceMeters,routes.duration",
    }
    client = http if http is not None else requests
    try:
        r = client.post(GOOGLE_ROUTES_URL, json=body, headers=headers,
                        timeout=REQUEST_TIMEOUT_SECONDS)
        if r.status_code >= 400:
            # Don't include sensitive request headers in the error message.
            msg = ""
            try:
                error = r.json().get("error", {})
                msg = str(error.get("message", ""))[:300]
            except Exception:
                pass
            raise RuntimeError(f"Google Routes HTTP {r.status_code}: {msg}")
        data = r.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Google Routes connection failed: {type(exc).__name__}: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("Google Routes returned invalid JSON") from exc

    routes = data.get("routes") or []
    if not routes:
        raise RuntimeError("Google returned no drivable route for these coordinates")
    first = routes[0]
    meters = first.get("distanceMeters")
    duration = first.get("duration")
    if meters is None or duration is None:
        raise RuntimeError("Google response omitted road distance or duration")
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", str(duration))
    if not m:
        raise RuntimeError(f"Unexpected Google duration representation: {duration!r}")
    road_km = float(meters) / 1000.0
    mins = float(m.group(1)) / 60.0
    if not (math.isfinite(road_km) and math.isfinite(mins) and road_km >= 0 and mins >= 0):
        raise RuntimeError("Google returned invalid road distance or travel duration")
    return {"road_distance_km": road_km, "drive_minutes": mins,
            "route_provider": "Google Routes API (car, no live traffic)"}


def _maps_directions_link(olat, olon, dlat, dlon):
    return "https://www.google.com/maps/dir/?" + urlencode({
        "api": "1",
        "origin": f"{float(olat):.6f},{float(olon):.6f}",
        "destination": f"{float(dlat):.6f},{float(dlon):.6f}",
        "travelmode": "driving",
    })


def _fatal_route_error(message):
    msg = message.lower()
    return any(token in msg for token in [
        "http 400", "http 401", "http 403", "http 429", "quota", "api key",
        "name resolution", "network is unreachable", "connection failed",
        "blocked", "permission_denied", "requests timeout", "connectionerror",
    ])

print("Google Routes API helper ready.")

# COMMAND ----------

# New public entry point: recommend() now uses Google road routing, NOT Haversine ranking.
# The service-matching and evidence/readiness/trust logic comes from original 06.

def _choose_route_shortlist(candidates, limit):
    """Use Haversine ONLY for a low-cost shortlist, prioritizing service evidence."""
    if candidates.empty:
        return candidates.copy()
    c = candidates.copy()
    c["straight_line_km"] = c["distance_km"].astype(float)
    order = {"strong": 0, "moderate": 1, "weak": 2}
    c["_tier_order"] = c["evidence_tier"].map(order).fillna(3)
    nearest = c.sort_values(["straight_line_km", "_tier_order"])
    evidence = c.sort_values(
        ["_tier_order", "evidence_score", "straight_line_km"],
        ascending=[True, False, True])
    n_near = max(1, limit // 2)
    # First allocate slots to nearby and high-evidence options, then fill from nearest.
    chosen = pd.concat([
        nearest.head(n_near), evidence.head(limit - n_near), nearest
    ]).drop_duplicates(subset=["unique_id"]).head(limit)
    return chosen.drop(columns=["_tier_order"]).copy()


def _pick_top3_by_driving(scored):
    """Same roles as original 06 except 'closest' means smallest Google drive time."""
    if scored.empty:
        return []
    ordered = scored.sort_values("final_score", ascending=False).reset_index(drop=True)
    clean = ordered[~ordered["possible_merged_record"].fillna(False).astype(bool)]
    primary = clean if len(clean) else ordered
    result, used = [], set()

    def take(row, role):
        if row["unique_id"] not in used:
            used.add(row["unique_id"])
            result.append({"role": role, "row": row})

    take(primary.iloc[0], "Best overall")
    credible = primary[primary["evidence_tier"].isin(["strong", "moderate"])]
    if not credible.empty:
        take(credible.sort_values("drive_minutes").iloc[0], "Fastest credible by car")
    take(primary.sort_values("evidence_component", ascending=False).iloc[0], "Strongest evidence")
    for _, row in primary.iterrows():
        if len(result) >= 3:
            break
        take(row, "Next best option")
    for _, row in ordered.iterrows():
        if len(result) >= 3:
            break
        take(row, "Next best option")
    return result[:3]


def recommend(service_text, location_text, *, use_llm=False, max_routes=None):
    """Return evidence-backed recommendations ranked using genuine Google driving minutes.

    Fails closed: if Databricks cannot call Google's API, returns routing_unavailable.
    Does not make fake road-distance estimates or use straight-line rankings as a substitute.
    """
    limit = MAX_GOOGLE_ROUTES if max_routes is None else int(max_routes)
    if not 1 <= limit <= 20:
        raise ValueError("max_routes must be between 1 and 20")
    svc = resolve_service(service_text, use_llm=use_llm)
    loc = resolve_location(location_text)
    base = {
        "service_resolution": svc, "location_resolution": loc, "results": [],
        "route_errors": [], "search_notes": [], "how_we_decided": {},
    }
    if svc["status"] not in ("matched", "generated"):
        base.update(status="needs_service_clarification" if svc["status"] == "ambiguous" else "unknown_service",
                    search_notes=[svc.get("note")])
        return base
    if loc["status"] != "resolved":
        base.update(status="needs_location_clarification" if loc["status"] == "ambiguous" else "unknown_location",
                    search_notes=[loc.get("note")])
        return base

    found = get_candidates(svc, loc)
    if found["candidates"].empty:
        notes = list(found["search_notes"])
        n = found.get("nearest_evidence")
        if n:
            notes.append(f"Nearest facility with any service evidence: {n['name']} — "
                         f"{n['distance_km']:.0f} km straight-line (NOT driving).")
        return dict(base, status="no_results", search_notes=notes)

    chosen = _choose_route_shortlist(found["candidates"], limit)
    routed, errors = [], []
    for _, row in chosen.iterrows():
        try:
            route = _google_route(loc["lat"], loc["lon"], row["final_lat"], row["final_lon"])
            r = row.copy()
            for k, value in route.items():
                r[k] = value
            # Keep an explicit straight-line column. The base score is recomputed below.
            r["distance_km"] = route["road_distance_km"]
            routed.append(r)
        except Exception as exc:
            message = str(exc)
            errors.append({"facility_id": str(row["unique_id"]),
                           "facility": str(row["name"]), "error": message})
            if _fatal_route_error(message):
                break

    if not routed:
        return dict(base, status="routing_unavailable", route_errors=errors,
                    search_notes=[
                        "Google driving routes could not be retrieved. No road distance has been estimated.",
                        "If this is a network error, Databricks Free Edition may block routes.googleapis.com.",
                        "Use a precise patient coordinate for a meaningful route comparison.",
                    ])

    routed_df = pd.DataFrame(routed)
    urgency = urgency_for(svc)
    scored = score_candidates(routed_df, urgency)
    scored["access_component"] = np.exp(
        -scored["drive_minutes"].astype(float) / DRIVE_DECAY_MINUTES[urgency])
    weights = WEIGHTS[urgency]
    scored["final_score"] = sum(
        weights[k] * scored[f"{k}_component"]
        for k in ("evidence", "access", "readiness", "trust"))
    picks = _pick_top3_by_driving(scored)

    # The original explanatory phrasing referred to proximity, so make it explicitly driving time.
    COMPONENT_PHRASES["access"] = "its shorter estimated drive time"
    results = []
    for pick in picks:
        rec = explain_result(pick, svc, weights)
        row = pick["row"]
        drive_km = float(row["road_distance_km"])
        drive_min = float(row["drive_minutes"])
        rec.update({
            "distance_km": round(drive_km, 1),
            "road_distance_km": round(drive_km, 1),
            "straight_line_km": round(float(row["straight_line_km"]), 1),
            "drive_minutes": round(drive_min, 1),
            "route_provider": row["route_provider"],
            "route_origin_lat": float(loc["lat"]),
            "route_origin_lon": float(loc["lon"]),
            "route_destination_lat": float(row["final_lat"]),
            "route_destination_lon": float(row["final_lon"]),
            "maps_directions_url": _maps_directions_link(
                loc["lat"], loc["lon"], row["final_lat"], row["final_lon"]),
            "headline": f"{row['name']} — {drive_km:.0f} km by road, "
                        f"{drive_min:.0f} min by car — {pick['role']}",
            "distance_method": "Google Routes API, car, no live traffic",
        })
        if loc["method"] != "coordinates":
            rec["warnings"].append(
                "Route begins at an approximate PIN/place/district center, not the patient's address.")
        if len(routed) < len(chosen):
            rec["warnings"].append("Some other candidates could not be routed; ranking is partial.")
        results.append(rec)

    notes = list(found["search_notes"])
    notes.append(f"Google road routes calculated for {len(routed)}/{len(chosen)} shortlisted hospitals.")
    notes.append("Candidate selection used straight-line km; final ranking used driving minutes.")
    if len(found["candidates"]) > len(chosen):
        notes.append("This is NOT an exhaustive road-time ranking of every eligible facility.")
    if loc["method"] != "coordinates":
        notes.append(f"Approximate origin used: {loc['lat']:.6f}, {loc['lon']:.6f} "
                     f"(resolved as {loc['method']}).")
    if errors:
        notes.append(f"{len(errors)} route(s) failed; no road distances were invented for them.")

    decision = how_we_decided(svc, loc, urgency, notes, found["search_radius_km"])
    decision["access_method"] = "Google DRIVE duration (minutes), traffic-unaware"
    decision["selection_method"] = "Haversine prefilter, Google Routes shortlist, drive-time re-rank"
    decision["distance_note"] = "Initial search radius is straight-line km, not a driving limit."

    return dict(base, status="ok", results=results, route_errors=errors,
                search_notes=notes, how_we_decided=decision,
                routing_status="complete" if not errors else "partial",
                number_candidates_routed=len(routed),
                number_candidates_available=len(found["candidates"]))

print("recommend() ready: Google driving minutes are now used for final access ranking.")

# COMMAND ----------

# RUN THIS CELL to test the full workflow inside Databricks.
# It uses the same source tables and service evidence as the original 06 notebook.
# It makes up to MAX_GOOGLE_ROUTES Google requests, only if your key is configured.
TEST_SERVICE = "heart attack"
TEST_LOCATION = "Samaspura, Bihar 844126, India"  # Or precise coordinates: "26.1542, 85.8918"

out = recommend(TEST_SERVICE, TEST_LOCATION, use_llm=False)
print("=" * 78)
print(f"INPUT: {TEST_SERVICE!r} near {TEST_LOCATION!r}")
print("STATUS:", out["status"])
loc = out.get("location_resolution") or {}
if loc.get("status") == "resolved":
    print(f"ROUTE ORIGIN: {loc['lat']:.6f}, {loc['lon']:.6f} ({loc['method']})")
    print("NOTE:", loc.get("note"))
print("=" * 78)

for i, result in enumerate(out.get("results") or [], 1):
    print(f"{i}. {result['headline']}")
    print(f"   Straight-line (for comparison): {result['straight_line_km']:.1f} km")
    print("   Why ranked:", result["why_ranked"])
    print("   Evidence:", result["why_service"][:250])
    print("   Directions (exact origin + hospital coords):", result["maps_directions_url"])
    for warning in result.get("warnings") or []:
        print("   Warning:", warning)

for note in out.get("search_notes") or []:
    print("NOTE:", note)
for err in out.get("route_errors") or []:
    print("ROUTING ERROR:", err["facility"], "—", err["error"])

if out.get("results"):
    display(pd.DataFrame([{
        "Role": r["role"], "Facility": r["name"],
        "Road km": r["road_distance_km"], "Driving minutes": r["drive_minutes"],
        "Straight-line km": r["straight_line_km"],
        "Evidence tier": r["evidence_tier"],
        "Google Maps URL": r["maps_directions_url"],
    } for r in out["results"]]))
else:
    print("No road-based recommendations to show; review STATUS and ROUTING ERROR above.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notes for using the result in your app
# MAGIC - Call `recommend("dialysis", "Darbhanga")` as the public entry point, or use a PIN
# MAGIC   or `"26.1542, 85.8918"` for a specific patient starting point.
# MAGIC - The result dictionary includes `results[].road_distance_km`, `drive_minutes`,
# MAGIC   `straight_line_km`, `maps_directions_url`, `route_provider`, and uncertainty warnings.
# MAGIC - **Do not compare** Google's name-based Darbhanga origin to this notebook's district
# MAGIC   centroid; open the provided directions URL to compare identical coordinate pairs.
# MAGIC - Google outputs are traffic-unaware driving estimates, not live traffic or ambulances.
# MAGIC - If status is `routing_unavailable`, the environment/key/quota prevented access;
# MAGIC   no fake road distances are shown.
# MAGIC - Existing processed tables are read-only in this notebook; service mapping creation
# MAGIC   only occurs if `use_llm=True` is explicitly passed for an unrecognized service.
# MAGIC - The original 06 notebook's old `recommend()` used straight-line ranking. This merged
# MAGIC   notebook replaces it; rerun this notebook before calling the function from another notebook.