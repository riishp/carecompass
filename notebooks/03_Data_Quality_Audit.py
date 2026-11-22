# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 03 — Data Quality Audit (all three tables, before cleaning)
# MAGIC
# MAGIC **Sources (read-only):** `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset`
# MAGIC `facilities` · `india_post_pincode_directory` · `nfhs_5_district_health_indicators`
# MAGIC
# MAGIC **This notebook changes nothing.** It only identifies problems so we can agree on cleaning rules
# MAGIC before writing `Data Cleaning: Raw to Bronze`. It builds on `02_Facilities_EDA` (which covered
# MAGIC facilities-only missingness, duplicate `unique_id`, raw state counts, and out-of-India coordinates) and adds:
# MAGIC
# MAGIC | § | Question | What's new vs. 02 |
# MAGIC |---|---|---|
# MAGIC | 1 | **What is missing?** | All 3 tables; splits SQL NULL vs `"null"` strings vs empty lists `[]`; blanks *inside* lists |
# MAGIC | 2 | **What repeats?** | Exact duplicate rows, duplicates by name+city / phone / website / email / identical coordinates, repeated items inside lists, pincode & district duplicates |
# MAGIC | 3 | **What contradicts itself?** | Coordinates vs. the facility's own pincode location (km), lat/lon vs `coordinates` JSON, swapped lat/lon, state & city vs. pincode, public/private vs. name, impossible numbers, shifted rows |
# MAGIC | 4 | **What needs adjusting?** | State/city/district spelling, capitalization & renamed-city variants across tables, whitespace, pincode & phone formats, category vocabularies, numbers stored as text |
# MAGIC | 5 | **Scorecard** | One table of every headline number |
# MAGIC
# MAGIC Each section ends with a **Findings** cell. Run top-to-bottom on serverless compute (~2–3 min).

# COMMAND ----------

# DBTITLE 1,Setup: load tables (read-only) and helpers
import re, json, difflib
import pandas as pd
from pyspark.sql import functions as F

_display = display
def display(x, *args, **kwargs):
    """display() that prints '(no rows)' instead of erroring on an empty pandas table."""
    if isinstance(x, pd.DataFrame) and len(x) == 0:
        print("(no rows — nothing found for this check)")
        return
    return _display(x, *args, **kwargs)

CATALOG = "databricks_virtue_foundation_dataset_dais_2026"
SCHEMA = "virtue_foundation_dataset"
TABLES = {
    "fac": "facilities",
    "pin": "india_post_pincode_directory",
    "nfhs": "nfhs_5_district_health_indicators",
}

# Tokens that mean "missing" in this dataset (same list as 02_Facilities_EDA)
MISSING_TOKENS = ["", "null", "nan", "none", "n/a", "na", "-"]
EMPTY_CONTAINERS = ["[]", "{}", '[""]', "[null]"]

def nullify(df):
    """Return a copy where missing-tokens in string columns become real NULLs (analysis only — nothing is written)."""
    return df.select([
        (F.when(F.lower(F.trim(F.col(c))).isin(MISSING_TOKENS + EMPTY_CONTAINERS), None).otherwise(F.col(c)).alias(c)
         if t == "string" else F.col(c))
        for c, t in df.dtypes
    ])

for short, name in TABLES.items():
    raw = spark.table(f"`{CATALOG}`.`{SCHEMA}`.`{name}`")
    raw.createOrReplaceTempView(short)              # raw, exactly as shared
    nullify(raw).createOrReplaceTempView(short + "n")  # same rows, missing-tokens -> NULL

q = spark.sql
def scalar(sql):
    return q(sql).collect()[0][0]

SUMMARY = []  # (section, check, value) rows for the scorecard in §5
def record(section, check, value):
    SUMMARY.append({"section": section, "check": check, "value": value})
    return value

# Reusable SQL snippets
PIN6 = "case when length(regexp_replace(address_zipOrPostcode,'[^0-9]','')) = 6 then cast(regexp_replace(address_zipOrPostcode,'[^0-9]','') as bigint) end"
def haversine_km(lat1, lon1, lat2, lon2):
    return (f"6371 * 2 * asin(sqrt(pow(sin(radians({lat2}-{lat1})/2),2) + "
            f"cos(radians({lat1}))*cos(radians({lat2}))*pow(sin(radians({lon2}-{lon1})/2),2)))")

print({k: scalar(f"select count(*) from {k}") for k in TABLES})

# COMMAND ----------

# DBTITLE 1,Setup: reference lists (canonical states, renamed cities)
# MAGIC %md
# MAGIC ### Reference lists used for comparison
# MAGIC - **36 canonical states/UTs** (current official names). Every raw state value in all three tables is matched to this list and classified:
# MAGIC   `exact` · `case/format variant` (same letters, different caps/spaces/`&`) · `old name / abbreviation` · `probable misspelling` · `city/district in state field` · `JSON / junk` · `unrecognized`.
# MAGIC - **Renamed cities** (e.g. Gurgaon→Gurugram, Bangalore→Bengaluru) so old and new names can be detected as the same place.

# COMMAND ----------

# DBTITLE 1,Build state map
CANONICAL_STATES = [
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
    "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
]

def skey(s):
    """Letters-only key: 'Tamil Nadu', 'TAMIL NADU', 'Tamilnadu' and 'tamil  nadu ' all -> 'tamilnadu'."""
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z]", "", s)
    return re.sub(r"^the", "", s)

CANON = {skey(s): s for s in CANONICAL_STATES}
STATE_ALIASES = {
    "orissa": "Odisha", "pondicherry": "Puducherry", "pondy": "Puducherry", "uttaranchal": "Uttarakhand",
    "newdelhi": "Delhi", "nctofdelhi": "Delhi", "delhinct": "Delhi", "nationalcapitalterritoryofdelhi": "Delhi",
    "up": "Uttar Pradesh", "mp": "Madhya Pradesh", "hp": "Himachal Pradesh", "ap": "Andhra Pradesh",
    "wb": "West Bengal", "tn": "Tamil Nadu", "jk": "Jammu and Kashmir", "jandk": "Jammu and Kashmir",
    "dadraandnagarhaveli": "Dadra and Nagar Haveli and Daman and Diu",
    "damananddiu": "Dadra and Nagar Haveli and Daman and Diu",
    "andamanandnicobar": "Andaman and Nicobar Islands", "andamannicobar": "Andaman and Nicobar Islands",
}

CITY_RENAMES = {  # old/alternate -> current
    "gurgaon": "gurugram", "bangalore": "bengaluru", "bombay": "mumbai", "calcutta": "kolkata",
    "madras": "chennai", "poona": "pune", "mysore": "mysuru", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "baroda": "vadodara", "allahabad": "prayagraj", "benares": "varanasi",
    "banaras": "varanasi", "pondicherry": "puducherry", "vizag": "visakhapatnam", "calicut": "kozhikode",
    "mangalore": "mangaluru", "belgaum": "belagavi", "hubli": "hubballi", "gulbarga": "kalaburagi",
    "trichy": "tiruchirappalli", "tiruchirapalli": "tiruchirappalli", "shimoga": "shivamogga",
    "tumkur": "tumakuru", "bellary": "ballari",
    "osmanabad": "dharashiv", "faizabad": "ayodhya", "mughalsarai": "pandeendayalupadhyayanagar",
}
def ckey(s):
    return re.sub(r"[^a-z]", "", str(s).lower()) if s is not None else ""
def city_canon(k):
    return CITY_RENAMES.get(k, k)

# Known place names (to detect a city/district typed into the state field)
place_keys = set(ckey(r[0]) for r in q("select distinct district from pinn where district is not null").collect())
place_keys |= set(ckey(r[0]) for r in q("select distinct address_city from facn where address_city is not null").collect())
place_keys -= {""}

def classify_state(raw):
    if raw is None or not str(raw).strip():
        return None, "missing"
    s = str(raw).strip()
    if s.startswith(("{", "[")):
        return None, "JSON / junk"
    k = skey(s)
    if not k:
        return None, "JSON / junk"
    if k in CANON:
        return CANON[k], ("exact" if str(raw) == CANON[k] else "case/format variant")
    if k in STATE_ALIASES:
        return STATE_ALIASES[k], "old name / abbreviation"
    m = difflib.get_close_matches(k, CANON.keys(), n=1, cutoff=0.85)
    if m:
        return CANON[m[0]], "probable misspelling"
    if k in place_keys:
        return None, "city/district in state field"
    return None, "unrecognized"

state_rows = []
for src, sql in [
    ("facilities", "select address_stateOrRegion v, count(*) n from fac group by 1"),
    ("pincode_dir", "select statename v, count(*) n from pin group by 1"),
    ("nfhs", "select state_ut v, count(*) n from nfhs group by 1"),
]:
    for r in q(sql).collect():
        canon, issue = classify_state(r["v"])
        state_rows.append({"source": src, "raw": r["v"], "rows": r["n"], "canonical": canon, "issue": issue})
state_map = pd.DataFrame(state_rows)
spark.createDataFrame(state_map.astype({"rows": "int64"})).createOrReplaceTempView("state_map")
print("state_map built:", len(state_map), "distinct raw state values across 3 tables")

# COMMAND ----------

# DBTITLE 1,§1 Missing values
# MAGIC %md
# MAGIC ## §1 — What information is missing?
# MAGIC Missing data hides in three encodings, counted separately: real SQL `NULL`, null-like strings (`"null"`, `"NA"`, `""`…), and empty lists (`[]`).

# COMMAND ----------

# DBTITLE 1,1a Missing by column, all tables
tok = ",".join(f"'{t}'" for t in MISSING_TOKENS)
emp = ",".join(f"'{t}'" for t in EMPTY_CONTAINERS)
rows = []
for short, name in TABLES.items():
    cols = spark.table(short).columns
    exprs = []
    for i, c in enumerate(cols):
        s = f"lower(trim(cast(`{c}` as string)))"
        exprs += [f"count_if(`{c}` is null) as c{i}_a", f"count_if({s} in ({tok})) as c{i}_b", f"count_if({s} in ({emp})) as c{i}_c"]
    r = q(f"select count(*) as n, {', '.join(exprs)} from {short}").collect()[0].asDict()
    for i, c in enumerate(cols):
        a, b, e = r[f"c{i}_a"], r[f"c{i}_b"], r[f"c{i}_c"]
        rows.append({"table": name, "column": c, "rows": r["n"], "sql_null": a, "null_like_string": b,
                     "empty_list": e, "total_missing": a + b + e, "pct_missing": round(100 * (a + b + e) / r["n"], 2)})
missing = pd.DataFrame(rows).sort_values(["table", "pct_missing"], ascending=[True, False])
display(missing[missing.total_missing > 0])

by_table = missing.groupby("table").apply(lambda d: pd.Series({
    "columns": len(d),
    "cols_with_any_missing": int((d.total_missing > 0).sum()),
    "cols_over_50pct_missing": int((d.pct_missing > 50).sum()),
    "cells_missing_pct": round(100 * d.total_missing.sum() / (d.rows.iloc[0] * len(d)), 1),
    "missing_as_string_not_null_pct": round(100 * (d.null_like_string.sum() + d.empty_list.sum()) / max(d.total_missing.sum(), 1), 1),
}))
display(by_table.reset_index())
for t, r in by_table.iterrows():
    record("1 Missing", f"{t}: % of all cells missing", r.cells_missing_pct)
    record("1 Missing", f"{t}: columns >50% missing", int(r.cols_over_50pct_missing))

# COMMAND ----------

# DBTITLE 1,1b Core fields needed by the app
core = q(f"""
select count(*) as facilities,
  count_if(name is null)                                  as no_name,
  count_if(address_city is null)                          as no_city,
  count_if(address_stateOrRegion is null)                 as no_state,
  count_if({PIN6} is null)                                as no_valid_pincode,
  count_if(latitude is null or longitude is null)         as no_coordinates,
  count_if(officialPhone is null and phone_numbers is null) as no_phone_at_all,
  count_if(specialties is null)                           as no_specialties,
  count_if(specialties is null and procedure is null and equipment is null and capability is null) as no_service_info_at_all,
  count_if(name is null and address_line1 is null and address_city is null) as unidentifiable_rows
from facn""").toPandas()
display(core)
for c in core.columns[1:]:
    record("1 Missing", f"facilities: {c}", int(core[c][0]))

# COMMAND ----------

# DBTITLE 1,1c Blank items hidden inside lists
LIST_COLS = ["specialties", "procedure", "equipment", "capability", "phone_numbers", "websites", "source_types", "source_urls"]
rows = []
for c in LIST_COLS:
    r = q(f"""
      with x as (select from_json(`{c}`, 'array<string>') a from fac)
      select count(*) as parsed_as_list,
             count_if(size(a) = 0) as empty_list,
             count_if(exists(a, e -> e is null or trim(e) = '' or lower(trim(e)) in ('null','none','n/a'))) as rows_with_blank_items,
             sum(size(filter(a, e -> e is null or trim(e) = '' or lower(trim(e)) in ('null','none','n/a')))) as blank_items,
             sum(size(a)) as total_items
      from x where a is not null""").collect()[0].asDict()
    rows.append({"column": c, **r})
list_blanks = pd.DataFrame(rows)
display(list_blanks)
record("1 Missing", "facilities: rows with blank items inside lists (any list col)",
       int(scalar("select count(*) from fac where " + " or ".join(
           f"exists(from_json(`{c}`,'array<string>'), e -> e is null or trim(e) = '')" for c in LIST_COLS))))

# COMMAND ----------

# DBTITLE 1,§1 Findings
# MAGIC %md
# MAGIC ### §1 Findings — missing information
# MAGIC - **21% of all cells in `facilities` are missing**, and **12 of 51 columns are >50% empty** (`countries`, `acceptsVolunteers`, `area`, `capacity`, `numberDoctors`, `yearEstablished`, social-media metrics…). The pincode directory (1.4%) and NFHS (~0%) are nearly complete.
# MAGIC - Most gaps are the **text `"null"` or `[]`, not real NULLs**. Anything that only checks `IS NULL` will think the data is complete.
# MAGIC - **Fields the app needs are mostly present:** of 10,088 facilities, 65 have no name, 64 no city, 63 no state, **159 no usable pincode**, **118 no coordinates**, **339 no phone at all**, 115 no service info at all (no specialties, procedures, equipment or capabilities).
# MAGIC - **57 rows have no name, street or city.** They can't be identified and should be dropped.
# MAGIC - **Blank items hide inside lists:** 9,716 rows have a `""` inside some list. That's almost all `source_urls` (8,770, harmless), but also `equipment` (2,322 rows), `procedure` (2,319) and `capability` (716). The app would show these as empty bullets.

# COMMAND ----------

# DBTITLE 1,§2 Repeats
# MAGIC %md
# MAGIC ## §2 — What repeats?
# MAGIC Four kinds: identical rows, the same facility under different rows, shared contact details, and repeated items inside one list.

# COMMAND ----------

# DBTITLE 1,2a Exact duplicate rows (every column identical)
rows = []
for short, name in TABLES.items():
    total = scalar(f"select count(*) from {short}")
    distinct = scalar(f"select count(*) from (select distinct * from {short})")
    rows.append({"table": name, "rows": total, "distinct_rows": distinct, "exact_duplicates": total - distinct})
    record("2 Repeats", f"{name}: exact duplicate rows", total - distinct)
display(pd.DataFrame(rows))

# COMMAND ----------

# DBTITLE 1,2b Same facility / shared details under different rows
NORM = lambda c: f"lower(regexp_replace({c}, '[^A-Za-z0-9]', ''))"
DUP_KEYS = {
    "unique_id":                       ("unique_id", "unique_id is not null"),
    "cluster_id":                      ("cluster_id", "cluster_id is not null"),
    "name + city (normalized)":        (f"concat({NORM('name')}, '|', {NORM('address_city')})", "name is not null and address_city is not null"),
    "name + address_line1 (normalized)": (f"concat({NORM('name')}, '|', {NORM('address_line1')})", "name is not null and address_line1 is not null"),
    "name only (normalized)":          (NORM("name"), "name is not null"),
    "officialPhone (digits)":          ("regexp_replace(officialPhone, '[^0-9]', '')", "officialPhone is not null"),
    "officialWebsite (domain)":        ("lower(regexp_extract(officialWebsite, '^(?:https?://)?(?:www\\\\.)?([^/?#]+)', 1))", "officialWebsite is not null"),
    "email":                           ("lower(trim(email))", "email is not null"),
    "exact same lat/lon (5 dp)":       ("concat(round(latitude,5), ',', round(longitude,5))", "latitude is not null and longitude is not null"),
}
rows = []
for label, (expr, where) in DUP_KEYS.items():
    r = q(f"""select count(*) as groups, coalesce(sum(n),0) as rows_involved, coalesce(max(n),0) as largest_group
              from (select {expr} k, count(*) n from facn where {where} group by 1 having count(*) > 1)""").collect()[0]
    rows.append({"duplicate key": label, "groups with >1 row": r.groups, "rows involved": r.rows_involved, "largest group": r.largest_group})
    record("2 Repeats", f"facilities: rows sharing {label}", int(r.rows_involved))
display(pd.DataFrame(rows))

# COMMAND ----------

# DBTITLE 1,2b examples: same name + city
display(q(f"""
  with k as (select *, concat({NORM('name')}, '|', {NORM('address_city')}) key from facn where name is not null and address_city is not null)
  select key, count(*) n, collect_list(name) names, collect_list(unique_id) ids, collect_set(address_zipOrPostcode) pincodes,
         collect_set(officialPhone) phones, collect_set(round(latitude,3)) lats
  from k group by key having count(*) > 1 order by n desc limit 30"""))

# COMMAND ----------

# DBTITLE 1,2b examples: one phone / location shared by many facilities
display(q("""
  select regexp_replace(officialPhone,'[^0-9]','') phone, count(*) n, slice(collect_set(name),1,6) sample_names, collect_set(address_city) cities
  from facn where officialPhone is not null group by 1 having count(*) > 2 order by n desc limit 20"""))
display(q("""
  select round(latitude,5) lat, round(longitude,5) lon, count(*) n, slice(collect_set(name),1,6) sample_names, collect_set(address_city) cities
  from facn where latitude is not null group by 1,2 having count(*) > 2 order by n desc limit 20"""))

# COMMAND ----------

# DBTITLE 1,2c Repeated items inside a single list
rows = []
for c in LIST_COLS:
    r = q(f"""
      with x as (select from_json(`{c}`, 'array<string>') a from fac)
      select count(*) as lists,
             count_if(size(a) > size(array_distinct(a))) as lists_with_repeats,
             sum(size(a)) as items, sum(size(array_distinct(a))) as distinct_items,
             count_if(size(array_distinct(transform(a, e -> lower(trim(e))))) < size(array_distinct(a))) as lists_with_case_only_repeats
      from x where a is not null and size(a) > 0""").collect()[0].asDict()
    r["pct_items_redundant"] = round(100 * (r["items"] - r["distinct_items"]) / max(r["items"], 1), 1)
    rows.append({"column": c, **r})
list_repeats = pd.DataFrame(rows)
display(list_repeats)
for _, r in list_repeats.iterrows():
    record("2 Repeats", f"facilities.{r.column}: lists with repeated items", int(r.lists_with_repeats))

# Worst offenders
display(q("""
  select name, size(from_json(specialties,'array<string>')) n_items, size(array_distinct(from_json(specialties,'array<string>'))) n_distinct,
         array_distinct(from_json(specialties,'array<string>')) distinct_specialties
  from fac where from_json(specialties,'array<string>') is not null order by n_items - n_distinct desc limit 10"""))

# COMMAND ----------

# DBTITLE 1,2d Repeats in pincode directory and NFHS
pin_dups = q("""
  select count(*) pincodes,
         count_if(n_states > 1)    as pincodes_in_2plus_states,
         count_if(n_districts > 1) as pincodes_in_2plus_districts,
         count_if(n_offices > 1)   as pincodes_with_2plus_offices_expected
  from (select pincode, count(distinct statename) n_states, count(distinct district) n_districts, count(*) n_offices from pinn group by pincode)""").toPandas()
display(pin_dups)
record("2 Repeats", "pincode_dir: pincodes listed under 2+ states", int(pin_dups.pincodes_in_2plus_states[0]))
record("2 Repeats", "pincode_dir: pincodes listed under 2+ districts", int(pin_dups.pincodes_in_2plus_districts[0]))
display(q("""
  select pincode, collect_set(statename) states, collect_set(district) districts, count(*) offices
  from pinn group by pincode having count(distinct statename) > 1 or count(distinct district) > 1 order by offices desc limit 20"""))

nfhs_dups = q("""
  select lower(trim(state_ut)) state, lower(regexp_replace(district_name,'[^A-Za-z]','')) district_key,
         count(*) n, collect_list(district_name) raw_names
  from nfhsn group by 1,2 having count(*) > 1""")
display(nfhs_dups)
record("2 Repeats", "nfhs: district rows duplicated within a state", int(nfhs_dups.count()))
display(q("""
  select trim(district_name) district, count(distinct state_ut) n_states, collect_set(state_ut) states
  from nfhsn group by 1 having count(distinct state_ut) > 1 order by 2 desc"""))

# COMMAND ----------

# DBTITLE 1,§2 Findings
# MAGIC %md
# MAGIC ### §2 Findings — repeats
# MAGIC - **11 rows are exact copies** of another row (every column identical), the same 11 behind the duplicate `unique_id`s in 02. The pincode directory has 2 exact duplicate rows; NFHS has none.
# MAGIC - **Same facility, slightly different rows:** 38 rows share a normalized name and city, and 24 share name and street (examples in 2b). These need merging, not just dropping, since each copy may hold different phone numbers or coordinates.
# MAGIC - **Shared phone numbers are mostly scraped junk, not chains.** `+919693969347` is the "official" phone for **87 unrelated facilities** in different cities, and a few other numbers appear 10–34 times. 471 facilities share an official phone, 752 a website domain and 612 an email. Each needs checking: Apollo, Fortis and similar chains are legitimate; directory/aggregator numbers are not.
# MAGIC - **Same name, many rows (884)** is mostly real chains (e.g. *Apollo Hospitals* in many cities). That's expected and not a problem.
# MAGIC - **Lists repeat items heavily:** 9,848 of ~9,970 specialty lists repeat at least one item (e.g. `internalMedicine` ×20). Phone lists (9,016) and website lists (6,374) do too. Deduplicate items before counting or displaying.
# MAGIC - **The pincode directory isn't one row per pincode.** Several post offices share a pincode, which is expected. But **52 pincodes are listed under 2+ states and 1,258 under 2+ districts**, so a pincode → district lookup must pick one deliberately (e.g. most common).
# MAGIC - **NFHS:** no duplicate districts within a state. 8 district names exist in two states (*Aurangabad*, *Bilaspur*, *Bijapur*, *Hamirpur*, *Pratapgarh*, *Raigarh*, *Balrampur*), so always join on **state + district**, never district alone. One looks like an error: **Chandel is listed under both Manipur and Mizoram**, but it's only in Manipur.

# COMMAND ----------

# DBTITLE 1,§3 Conflicts
# MAGIC %md
# MAGIC ## §3 — What information contradicts itself?
# MAGIC The most important check for a location app: **does each facility's latitude/longitude agree with its own pincode, state and city?**
# MAGIC Pincode locations come from `india_post_pincode_directory` (median of each pincode's post offices that have valid coordinates).

# COMMAND ----------

# DBTITLE 1,3a Build pincode lookup and facility geo-check view
q("""
create or replace temp view pin_lookup as
select pincode,
       collect_set(statename) states, collect_set(district) districts,
       first(statename, true) state, first(district, true) district,
       -- median (not average) of the pincode's post offices: the directory has bad office coordinates (see 3i)
       percentile_approx(case when try_cast(latitude as double) between 6 and 37 and try_cast(longitude as double) between 68 and 98 then try_cast(latitude as double) end, 0.5)  lat,
       percentile_approx(case when try_cast(latitude as double) between 6 and 37 and try_cast(longitude as double) between 68 and 98 then try_cast(longitude as double) end, 0.5) lon
from pinn group by pincode""")

q(f"""
create or replace temp view fac_geo as
select f.unique_id, f.name, f.address_city, f.address_stateOrRegion, f.address_zipOrPostcode,
       {PIN6} as pin6, f.latitude, f.longitude,
       sm.canonical as state_canon, sm.issue as state_issue,
       p.pincode is not null as pin_found, p.state as pin_state_raw, p.district as pin_district, p.lat as pin_lat, p.lon as pin_lon,
       psm.canonical as pin_state_canon,
       case when f.latitude is not null and p.lat is not null then {haversine_km('f.latitude','f.longitude','p.lat','p.lon')} end as km_from_pincode
from facn f
left join pin_lookup p on {PIN6.replace('address_zipOrPostcode','f.address_zipOrPostcode')} = p.pincode
left join state_map sm  on sm.source = 'facilities'  and sm.raw <=> f.address_stateOrRegion
left join state_map psm on psm.source = 'pincode_dir' and psm.raw <=> p.state
""")
print("fac_geo rows:", scalar("select count(*) from fac_geo"))

# COMMAND ----------

# DBTITLE 1,3b Coordinate sanity: missing, swapped, outside India, JSON disagrees
geo = q("""
with x as (
  select f.*, from_json(f.coordinates, 'struct<coordinates:array<double>,type:string>') g from facn f)
select
  count_if(latitude is null or longitude is null)                                        as missing,
  count_if(abs(latitude) < 0.01 and abs(longitude) < 0.01)                               as zero_zero,
  count_if(latitude between 68 and 98 and longitude between 6 and 37)                    as looks_swapped,
  count_if(latitude is not null and not (latitude between 6 and 37 and longitude between 68 and 98)) as outside_india_box,
  count_if(g is not null and latitude is not null and
           (abs(try_element_at(g.coordinates, 2) - latitude) > 1e-4 or abs(try_element_at(g.coordinates, 1) - longitude) > 1e-4)) as json_vs_columns_disagree,
  count_if(g is null and coordinates is not null)                                        as coordinates_json_unparseable
from x""").toPandas()
display(geo)
for c in geo.columns:
    record("3 Conflicts", f"facilities coordinates: {c}", int(geo[c][0]))

# COMMAND ----------

# DBTITLE 1,3c Coordinates vs. the facility's own pincode location
dist = q("""
select case when pin6 is null then '0 no valid pincode'
            when not pin_found then '1 pincode not in directory'
            when km_from_pincode is null then '2 no coords to compare'
            when km_from_pincode < 10 then '3 < 10 km (agrees)'
            when km_from_pincode < 50 then '4 10–50 km (plausible)'
            when km_from_pincode < 200 then '5 50–200 km (suspicious)'
            else '6 > 200 km (contradiction)' end as bucket,
       count(*) facilities
from fac_geo group by 1 order by 1""").toPandas()
display(dist)
record("3 Conflicts", "facilities > 200 km from own pincode", int(dist.loc[dist.bucket.str.startswith('6'), 'facilities'].sum()))
record("3 Conflicts", "facilities 50–200 km from own pincode", int(dist.loc[dist.bucket.str.startswith('5'), 'facilities'].sum()))
record("3 Conflicts", "facility pincode not found in directory", int(dist.loc[dist.bucket.str.startswith('1'), 'facilities'].sum()))

display(q("""
select name, address_city, address_stateOrRegion, address_zipOrPostcode, pin_district, pin_state_raw,
       round(latitude,4) lat, round(longitude,4) lon, round(km_from_pincode) km_from_pincode
from fac_geo where km_from_pincode > 200 order by km_from_pincode desc limit 40"""))

# COMMAND ----------

# DBTITLE 1,3d State vs. pincode state
st = q("""
select case when state_canon is null and pin_state_canon is null then 'cannot compare (both unknown)'
            when state_canon is null then 'facility state unrecognized'
            when pin_state_canon is null then 'pincode unknown / no state'
            when state_canon = pin_state_canon then 'agree'
            else 'CONFLICT' end as result, count(*) facilities
from fac_geo group by 1 order by 2 desc""").toPandas()
display(st)
record("3 Conflicts", "facility state contradicts pincode state", int(st.loc[st.result == 'CONFLICT', 'facilities'].sum()))
display(q("""
select state_canon facility_state, pin_state_canon pincode_state, count(*) n,
       slice(collect_list(concat(name, ' (', address_city, ', ', address_zipOrPostcode, ')')), 1, 4) examples
from fac_geo where state_canon is not null and pin_state_canon is not null and state_canon != pin_state_canon
group by 1,2 order by n desc limit 25"""))

# COMMAND ----------

# DBTITLE 1,3e City vs. pincode area (fuzzy)
# A city "agrees" with its pincode if it resembles the pincode's district, division, region or any post-office name there.
pin_names = q("""
  select pincode, collect_set(lower(regexp_replace(n, '(?i)\\\\b(b\\\\.?o|s\\\\.?o|h\\\\.?o|division|region|circle)\\\\b|[^A-Za-z]', ''))) names
  from (select pincode, explode(array(officename, district, divisionname, regionname)) n from pinn) where n is not null group by pincode
""").toPandas()
pin_names = dict(zip(pin_names.pincode, pin_names.names))
fac_city = q("select unique_id, name, address_city, address_stateOrRegion, address_zipOrPostcode, pin6, pin_district, km_from_pincode from fac_geo").toPandas()

def city_vs_pin(row):
    if not row.address_city:
        return "no city"
    if pd.isna(row.pin6) or int(row.pin6) not in pin_names:
        return "pincode unknown"
    c = city_canon(ckey(row.address_city))
    if len(c) < 3:
        return "city too short to judge"
    for n in pin_names[int(row.pin6)]:
        n = city_canon(n)
        if not n:
            continue
        if c in n or (len(n) >= 4 and n in c) or difflib.SequenceMatcher(None, c, n).ratio() >= 0.8:
            return "agree"
    return "CONFLICT"

fac_city["city_check"] = fac_city.apply(city_vs_pin, axis=1)
display(fac_city.city_check.value_counts().rename_axis("result").reset_index(name="facilities"))
city_conf = fac_city[fac_city.city_check == "CONFLICT"]
record("3 Conflicts", "facility city does not match its pincode area", len(city_conf))
# Strongest evidence: city mismatch AND coordinates far from pincode -> pincode is probably the wrong field
display(city_conf.sort_values("km_from_pincode", ascending=False).head(40))

# COMMAND ----------

# DBTITLE 1,3f Which location field is wrong? Coordinates vs pincode vs city (2-out-of-3 vote)
# Three independent location signals per facility:
#   F = the facility's own lat/lon
#   P = its pincode's centre (median of post offices)
#   C = its city's centre (median lat/lon of all facilities in that city; cities with >=5 facilities only)
# If two agree (<50 km) and the third is far, the third is the likely error.
q("""
create or replace temp view city_centre as
select lower(regexp_replace(address_city,'[^A-Za-z]','')) ck,
       percentile_approx(latitude, 0.5) lat, percentile_approx(longitude, 0.5) lon, count(*) n
from facn where address_city is not null and latitude between 6 and 37 and longitude between 68 and 98
group by 1 having count(*) >= 5""")
q(f"""
create or replace temp view fac_location_verdict as
with x as (
  select g.*, c.lat c_lat, c.lon c_lon,
         g.km_from_pincode as d_fp,
         {haversine_km('g.latitude','g.longitude','c.lat','c.lon')} as d_fc,
         {haversine_km('g.pin_lat','g.pin_lon','c.lat','c.lon')} as d_pc
  from fac_geo g join city_centre c on lower(regexp_replace(g.address_city,'[^A-Za-z]','')) = c.ck
  where g.latitude between 6 and 37 and g.longitude between 68 and 98 and g.pin_lat is not null)
select *, case when d_fp < 50 and d_fc < 50                    then '1 all three agree'
               when d_fc < 50 and d_fp >= 50 and d_pc >= 50   then '2 pincode is the odd one out'
               when d_fp < 50 and d_fc >= 50 and d_pc >= 50   then '3 city label is the odd one out'
               when d_pc < 50 and d_fp >= 50 and d_fc >= 50   then '4 coordinates are the odd one out'
               when d_fp >= 50 and d_fc >= 50 and d_pc >= 50  then '5 all three disagree'
               else '6 borderline' end as verdict
from x""")
tri = q("select verdict, count(*) facilities from fac_location_verdict group by 1 order by 1").toPandas()
display(tri)
print("Facilities that could be checked (valid coords + known pincode + city with >=5 facilities):", int(tri.facilities.sum()))
for _, r in tri.iterrows():
    record("3 Conflicts", f"location vote: {r.verdict[2:]}", int(r.facilities))
display(q("""select verdict, name, address_city, address_stateOrRegion, address_zipOrPostcode, pin_district, pin_state_raw,
                    round(latitude,3) lat, round(longitude,3) lon, round(d_fp) km_coords_to_pincode, round(d_fc) km_coords_to_city
             from fac_location_verdict where verdict between '2' and '5z' order by verdict, d_fp desc"""))

# COMMAND ----------

# DBTITLE 1,3g Public/private vs. name, type vs. name, impossible numbers
types = q("""
select
  count_if(operatorTypeId = 'private' and lower(name) rlike '\\\\b(govt|government|civil hospital|district hospital|phc|chc|primary health cent|community health cent|esi|aiims|municipal)\\\\b') as private_but_govt_name,
  count_if(operatorTypeId in ('public','government') and lower(name) rlike '\\\\b(pvt|private|ltd)\\\\b')            as public_but_private_name,
  count_if(facilityTypeId = 'clinic'  and lower(name) rlike 'hospital')                                               as clinic_named_hospital,
  count_if(facilityTypeId = 'hospital' and lower(name) rlike '\\\\b(clinic|dental)\\\\b' and lower(name) not rlike 'hospital') as hospital_named_clinic,
  count_if(facilityTypeId = 'dentist' and lower(name) not rlike 'dent|tooth|teeth|smile|oral|ortho') as dentist_without_dental_name
from facn""").toPandas()
display(types)
for c in types.columns:
    record("3 Conflicts", f"facilities: {c}", int(types[c][0]))

nums = q("""
with x as (select *, try_cast(numberDoctors as double) d, try_cast(capacity as double) b, try_cast(yearEstablished as double) y from facn)
select
  count_if(numberDoctors is not null and d is null)  as doctors_not_numeric,
  count_if(capacity is not null and b is null)       as capacity_not_numeric,
  count_if(yearEstablished is not null and y is null) as year_not_numeric,
  count_if(d <= 0) as doctors_zero_or_negative,
  count_if(d > 2000) as doctors_over_2000,
  count_if(b <= 0) as beds_zero_or_negative,
  count_if(b > 5000) as beds_over_5000,
  count_if(y < 1800 or y > year(current_date())) as year_impossible,
  count_if(d > 0 and b > 0 and d > 5 * b and d > 50) as doctors_far_exceed_beds
from x""").toPandas()
display(nums)
for c in nums.columns:
    record("3 Conflicts", f"facilities: {c}", int(nums[c][0]))
display(q("""select name, address_city, numberDoctors, capacity, yearEstablished from facn
             where try_cast(numberDoctors as double) > 2000 or try_cast(capacity as double) > 5000
                or try_cast(yearEstablished as double) < 1800 or try_cast(yearEstablished as double) > year(current_date())
             order by try_cast(numberDoctors as double) desc nulls last limit 25"""))

# COMMAND ----------

# DBTITLE 1,3h Shifted / misaligned rows (values in the wrong column)
shift = q("""
select unique_id, name, address_country, address_countryCode, address_stateOrRegion, address_city, latitude, longitude
from fac
where (address_country is not null and lower(trim(address_country)) not in ('india','null',''))
   or (address_countryCode is not null and upper(trim(address_countryCode)) not in ('IN','NULL',''))""")
display(shift)
record("3 Conflicts", "facilities: rows with non-India / garbage country (likely shifted columns)", int(shift.count()))

# COMMAND ----------

# DBTITLE 1,3i Phone contradictions
ph = q("""
with x as (select *, from_json(phone_numbers,'array<string>') a from facn)
select
  count_if(officialPhone is not null and a is not null and not array_contains(a, officialPhone)) as official_phone_not_in_phone_list,
  count_if(officialPhone is not null and regexp_replace(officialPhone,'[^0-9]','') not rlike '^(91)?[0-9]{10}$' and regexp_replace(officialPhone,'[^0-9]','') not rlike '^(91)?1800[0-9]{6,7}$') as official_phone_bad_length,
  count_if(officialPhone is not null and officialPhone not like '+91%') as official_phone_not_plus91
from x""").toPandas()
display(ph)
for c in ph.columns:
    record("3 Conflicts", f"facilities: {c}", int(ph[c][0]))

# COMMAND ----------

# DBTITLE 1,3j Conflicts inside the pincode directory and NFHS
pin_geo = q(f"""
with o as (select p.*, try_cast(p.latitude as double) la, try_cast(p.longitude as double) lo from pinn p)
select count(*) offices,
       count_if(latitude is null) office_coords_missing,
       count_if(latitude is not null and la is null) office_coords_not_numeric,
       count_if(la is not null and not (la between 6 and 37 and lo between 68 and 98)) office_outside_india,
       count_if(la between 68 and 98 and lo between 6 and 37) office_lat_lon_swapped,
       count_if(la is not null and la = lo) office_lat_equals_lon,
       count_if(l.lat is not null and la between 6 and 37 and lo between 68 and 98 and {haversine_km('la','lo','l.lat','l.lon')} > 50) office_50km_from_own_pincode_centre
from o join pin_lookup l using (pincode)""").toPandas()
display(pin_geo)
for c in pin_geo.columns[1:]:
    record("3 Conflicts", f"pincode_dir: {c}", int(pin_geo[c][0]))

# NFHS: percentage columns outside 0–100, or stored as text with markers
nf_cols = [c for c, t in spark.table("nfhs").dtypes if c.endswith("_pct")]
nf = q("select " + ", ".join(
    f"count_if(try_cast(`{c}` as double) < 0 or try_cast(`{c}` as double) > 100) as `{c}`" for c in nf_cols) + " from nfhsn").toPandas().T
nf.columns = ["values_outside_0_100"]
display(nf[nf.values_outside_0_100 > 0].reset_index(names="column"))
record("3 Conflicts", "nfhs: % columns with values outside 0–100", int((nf.values_outside_0_100 > 0).sum()))

# COMMAND ----------

# DBTITLE 1,§3 Findings
# MAGIC %md
# MAGIC ### §3 Findings — contradictions
# MAGIC **Location is the biggest problem.** Comparing each facility's coordinates, its pincode's location and its city's location (3f):
# MAGIC
# MAGIC | Result | Facilities |
# MAGIC |---|---|
# MAGIC | All three agree (within 50 km) | ~6,540 (85% of checkable) |
# MAGIC | **Coordinates** are the odd one out | ~530 |
# MAGIC | **Pincode** is the odd one out | ~345 |
# MAGIC | **City label** is the odd one out | ~150 |
# MAGIC | All three disagree | ~200 |
# MAGIC
# MAGIC So **no single location field can be trusted on its own**, but a 2-out-of-3 vote can fix most rows automatically. Examples: *Aravind Eye Hospital, Hyderabad* has a Delhi pincode (110002). Several "Hyderabad" facilities have Visakhapatnam/Vijayawada pincodes and state = Andhra Pradesh, which looks like a leftover from before Telangana split off from Andhra Pradesh in 2014.
# MAGIC
# MAGIC - **Coordinates vs. own pincode (3c):** 765 facilities are >200 km from their pincode and 795 more are 50–200 km away. 212 have a pincode that doesn't exist in the directory.
# MAGIC - **State vs. pincode state:** 65 direct contradictions (3d). **City vs. pincode area:** 1,183 don't match (3e). This check is fuzzy, so confirm with 3f before acting.
# MAGIC - **Coordinates themselves:** 6 are outside India (see 02). None are swapped, at (0,0), or disagree with the `coordinates` JSON. 103 facilities share identical coordinates with another facility (at most 4 per point). That's minor, but they may have been geocoded to a locality rather than the building.
# MAGIC - **29 rows are broken by column shifting.** Text from `description` has spilled into later columns, so `address_country` holds numbers, JSON, or `kie`, and the name is a sentence fragment. Re-parse them from source or drop them.
# MAGIC - **Labels contradicting names (small):** 51 "clinics" are named *Hospital* and 25 "hospitals" are named *Clinic*. 4 "private" facilities have government names (*District Hospital*, *PHC*, *ESI*…).
# MAGIC - **Impossible numbers:** 7 facilities claim >2,000 doctors (max 15,000), 13 claim 0 doctors or 0 beds, 13 have an impossible founding year, and 5 have far more doctors than beds. Treat `numberDoctors` and `capacity` as unreliable.
# MAGIC - **Phones:** 44 official phones aren't in the facility's own phone list, and 31 have the wrong number of digits.
# MAGIC - **The pincode directory's own coordinates are noisy:** 12,007 offices have none, 2,607 are outside India (**791 swapped lat/lon, 730 lat = lon**), and ~11,900 are >50 km from their pincode's centre. Use the **median** office per pincode (as this notebook does), never the average or first row.
# MAGIC - **NFHS:** every percentage is within 0–100. No contradictions found.

# COMMAND ----------

# DBTITLE 1,§4 Needs adjusting
# MAGIC %md
# MAGIC ## §4 — What needs to be "adjusted"?
# MAGIC Values that are *right in meaning but wrong in form*: spelling, capitalization, old names, spacing, formats, numbers stored as text.

# COMMAND ----------

# DBTITLE 1,4a State names, all three tables
display(state_map.groupby(["source", "issue"]).agg(distinct_values=("raw", "count"), rows=("rows", "sum")).reset_index())
for _, r in state_map[state_map.issue != "exact"].groupby(["source", "issue"]).rows.sum().reset_index().iterrows():
    record("4 Adjust", f"{r.source}: state values that are '{r.issue}'", int(r.rows))
display(state_map[state_map.issue != "exact"].sort_values(["source", "issue", "rows"], ascending=[True, True, False]))

# COMMAND ----------

# DBTITLE 1,4b Same state spelled differently across tables
cross = (state_map.dropna(subset=["canonical"])
         .groupby(["canonical", "source"]).raw.apply(lambda s: " | ".join(sorted(set(map(str, s))))).unstack())
display(cross.reset_index())

# COMMAND ----------

# DBTITLE 1,4c City spelling, capitalization and renamed cities
cities = q("""select address_city raw, count(*) n, first(address_stateOrRegion, true) state_raw from facn
              where address_city is not null group by 1""").toPandas()
cities["key"] = cities.raw.map(ckey)
cities["canon_key"] = cities.key.map(city_canon)
cities["state"] = cities.state_raw.map(lambda s: classify_state(s)[0])

# (i) case / spacing / punctuation variants: same letters, different raw text
v = cities.groupby("key").filter(lambda d: len(d) > 1).sort_values(["key", "n"], ascending=[True, False])
case_variants = v.groupby("key").agg(spellings=("raw", lambda s: " | ".join(s)), rows=("n", "sum"), n_spellings=("raw", "count")).reset_index()
display(case_variants.sort_values("rows", ascending=False))
record("4 Adjust", "city: groups spelled with different case/spacing", len(case_variants))

# (ii) old vs new names both present
ren = cities[cities.key != cities.canon_key].groupby("canon_key").agg(old_names=("raw", lambda s: " | ".join(sorted(set(s)))), old_rows=("n", "sum")).reset_index()
ren = ren.merge(cities[cities.key == cities.canon_key].groupby("key").agg(new_names=("raw", lambda s: " | ".join(sorted(set(s)))), new_rows=("n", "sum")).reset_index(),
                left_on="canon_key", right_on="key", how="left").drop(columns="key")
display(ren)
record("4 Adjust", "city: facilities using an old city name (e.g. Gurgaon, Bangalore)", int(ren.old_rows.sum()))

# (iii) probable misspellings: different keys, very similar, same state
pairs = []
ck = cities.groupby(["state", "canon_key"]).agg(raw=("raw", "first"), n=("n", "sum")).reset_index()
for st_name, d in ck.dropna(subset=["state"]).groupby("state"):
    keys = d.to_dict("records")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            if min(len(a["canon_key"]), len(b["canon_key"])) < 5 or a["canon_key"][0] != b["canon_key"][0]:
                continue
            r = difflib.SequenceMatcher(None, a["canon_key"], b["canon_key"]).ratio()
            if r >= 0.88:
                main, other = (a, b) if a["n"] >= b["n"] else (b, a)
                pairs.append({"state": st_name, "likely_correct": main["raw"], "rows": main["n"], "variant": other["raw"], "variant_rows": other["n"], "similarity": round(r, 2)})
near = pd.DataFrame(pairs).sort_values("similarity", ascending=False) if pairs else pd.DataFrame()
display(near)
record("4 Adjust", "city: probable misspelling pairs (same state, ≥88% similar)", len(near))

# COMMAND ----------

# DBTITLE 1,4d District names: NFHS vs pincode directory
pin_d = q("select distinct statename, district from pinn where district is not null").toPandas()
pin_d["state"] = pin_d.statename.map(lambda s: classify_state(s)[0])
pin_d["key"] = pin_d.district.map(ckey)
nf_d = q("select distinct state_ut, district_name from nfhs").toPandas()
nf_d["state"] = nf_d.state_ut.map(lambda s: classify_state(s)[0])
nf_d["key"] = nf_d.district_name.map(ckey)
nf_d["has_extra_whitespace"] = nf_d.district_name.map(lambda s: s is not None and (s != s.strip() or "  " in s))

def match_district(r):
    cands = pin_d[pin_d.state == r.state]
    if r.key in set(cands.key):
        return pd.Series({"match": "exact (after case/space normalization)", "suggested_pincode_district": cands[cands.key == r.key].district.iloc[0]})
    m = difflib.get_close_matches(r.key, list(cands.key), n=1, cutoff=0.75)
    if m:
        return pd.Series({"match": "fuzzy — spelling differs", "suggested_pincode_district": cands[cands.key == m[0]].district.iloc[0]})
    return pd.Series({"match": "no match (renamed/split district?)", "suggested_pincode_district": None})

nf_d = pd.concat([nf_d, nf_d.apply(match_district, axis=1)], axis=1)
display(nf_d.match.value_counts().rename_axis("NFHS district → pincode district").reset_index(name="districts"))
display(nf_d[nf_d.match != "exact (after case/space normalization)"].sort_values(["match", "state"])[["state_ut", "district_name", "match", "suggested_pincode_district"]])
record("4 Adjust", "nfhs: district names with extra whitespace", int(nf_d.has_extra_whitespace.sum()))
record("4 Adjust", "nfhs: districts spelled differently from pincode dir", int((nf_d.match == "fuzzy — spelling differs").sum()))
record("4 Adjust", "nfhs: districts with no pincode-dir match", int((nf_d.match == "no match (renamed/split district?)").sum()))

# COMMAND ----------

# DBTITLE 1,4e Whitespace and mixed capitalization, every text column
rows = []
for short, name in TABLES.items():
    scols = [c for c, t in spark.table(short).dtypes if t == "string"]
    exprs = []
    for i, c in enumerate(scols):
        exprs += [f"count_if(`{c}` != trim(`{c}`)) as w{i}", f"count_if(`{c}` rlike '\\\\s{{2,}}') as d{i}"]
    r = q(f"select {', '.join(exprs)} from {short}n").collect()[0].asDict()
    for i, c in enumerate(scols):
        if r[f"w{i}"] or r[f"d{i}"]:
            rows.append({"table": name, "column": c, "leading/trailing spaces": r[f"w{i}"], "double spaces": r[f"d{i}"]})
ws = pd.DataFrame(rows)
display(ws.sort_values("leading/trailing spaces", ascending=False) if len(ws) else ws)

# Categorical columns where the same value appears with different capitalization
CATS = {"fac": ["address_city", "address_stateOrRegion", "facilityTypeId", "operatorTypeId", "organization_type", "address_country"],
        "pin": ["statename", "district", "officetype", "delivery"],
        "nfhs": ["state_ut", "district_name"]}
rows = []
for short, cols in CATS.items():
    for c in cols:
        r = q(f"""select count(*) groups, coalesce(sum(n),0) values from (
                    select lower(trim(`{c}`)) k, count(distinct `{c}`) n from {short}n where `{c}` is not null group by 1 having count(distinct `{c}`) > 1)""").collect()[0]
        rows.append({"table": TABLES[short], "column": c, "values written >1 way": r.groups, "raw variants": r["values"]})
        upper = scalar(f"select count_if(`{c}` = upper(`{c}`) and `{c}` rlike '[A-Z]{{3}}') from {short}n")
        rows[-1]["ALL-CAPS values"] = upper
display(pd.DataFrame(rows))

# COMMAND ----------

# DBTITLE 1,4f Pincode and phone formats
pf = q("""
select case when address_zipOrPostcode is null then 'missing'
            when address_zipOrPostcode rlike '^[1-9][0-9]{5}$' then 'ok: 6 digits'
            when regexp_replace(address_zipOrPostcode,'[^0-9]','') rlike '^[1-9][0-9]{5}$' then 'fixable: spaces/punctuation (e.g. 560 052)'
            when address_zipOrPostcode rlike '^[0-9]+$' then concat('wrong length: ', length(address_zipOrPostcode), ' digits')
            else 'not a pincode' end as pincode_format, count(*) n, slice(collect_set(address_zipOrPostcode), 1, 5) examples
from facn group by 1 order by n desc""").toPandas()
display(pf)
record("4 Adjust", "facilities: pincodes fixable by stripping spaces", int(pf.loc[pf.pincode_format.str.startswith('fixable'), 'n'].sum()))
record("4 Adjust", "facilities: pincodes unusable (wrong length / not a pincode)",
       int(pf.loc[pf.pincode_format.str.startswith(('wrong', 'not')), 'n'].sum()))

display(q("""
select regexp_replace(regexp_replace(officialPhone, '[0-9]', '9'), '9+', '9…') phone_shape, count(*) n, slice(collect_set(officialPhone),1,3) examples
from facn where officialPhone is not null group by 1 order by n desc limit 15"""))

# COMMAND ----------

# DBTITLE 1,4g Category vocabularies (typos, synonyms)
for c in ["organization_type", "facilityTypeId", "operatorTypeId", "affiliationTypeIds", "source"]:
    display(q(f"select `{c}`, count(*) n from fac group by 1 order by n desc limit 25"))

# COMMAND ----------

# DBTITLE 1,4h Specialty vocabulary
spec = q("""
with s as (select unique_id, explode(array_distinct(from_json(specialties,'array<string>'))) sp from fac)
select sp, count(*) facilities from s group by sp""").toPandas()
spec["key"] = spec.sp.str.lower().str.replace(r"[^a-z]", "", regex=True)
spec_summary = {
    "distinct specialty strings": len(spec),
    "used by only 1 facility": int((spec.facilities == 1).sum()),
    "contain spaces/punctuation (not camelCase)": int(spec.sp.str.contains(r"[^A-Za-z]").sum()),
    "case-only duplicate groups": int(spec.groupby("key").sp.count().gt(1).sum()),
    "covering 95% of all tags": int((spec.sort_values("facilities", ascending=False).facilities.cumsum() / spec.facilities.sum() <= 0.95).sum() + 1),
}
display(pd.DataFrame([spec_summary]))
for k, val in spec_summary.items():
    record("4 Adjust", f"specialties: {k}", val)
display(spec[spec.sp.str.contains(r"[^A-Za-z]")].sort_values("facilities", ascending=False).head(25))
display(spec.groupby("key").filter(lambda d: len(d) > 1).sort_values("key"))

# COMMAND ----------

# DBTITLE 1,4i Numbers stored as text
rows = []
for short, name in TABLES.items():
    for c, t in spark.table(short).dtypes:
        if t != "string":
            continue
        r = q(f"""select count(`{c}`) nonnull, count_if(try_cast(`{c}` as double) is not null) numeric,
                         slice(collect_set(case when try_cast(`{c}` as double) is null then `{c}` end), 1, 5) non_numeric_examples
                  from {short}n""").collect()[0]
        if r.nonnull and r.numeric / r.nonnull >= 0.8:  # column is mostly numbers
            rows.append({"table": name, "column": c, "non-null": r.nonnull, "parse as number": r.numeric,
                         "don't parse": r.nonnull - r.numeric, "examples that don't parse": r.non_numeric_examples})
num_text = pd.DataFrame(rows).sort_values("don't parse", ascending=False)
display(num_text)
record("4 Adjust", "numeric columns stored as text (all tables)", len(num_text))
record("4 Adjust", "values in those columns that won't convert to numbers", int(num_text["don't parse"].sum()))

# COMMAND ----------

# DBTITLE 1,§4 Findings
# MAGIC %md
# MAGIC ### §4 Findings — values that need adjusting
# MAGIC **States** (4a, 4b): same state, different spellings, across and within tables:
# MAGIC - `facilities`: **235 rows have a city in the state field** (*Thane, Mumbai, Navi Mumbai, Pune, Chennai, Hyderabad, Malappuram…*), 34 use old names or abbreviations (`Orissa`, `Pondicherry`, `Up`, `U.p.`, `New Delhi`), 5 are misspelled (`Chattisgarh`, `Telengana`), 92 differ only in caps or format (`Jammu & Kashmir`), 17 are JSON/junk and 91 are unrecognized (`Punjab Region`, `kie`…).
# MAGIC - `pincode directory`: every state is **ALL CAPS** (`TAMIL NADU`), and 715 rows have state = `NA`.
# MAGIC - `NFHS`: `Maharastra` is **misspelled in all 36 Maharashtra rows**. It also uses `&` (`Jammu & Kashmir`, `Andaman & Nicobar Islands`) and `NCT of Delhi`, and has `" Lakshadweep "` with surrounding spaces.
# MAGIC - ⇒ The three tables **won't join on state** until all are mapped to one canonical list (the `state_map` built at the top of this notebook already does this).
# MAGIC
# MAGIC **Cities** (4c):
# MAGIC - **460 facilities use an old city name**, often alongside the new one: `Bangalore` 204 vs `Bengaluru` 110, `Gurgaon` 82 vs `Gurugram` 65, `Trivandrum`/`Thiruvananthapuram`, `Cochin`/`Kochi`, `Calicut`/`Kozhikode`, `Allahabad`/`Prayagraj`, `Mysore`/`Mysuru`, `Mangalore`/`Mangaluru`, `Baroda`/`Vadodara`, `Vizag`/`Visakhapatnam`. A user searching "Bengaluru" would miss two-thirds of the city's facilities.
# MAGIC - 17 cities are spelled with different caps or spacing, and **47 pairs are probable misspellings** in the same state (list in 4c). Review these by eye before merging.
# MAGIC
# MAGIC **Districts** (4d): **704 of 706 NFHS district names have extra spaces** (`"Srikakulam "`). After trimming, 56 are spelled differently from the pincode directory and 37 have no match at all, likely renamed or newly split districts. These need a hand-made mapping before district-level analysis.
# MAGIC
# MAGIC **Formats** (4e–4i):
# MAGIC - **Pincodes:** 157 can be fixed by removing spaces (`560 052` → `560052`). 99 can't be used (wrong length or not a number).
# MAGIC - **Phones:** most are `+91` followed by 10 digits. A few toll-free and odd-length numbers exist (see 3i).
# MAGIC - **Categories:** `facilityTypeId` has both `farmacy` and `pharmacy`. `operatorTypeId` has both `public` and `government`, plus `null`.
# MAGIC - **Specialties:** 2,933 distinct strings, but **194 cover 95% of all tags**. 1,975 appear only once, 617 aren't in the standard camelCase format, and 363 differ only by capitalization. Map them to a fixed list of ~200 before building a "service" dropdown.
# MAGIC - **Numbers stored as text:** 41 numeric columns are text (in `facilities` and many NFHS `_pct` columns), with 579 values that won't convert. In NFHS these are mostly values in parentheses like `(18.0)`, which is how NFHS marks estimates from small samples. Keep the number but flag it as low-confidence. Convert them with `try_cast` and keep a flag for the values that failed.

# COMMAND ----------

# DBTITLE 1,§5 Scorecard
# MAGIC %md
# MAGIC ## §5 — Scorecard
# MAGIC Every headline number from this notebook in one place (also returned as JSON when run as a job).

# COMMAND ----------

# DBTITLE 1,Scorecard
scorecard = pd.DataFrame(SUMMARY)
display(scorecard)

# COMMAND ----------

# DBTITLE 1,Recommended cleaning rules
# MAGIC %md
# MAGIC ### Recommended cleaning rules (for `Data Cleaning: Raw to Bronze`), in order
# MAGIC 1. **Missing:** convert `"null"`, `"NA"`, `""`, `[]` and similar to real NULL. Drop the 57 unidentifiable rows and the 29 column-shifted rows, or re-parse the shifted ones from source.
# MAGIC 2. **Text hygiene:** trim whitespace everywhere (fixes 704 NFHS districts and `" Lakshadweep "`).
# MAGIC 3. **States:** map every table to the 36 canonical names using `state_map`. Where a city sits in the state field, take the state from the pincode instead.
# MAGIC 4. **Cities:** use the current name (Bangalore → Bengaluru, Gurgaon → Gurugram…) and merge case variants. Review the 47 near-duplicate pairs by hand.
# MAGIC 5. **Pincodes:** strip spaces, keep only valid 6-digit codes, and resolve pincode → district/state by most common value.
# MAGIC 6. **Location vote:** use the 2-out-of-3 check from 3f to flag which of coordinates/pincode/city is wrong. Re-derive bad coordinates from the pincode median and fix bad pincodes from coordinates. Keep a `location_confidence` column.
# MAGIC 7. **Duplicates:** remove the 11 exact copies, then merge same name+city/address rows, combining their phone lists. Flag phones and emails shared by many unrelated facilities as unreliable.
# MAGIC 8. **Lists:** parse JSON arrays, remove blank items, deduplicate items, and map specialties to a standard list of ~200.
# MAGIC 9. **Numbers:** convert to numeric with `try_cast`. Null out impossible values (doctors >2,000 or 0, founding year outside 1800–today) and keep a flag.
# MAGIC 10. **Pincode directory:** fix the 791 swapped lat/lon offices and ignore offices where lat = lon or that fall outside India.
# MAGIC 11. **Joins:** always join NFHS on state + district (never district alone), using a hand-made table for the 93 district names that don't match.

# COMMAND ----------

dbutils.notebook.exit(json.dumps(SUMMARY, default=str))
