# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 03_geo — PIN centres, place gazetteer, and `locate()`
# MAGIC
# MAGIC Turns the India Post directory into a lookup that can place **any user input** — a PIN, a place name, or `"lat, lon"` — on the map.
# MAGIC
# MAGIC **Input (read-only):** `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.india_post_pincode_directory` (165,627 post offices, 19,586 PINs)
# MAGIC
# MAGIC **Outputs (CREATE OR REPLACE — safe to re-run), in `workspace.datathon_2026`:**
# MAGIC | table | one row per | columns |
# MAGIC |---|---|---|
# MAGIC | `pin_centers` | PIN | `pincode, lat, lon, district, state, n_offices, n_offices_with_coords, coord_source, admin_source` |
# MAGIC | `geo_lookup` | place × type × district × state | `place, type, lat, lon, district, state, nearest_pin, n_pins` |
# MAGIC | `cleaning_audit` | rule | shared team audit table (appended) |
# MAGIC
# MAGIC **Rules**
# MAGIC 1. **Inside India:** a coordinate is valid only inside India's bounding box (lat 6–37.6, lon 68–97.5). This drops the swapped and lat = lon offices found in `03_Data_Quality_Audit`.
# MAGIC 2. **Filler values:** inside one PIN area (first 3 digits = postal sorting district), the same latitude (or longitude) used by 10+ offices with 5+ different partner values is a filler, not a location. Example: `15.5934` is used for hundreds of Gujarat and Kerala offices, which puts them in the Arabian Sea.
# MAGIC 3. **Far from its PIN area:** each 3-digit PIN area is anchored on its densest cluster of distinct coordinates. An office more than 200 km from that anchor has wrong coordinates (e.g. 118 Firozabad offices placed 300 km south). Exception: the island UTs.
# MAGIC    PIN areas are used instead of district names because district labels are sometimes wrong (Bastar PINs labelled *Raipur*) while PIN codes are routing data.
# MAGIC 4. **PIN centre** = **median** of its valid office coordinates. PINs with no valid office get their district's median (`coord_source = 'district_median'`), then their PIN area's anchor (`'pin_area_anchor'`).
# MAGIC 5. **PIN district and state** = the value most of its offices use, with states mapped to the 36 canonical names. PINs where the directory has no state take the state and district of the nearest PIN that has one (`admin_source = 'nearest_pin'`).
# MAGIC 6. **`geo_lookup` rows:**
# MAGIC    - `pin`: the PIN itself.
# MAGIC    - `district`: median of its PIN centres within 150 km of the district's densest cluster, so a few mislabelled PINs can't drag it.
# MAGIC    - `town`: office name minus ` B.O` / ` S.O` / ` H.O` / ` G.P.O` (repeated suffixes too), median of those offices.
# MAGIC 7. **`nearest_pin`** = closest PIN centre by great-circle distance (BallTree, haversine).
# MAGIC
# MAGIC `locate(text)` → `(lat, lon, nearest_pin, district, state)`. Exact match first, then a `difflib` fuzzy match (order in §5).

# COMMAND ----------

# DBTITLE 1,0. Setup
import re, difflib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from pyspark.sql import functions as F

SRC = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset"
PIN_SRC = f"{SRC}.india_post_pincode_directory"
TGT = "workspace.datathon_2026"
PIN_CENTERS = f"{TGT}.pin_centers"
GEO_LOOKUP = f"{TGT}.geo_lookup"
AUDIT = f"{TGT}.cleaning_audit"

LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 6.0, 37.6, 68.0, 97.5   # India incl. Andaman & Nicobar and Lakshadweep
EARTH_KM = 6371.0088

RUN_ID = "geo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
audit_rows = []
def log(step, rule, n, note=""):
    audit_rows.append((RUN_ID, f"geo-{step}", rule, int(n), note))
    print(f"[{step}] {rule}: {n:,} {note}")

CANON_STATES = [
    "Andaman and Nicobar Islands", "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
    "Chhattisgarh", "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Goa", "Gujarat", "Haryana",
    "Himachal Pradesh", "Jammu and Kashmir", "Jharkhand", "Karnataka", "Kerala", "Ladakh", "Lakshadweep",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Puducherry",
    "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
    "West Bengal",
]
def skey(s):
    s = re.sub(r"^\s*the\s+", "", str(s).lower())
    return re.sub(r"[^a-z]", "", s.replace("&", "and"))
STATE_KEYS = {skey(s): s for s in CANON_STATES}
STATE_KEYS.update({"maharastra": "Maharashtra", "nctofdelhi": "Delhi", "orissa": "Odisha", "pondicherry": "Puducherry"})

def norm_place(s):
    """Lowercase, '&' → 'and', single spaces, no stray punctuation at the ends."""
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"\s+", " ", s).strip(" .,-")
    return s

OFFICE_SUFFIX = re.compile(r"\s+(b\.?\s*o|s\.?\s*o|h\.?\s*o|g\.?\s*p\.?\s*o)\.?$", re.I)

def to_rad(lat, lon):
    return np.radians(np.column_stack([lat, lon]))

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_KM * np.arcsin(np.sqrt(a))

def dense_anchor(df, keys):
    """Per group: the median of the distinct coordinates in the group's densest 3×3-degree neighbourhood.
    Counting *distinct* coordinates stops a stack of identical junk points from winning."""
    u = df.dropna(subset=["lat", "lon"]).drop_duplicates(keys + ["lat", "lon"]).copy()
    u["cy"], u["cx"] = np.floor(u["lat"]).astype(int), np.floor(u["lon"]).astype(int)
    c = u.groupby(keys + ["cy", "cx"]).size().rename("n").reset_index()
    nb = (pd.concat([c.assign(cy=c["cy"] + dy, cx=c["cx"] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
            .groupby(keys + ["cy", "cx"])["n"].sum().reset_index()
            .merge(c[keys + ["cy", "cx"]], on=keys + ["cy", "cx"]))
    best = (nb.sort_values(["n", "cy", "cx"], ascending=[False, True, True]).drop_duplicates(keys)
              .rename(columns={"cy": "ay", "cx": "ax"})[keys + ["ay", "ax"]])
    u = u.merge(best, on=keys)
    core = u[(u["cy"] - u["ay"]).abs().le(1) & (u["cx"] - u["ax"]).abs().le(1)]
    return core.groupby(keys)[["lat", "lon"]].median().rename(columns={"lat": "a_lat", "lon": "a_lon"})

ISLAND_UTS = {"Andaman and Nicobar Islands", "Lakshadweep"}   # legitimately spread over hundreds of km of sea
FILLER_MIN_OFFICES, FILLER_MIN_PARTNERS = 10, 5
PREFIX_MAX_KM = 200

# COMMAND ----------

# DBTITLE 1,1. Load the directory
pin_raw = (spark.read.table(PIN_SRC)   # read-only
           .select("officename", "pincode", "district", "statename", "latitude", "longitude")
           .toPandas())
RAW_COUNT = len(pin_raw)
print(f"Post offices: {RAW_COUNT:,}")

# COMMAND ----------

# DBTITLE 1,2. Valid coordinates and canonical names
# MAGIC %md
# MAGIC ## 2. Valid coordinates and canonical names

# COMMAND ----------

o = pin_raw.copy()
for c in ["officename", "district", "statename", "latitude", "longitude"]:   # the table stores missing values as the text "NA"
    o[c] = o[c].where(~o[c].astype(str).str.strip().str.upper().isin(["NA", "", "NULL", "NONE", "NAN"]))
o["pincode"] = o["pincode"].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
bad_pin = ~o["pincode"].str.fullmatch(r"[1-9]\d{5}")
log(2, "offices with an invalid PIN (dropped)", bad_pin.sum())
o = o[~bad_pin].copy()
N_VALID_PINS = o["pincode"].nunique()

o["lat"] = pd.to_numeric(o["latitude"], errors="coerce")
o["lon"] = pd.to_numeric(o["longitude"], errors="coerce")
in_box = o["lat"].between(LAT_MIN, LAT_MAX) & o["lon"].between(LON_MIN, LON_MAX)
log(2, "offices with coordinates missing or outside India (ignored for centres)", (~in_box).sum(),
    f"(missing: {int(o['lat'].isna().sum()):,})")
o.loc[~in_box, ["lat", "lon"]] = np.nan

o["state"] = o["statename"].map(lambda v: STATE_KEYS.get(skey(v)) if pd.notna(v) else None)
unmapped = sorted(o.loc[o["state"].isna(), "statename"].dropna().unique())
assert not unmapped, f"Unmapped state names: {unmapped}"
o["district"] = o["district"].fillna("").map(lambda s: re.sub(r"\s+", " ", s).strip().title() or None)

def strip_suffix(name):
    prev = None
    while prev != name:                      # "Relawan  B.O. B.O" → "Relawan"; "Aghara So BO" → "Aghara"
        prev, name = name, OFFICE_SUFFIX.sub("", name.strip())
    return norm_place(name)
o["town"] = o["officename"].fillna("").map(strip_suffix)
log(2, "office names with a B.O / S.O / H.O / G.P.O suffix stripped",
    o["officename"].fillna("").str.contains(OFFICE_SUFFIX.pattern, flags=re.I, regex=True).sum())

# Rule 2 — filler values repeated inside one PIN area
o["pin_area"] = o["pincode"].str[:3]
filler = pd.Series(False, index=o.index)
for col, other in [("lat", "lon"), ("lon", "lat")]:
    v = o.dropna(subset=[col]).groupby(["pin_area", col]).agg(n=(other, "size"), k=(other, "nunique")).reset_index()
    v = v[(v["n"] >= FILLER_MIN_OFFICES) & (v["k"] >= FILLER_MIN_PARTNERS)]
    filler |= pd.Series(pd.MultiIndex.from_frame(o[["pin_area", col]]).isin(pd.MultiIndex.from_frame(v[["pin_area", col]])),
                        index=o.index)
log(2, "offices with a filler coordinate (ignored)", filler.sum())
o.loc[filler, ["lat", "lon"]] = np.nan

# Rule 3 — far from the PIN area's dense cluster
anchor = dense_anchor(o, ["pin_area"])
o = o.join(anchor, on="pin_area")
o["km_from_area"] = haversine_km(o["lat"], o["lon"], o["a_lat"], o["a_lon"])
far = (o["km_from_area"] > PREFIX_MAX_KM) & ~o["state"].isin(ISLAND_UTS)
log(2, f"offices > {PREFIX_MAX_KM} km from their PIN area (ignored)", far.sum())
o.loc[far, ["lat", "lon"]] = np.nan
log(2, "offices with a usable coordinate", o["lat"].notna().sum())

# COMMAND ----------

# DBTITLE 1,3. pin_centers
# MAGIC %md
# MAGIC ## 3. `pin_centers` — one row per PIN

# COMMAND ----------

def mode(s):
    s = s.dropna()
    return s.value_counts().index[0] if len(s) else None

g = o.groupby("pincode")
pin_centers = pd.DataFrame({
    "lat": g["lat"].median(),
    "lon": g["lon"].median(),
    "district": g["district"].agg(mode),
    "state": g["state"].agg(mode),
    "n_offices": g.size(),
    "n_offices_with_coords": g["lat"].count(),
}).reset_index()
pin_centers["coord_source"] = np.where(pin_centers["lat"].notna(), "offices", None)

# PINs without any valid office: use the median of the other PIN centres in their district
dist_med = (pin_centers[pin_centers["lat"].notna()]
            .groupby(["district", "state"])[["lat", "lon"]].median()
            .rename(columns={"lat": "d_lat", "lon": "d_lon"}))
pin_centers = pin_centers.join(dist_med, on=["district", "state"])
fill = pin_centers["lat"].isna() & pin_centers["d_lat"].notna()
pin_centers.loc[fill, "lat"] = pin_centers.loc[fill, "d_lat"]
pin_centers.loc[fill, "lon"] = pin_centers.loc[fill, "d_lon"]
pin_centers.loc[fill, "coord_source"] = "district_median"
pin_centers = pin_centers.drop(columns=["d_lat", "d_lon"])
pin_centers["pin_area"] = pin_centers["pincode"].str[:3]
pin_centers = pin_centers.join(anchor, on="pin_area")
fill2 = pin_centers["lat"].isna() & pin_centers["a_lat"].notna()
pin_centers.loc[fill2, "lat"] = pin_centers.loc[fill2, "a_lat"]
pin_centers.loc[fill2, "lon"] = pin_centers.loc[fill2, "a_lon"]
pin_centers.loc[fill2, "coord_source"] = "pin_area_anchor"
pin_centers = pin_centers.drop(columns=["a_lat", "a_lon"])

# PINs the directory gives no state/district: take them from the nearest PIN that has one
pin_centers["admin_source"] = np.where(pin_centers["state"].notna(), "directory", None)
known = pin_centers[pin_centers["state"].notna() & pin_centers["lat"].notna()].reset_index(drop=True)
need = pin_centers["state"].isna() & pin_centers["lat"].notna()
if need.any():
    _, j = BallTree(to_rad(known["lat"], known["lon"]), metric="haversine").query(
        to_rad(pin_centers.loc[need, "lat"], pin_centers.loc[need, "lon"]), k=1)
    pin_centers.loc[need, "state"] = known["state"].to_numpy()[j[:, 0]]
    pin_centers.loc[need, "district"] = known["district"].to_numpy()[j[:, 0]]
    pin_centers.loc[need, "admin_source"] = "nearest_pin"

log(3, "PINs", len(pin_centers))
log(3, "PINs placed from their own offices", (pin_centers["coord_source"] == "offices").sum())
log(3, "PINs placed at their district median (no valid office coordinates)", fill.sum())
log(3, "PINs placed at their PIN-area anchor", fill2.sum())
log(3, "PINs with no location at all", pin_centers["lat"].isna().sum())
log(3, "PINs given state/district from the nearest PIN", need.sum())
log(3, "PINs with no state", pin_centers["state"].isna().sum())

# BallTree over every located PIN centre (used for nearest_pin and by locate())
pc_loc = pin_centers[pin_centers["lat"].notna()].reset_index(drop=True)
PIN_TREE = BallTree(to_rad(pc_loc["lat"], pc_loc["lon"]), metric="haversine")
PIN_IDS = pc_loc["pincode"].to_numpy()

def nearest_pins(lat, lon):
    _, idx = PIN_TREE.query(to_rad(lat, lon), k=1)
    return PIN_IDS[idx[:, 0]]

# COMMAND ----------

# DBTITLE 1,4. geo_lookup
# MAGIC %md
# MAGIC ## 4. `geo_lookup` — every searchable place

# COMMAND ----------

# PIN rows
geo_pin = pin_centers.assign(place=pin_centers["pincode"], type="pin", nearest_pin=pin_centers["pincode"], n_pins=1)

# District rows: median of the PIN centres placed from real offices
pc_real = pin_centers[(pin_centers["coord_source"] == "offices") & pin_centers["district"].notna()]
pc_real = pc_real.join(dense_anchor(pc_real, ["district", "state"]), on=["district", "state"])
d_km = haversine_km(pc_real["lat"], pc_real["lon"], pc_real["a_lat"], pc_real["a_lon"])
log(4, "PIN centres > 150 km from their district's cluster (left out of the district centre)", (d_km > 150).sum())
pc_real = pc_real[(d_km <= 150) | pc_real["state"].isin(ISLAND_UTS)]
geo_district = (pc_real.groupby(["district", "state"])
                .agg(lat=("lat", "median"), lon=("lon", "median"), n_pins=("pincode", "nunique"))
                .reset_index())
geo_district["place"] = geo_district["district"].map(norm_place)
geo_district["type"] = "district"

# Town rows: offices sharing a name inside one district; fall back to the PIN centre if the office has no coordinates
ot = o[o["town"] != ""].merge(pin_centers[["pincode", "lat", "lon"]].rename(columns={"lat": "p_lat", "lon": "p_lon"}),
                             on="pincode", how="left")
ot["lat"] = ot["lat"].fillna(ot["p_lat"])
ot["lon"] = ot["lon"].fillna(ot["p_lon"])
geo_town = (ot.groupby(["town", "district", "state"], dropna=False)
            .agg(lat=("lat", "median"), lon=("lon", "median"), n_pins=("pincode", "nunique"))
            .reset_index().rename(columns={"town": "place"}))
geo_town["type"] = "town"

geo = pd.concat([geo_district, geo_town], ignore_index=True)
has = geo["lat"].notna()
geo["nearest_pin"] = None
geo.loc[has, "nearest_pin"] = nearest_pins(geo.loc[has, "lat"], geo.loc[has, "lon"])

GEO_COLS = ["place", "type", "lat", "lon", "district", "state", "nearest_pin", "n_pins"]
geo_lookup = pd.concat([geo_pin[GEO_COLS], geo[GEO_COLS]], ignore_index=True)
geo_lookup["n_pins"] = geo_lookup["n_pins"].astype("int64")
for t, n in geo_lookup["type"].value_counts().items():
    log(4, f"geo_lookup rows of type {t}", n)
log(4, "town/district rows without coordinates", (~has).sum())

# COMMAND ----------

# DBTITLE 1,5. locate()
# MAGIC %md
# MAGIC ## 5. `locate(text)` — PIN, place name, or `"lat, lon"`
# MAGIC Returns `(lat, lon, nearest_pin, district, state)`, or `None` if nothing matches.
# MAGIC Order: coordinates → 6-digit PIN → exact district → exact town covering 2+ PINs → close district spelling (≥ 0.85) → exact one-PIN village → fuzzy town (≥ 0.8).
# MAGIC When one name exists in several states, the place covering more PINs wins.
# MAGIC `locate(text, explain=True)` also returns what was matched and how.

# COMMAND ----------

_places = geo_lookup[(geo_lookup["type"] != "pin") & geo_lookup["lat"].notna()]
_places = _places.sort_values("n_pins", ascending=False)
_district_by_name = _places[_places["type"] == "district"].drop_duplicates("place").set_index("place")  # biggest district per name
_town_by_name = _places[_places["type"] == "town"].drop_duplicates("place").set_index("place")          # biggest town per name
_district_names = list(_district_by_name.index)
_town_names = list(_town_by_name.index)
_pin_index = pin_centers.set_index("pincode")
_COORD = re.compile(r"\s*\(?\s*(-?\d{1,2}(?:\.\d+)?)\s*[,\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*\)?\s*")

def _fuzzy(q, index, names, cutoff):
    """Closest spelling; ties go to the place covering more PINs."""
    cands = difflib.get_close_matches(q, names, n=10, cutoff=cutoff)
    if not cands:
        return None, 0.0
    scored = sorted(((difflib.SequenceMatcher(None, q, c).ratio(), index.loc[c, "n_pins"], c) for c in cands), reverse=True)
    return scored[0][2], scored[0][0]

def _match_place(q):
    """Returns (row, how, score). Order: exact district → exact town (≥2 PINs) → close district spelling
    → exact single-PIN village → fuzzy town. So 'patana' finds Patna district, not the hamlet Patana."""
    if q in _district_by_name.index:
        return _district_by_name.loc[q], "exact district", 1.0
    if q in _town_by_name.index and _town_by_name.loc[q, "n_pins"] >= 2:
        return _town_by_name.loc[q], "exact town", 1.0
    name, score = _fuzzy(q, _district_by_name, _district_names, 0.85)
    if name:
        return _district_by_name.loc[name], "fuzzy district", score
    if q in _town_by_name.index:
        return _town_by_name.loc[q], "exact town", 1.0
    name, score = _fuzzy(q, _town_by_name, _town_names, 0.8)
    if name:
        return _town_by_name.loc[name], "fuzzy town", score
    return None, None, 0.0

def locate(text, explain=False):
    t = str(text).strip()
    m = _COORD.fullmatch(t)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            return None
        pin = nearest_pins([lat], [lon])[0]
        r = _pin_index.loc[pin]
        out, how = (lat, lon, pin, r["district"], r["state"]), {"matched": pin, "how": "coordinates → nearest PIN", "score": None}
    elif re.fullmatch(r"\d{6}", t.replace(" ", "")):
        pin = t.replace(" ", "")
        if pin not in _pin_index.index or pd.isna(_pin_index.loc[pin, "lat"]):
            return None
        r = _pin_index.loc[pin]
        out, how = (float(r["lat"]), float(r["lon"]), pin, r["district"], r["state"]), {"matched": pin, "how": "PIN", "score": None}
    else:
        r, kind, score = _match_place(norm_place(t))
        if r is None:
            return None
        out = (float(r["lat"]), float(r["lon"]), r["nearest_pin"], r["district"], r["state"])
        how = {"matched": f"{r.name} ({r['type']})", "how": kind, "score": round(score, 3)}
    return (out, how) if explain else out

TESTS = ["patna", "Muzaffarpur", "800001", "25.59, 85.13", "patana"]
locate_results = []
for t in TESTS:
    res = locate(t, explain=True)
    (lat, lon, pin, dist, st), how = res
    locate_results.append({"input": t, "lat": round(lat, 4), "lon": round(lon, 4), "nearest_pin": pin,
                           "district": dist, "state": st, **how})
display(pd.DataFrame(locate_results))

# COMMAND ----------

# DBTITLE 1,6. Write tables and audit
# MAGIC %md
# MAGIC ## 6. Write `pin_centers`, `geo_lookup`, audit

# COMMAND ----------

PC_COLS = ["pincode", "lat", "lon", "district", "state", "n_offices", "n_offices_with_coords", "coord_source", "admin_source"]

def write_pd(pdf, table):
    (spark.createDataFrame(pdf).withColumn("cleaned_run_id", F.lit(RUN_ID))
          .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table))

write_pd(pin_centers[PC_COLS], PIN_CENTERS)
write_pd(geo_lookup[GEO_COLS], GEO_LOOKUP)
log(6, "pin_centers rows written", spark.read.table(PIN_CENTERS).count(), PIN_CENTERS)
log(6, "geo_lookup rows written", spark.read.table(GEO_LOOKUP).count(), GEO_LOOKUP)

(spark.createDataFrame(audit_rows, "run_id string, step string, rule string, rows_affected long, note string")
      .withColumn("logged_at", F.current_timestamp())
      .write.mode("append").option("mergeSchema", "true").saveAsTable(AUDIT))

# COMMAND ----------

# DBTITLE 1,7. Validation
# MAGIC %md
# MAGIC ## 7. Validation — all checks must PASS (the cell raises an error otherwise)

# COMMAND ----------

pc = spark.read.table(PIN_CENTERS).toPandas()
gl = spark.read.table(GEO_LOOKUP).toPandas()
checks = []
def check(name, ok, detail=""):
    checks.append((name, "PASS" if ok else "FAIL", str(detail)))

check("pin_centers: one row per valid source PIN", len(pc) == N_VALID_PINS and pc["pincode"].is_unique, f"{len(pc)} vs {N_VALID_PINS}")
inb = pc.dropna(subset=["lat"])
check("pin_centers: every centre inside India's box",
      inb["lat"].between(LAT_MIN, LAT_MAX).all() and inb["lon"].between(LON_MIN, LON_MAX).all())
check("pin_centers: ≥ 99% of PINs located", pc["lat"].notna().mean() >= 0.99, f"{pc['lat'].notna().mean():.2%}")
bad_st = pc.loc[~pc["state"].isin(CANON_STATES), "pincode"].tolist()
check("pin_centers: every state canonical", not bad_st, bad_st[:10])
check("geo_lookup: only pin / district / town types", set(gl["type"]) <= {"pin", "district", "town"}, sorted(set(gl["type"])))
check("geo_lookup: places are lowercase", (gl["place"] == gl["place"].str.lower()).all())
left_suffix = gl.loc[gl["type"] == "town", "place"].str.contains(OFFICE_SUFFIX.pattern, flags=re.I, regex=True).sum()
check("geo_lookup: no B.O / S.O / H.O / G.P.O suffix left", left_suffix == 0, f"{left_suffix} rows")
check("geo_lookup: every nearest_pin is a real PIN",
      gl["nearest_pin"].dropna().isin(set(pc["pincode"])).all())
check("geo_lookup: no duplicate (place, type, district, state)",
      not gl.duplicated(["place", "type", "district", "state"]).any())
# district centres the raw directory gets badly wrong (filler 15.5934, a shifted division, mislabelled PINs)
KNOWN = {("patan", "Gujarat"): (23.85, 72.13), ("vadodara", "Gujarat"): (22.31, 73.18), ("firozabad", "Uttar Pradesh"): (27.15, 78.40),
         ("azamgarh", "Uttar Pradesh"): (26.07, 83.18), ("raipur", "Chhattisgarh"): (21.25, 81.63), ("mandla", "Madhya Pradesh"): (22.60, 80.37),
         ("palakkad", "Kerala"): (10.78, 76.65), ("mumbai", "Maharashtra"): (19.08, 72.88), ("patna", "Bihar"): (25.59, 85.14)}
gd = gl[gl["type"] == "district"].set_index(["place", "state"])
known_err = {f"{k[0]}": round(float(haversine_km(gd.loc[k, "lat"], gd.loc[k, "lon"], *v)), 1) for k, v in KNOWN.items() if k in gd.index}
check("Known district centres within 60 km of the real city", len(known_err) == len(KNOWN) and max(known_err.values()) <= 60, known_err)
exp = {"patna": ("Bihar", "Patna"), "Muzaffarpur": ("Bihar", "Muzaffarpur"), "800001": ("Bihar", "Patna"),
       "25.59, 85.13": ("Bihar", "Patna"), "patana": ("Bihar", "Patna")}
got = {r["input"]: (r["state"], r["district"]) for r in locate_results}
check("locate(): test inputs land in the expected district", got == exp,
      {k: v for k, v in got.items() if exp.get(k) != v} or "all as expected")

result = pd.DataFrame(checks, columns=["check", "result", "detail"])
display(result)
failed = [x for x in checks if x[1] == "FAIL"]
if failed:
    raise AssertionError(f"{len(failed)} geo validation check(s) FAILED: " + "; ".join(x[0] for x in failed))
print(f"All {len(checks)} geo validation checks PASSED.")
