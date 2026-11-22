# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 04_travel_times_v1 — PIN → facility travel times (straight-line estimate)
# MAGIC
# MAGIC **Inputs:** `workspace.datathon_2026.pin_centers` (from `03_geo`), `workspace.datathon_2026.gold_facility_search` (from `09_Gold`, columns `unique_id, latitude, longitude`). These are the same facilities the app shows: duplicates removed, location-corrected coordinates
# MAGIC
# MAGIC **Output (CREATE OR REPLACE — safe to re-run):** `workspace.datathon_2026.travel_times`, one row per (PIN, facility) pair **within 150 km straight-line**
# MAGIC
# MAGIC | column | meaning |
# MAGIC |---|---|
# MAGIC | `pincode` | PIN centre the trip starts from |
# MAGIC | `facility_id` | facility `unique_id` |
# MAGIC | `km` | great-circle distance (haversine), km |
# MAGIC | `minutes_car` | `km × 1.4 / 40 × 60`: road detour factor 1.4, 40 km/h average |
# MAGIC | `minutes_walk` | `km × 1.2 / 4.5 × 60`: path detour factor 1.2, 4.5 km/h |
# MAGIC | `method` | `straight_line_v1` |
# MAGIC
# MAGIC Pairs come from a BallTree `query_radius` (haversine), not a double loop.

# COMMAND ----------

# DBTITLE 1,0. Setup
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from pyspark.sql import functions as F

TGT = "workspace.datathon_2026"
PIN_CENTERS = f"{TGT}.pin_centers"            # 03_geo
FACILITIES = f"{TGT}.gold_facility_search"    # 09_Gold
TRAVEL = f"{TGT}.travel_times"
AUDIT = f"{TGT}.cleaning_audit"

RADIUS_KM = 150.0
EARTH_KM = 6371.0088
CAR_DETOUR, CAR_KMH = 1.4, 40.0
WALK_DETOUR, WALK_KMH = 1.2, 4.5
METHOD = "straight_line_v1"
WRITE_CHUNK = 2_000_000

RUN_ID = "travel-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
audit_rows = []
def log(step, rule, n, note=""):
    audit_rows.append((RUN_ID, f"travel-{step}", rule, int(n), note))
    print(f"[{step}] {rule}: {n:,} {note}")

for t, src in [(PIN_CENTERS, "03_geo"), (FACILITIES, "09_Gold")]:
    assert spark.catalog.tableExists(t), f"{t} is missing — run {src} first"

# COMMAND ----------

# DBTITLE 1,1. Load PIN centres and facility coordinates
pins = (spark.read.table(PIN_CENTERS).where("lat IS NOT NULL")
        .select("pincode", "lat", "lon").toPandas())
fac = (spark.read.table(FACILITIES).where("latitude IS NOT NULL AND longitude IS NOT NULL")
       .select(F.col("unique_id").alias("facility_id"), F.col("latitude").alias("fac_lat"), F.col("longitude").alias("fac_lon"))
       .dropDuplicates(["facility_id"]).toPandas())
log(1, "PIN centres with coordinates", len(pins))
log(1, "facilities with coordinates", len(fac))

# COMMAND ----------

# DBTITLE 1,2. Pairs within 150 km (BallTree query_radius)
fac_tree = BallTree(np.radians(fac[["fac_lat", "fac_lon"]].to_numpy()), metric="haversine")
idx, dist = fac_tree.query_radius(np.radians(pins[["lat", "lon"]].to_numpy()),
                                  r=RADIUS_KM / EARTH_KM, return_distance=True)
counts = np.fromiter((len(i) for i in idx), dtype=np.int64, count=len(idx))

travel = pd.DataFrame({
    "pincode": np.repeat(pins["pincode"].to_numpy(), counts),
    "facility_id": fac["facility_id"].to_numpy()[np.concatenate(idx)],
    "km": np.round(np.concatenate(dist) * EARTH_KM, 3),
})
travel["minutes_car"] = np.round(travel["km"] * CAR_DETOUR / CAR_KMH * 60, 1)
travel["minutes_walk"] = np.round(travel["km"] * WALK_DETOUR / WALK_KMH * 60, 1)
travel["method"] = METHOD

print(f"travel_times rows: {len(travel):,}")
log(2, "travel_times rows (PIN, facility) within 150 km", len(travel))
log(2, "PINs with no facility within 150 km", int((counts == 0).sum()))
print(f"Facilities per PIN within 150 km — median {np.median(counts):.0f}, max {counts.max():,}")

# COMMAND ----------

# DBTITLE 1,3. Write travel_times and audit
for i, start in enumerate(range(0, len(travel), WRITE_CHUNK)):
    (spark.createDataFrame(travel.iloc[start:start + WRITE_CHUNK])
          .write.mode("overwrite" if i == 0 else "append").option("overwriteSchema", "true")
          .saveAsTable(TRAVEL))
TRAVEL_N = spark.read.table(TRAVEL).count()
log(3, "travel_times rows written", TRAVEL_N, TRAVEL)

(spark.createDataFrame(audit_rows, "run_id string, step string, rule string, rows_affected long, note string")
      .withColumn("logged_at", F.current_timestamp())
      .write.mode("append").option("mergeSchema", "true").saveAsTable(AUDIT))

# COMMAND ----------

# DBTITLE 1,4. TODO — v2 travel times
# MAGIC %md
# MAGIC ## 4. TODO — v2: friction-surface travel times (a teammate is preparing the Malaria Atlas data)
# MAGIC
# MAGIC v1 is a straight-line estimate with a fixed detour factor and speed. It ignores roads, rivers, hills and terrain.
# MAGIC
# MAGIC **v2 will replace `minutes_car` and `minutes_walk`** with least-cost-path times over the **Malaria Atlas Project friction surfaces**: the 2019 motorized and walking-only rasters, in minutes per metre at ~1 km resolution.
# MAGIC 1. Clip both rasters to India's box.
# MAGIC 2. For each facility, run `skimage.graph.MCP_Geometric(friction).find_costs(starts=[facility_cell])` once, limited to the 150 km window. This gives travel minutes to every cell.
# MAGIC 3. Read the cost at each PIN centre's cell and write it into the existing pair rows.
# MAGIC
# MAGIC **Keep the same table and columns** (`pincode, facility_id, km, minutes_car, minutes_walk, method`) and set `method = "friction_mcp_v2"`, so nothing downstream changes. `km` stays the straight-line distance.

# COMMAND ----------

# DBTITLE 1,5. Validation
# MAGIC %md
# MAGIC ## 5. Validation — all checks must PASS (the cell raises an error otherwise)

# COMMAND ----------

tt = spark.read.table(TRAVEL)
checks = []
def check(name, ok, detail=""):
    checks.append((name, "PASS" if ok else "FAIL", str(detail)))

check("Expected columns", tt.columns == ["pincode", "facility_id", "km", "minutes_car", "minutes_walk", "method"], tt.columns)
check("Rows written = rows computed", TRAVEL_N == len(travel), f"{TRAVEL_N} vs {len(travel)}")
car_err = (travel["minutes_car"] - travel["km"] * CAR_DETOUR / CAR_KMH * 60).abs().max()
walk_err = (travel["minutes_walk"] - travel["km"] * WALK_DETOUR / WALK_KMH * 60).abs().max()
check("0 ≤ km ≤ 150", travel["km"].min() >= 0 and travel["km"].max() <= RADIUS_KM + 1e-6,
      f"{travel['km'].min():.3f} – {travel['km'].max():.3f}")
check("minutes_car = km × 1.4 / 40 × 60 (±0.1)", car_err <= 0.11, f"max error {car_err:.3f}")
check("minutes_walk = km × 1.2 / 4.5 × 60 (±0.1)", walk_err <= 0.11, f"max error {walk_err:.3f}")
check("Single method value", set(travel["method"]) == {METHOD})
dups = int(travel.duplicated(["pincode", "facility_id"]).sum())
check("One row per (pincode, facility_id)", dups == 0, f"{dups} duplicates")
# brute-force spot check: 200 random PINs, full haversine to every facility
rng = np.random.default_rng(0)
sample = pins.iloc[rng.choice(len(pins), size=min(200, len(pins)), replace=False)]
la1, lo1 = np.radians(sample["lat"].to_numpy())[:, None], np.radians(sample["lon"].to_numpy())[:, None]
la2, lo2 = np.radians(fac["fac_lat"].to_numpy())[None, :], np.radians(fac["fac_lon"].to_numpy())[None, :]
hv = 2 * EARTH_KM * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
brute = int((hv <= RADIUS_KM).sum())
tree = int(travel["pincode"].isin(set(sample["pincode"])).sum())
check("BallTree pairs match a brute-force check on 200 PINs", abs(brute - tree) <= 2, f"brute {brute:,} vs tree {tree:,}")

result = pd.DataFrame(checks, columns=["check", "result", "detail"])
display(result)
failed = [x for x in checks if x[1] == "FAIL"]
if failed:
    raise AssertionError(f"{len(failed)} travel-time validation check(s) FAILED: " + "; ".join(x[0] for x in failed))
print(f"All {len(checks)} travel-time validation checks PASSED.")
