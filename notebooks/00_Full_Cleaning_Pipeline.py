# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 00 — Full Cleaning Pipeline (one button)
# MAGIC
# MAGIC Runs every cleaning notebook **in order**, from the raw shared tables to Gold. Each notebook ends with validation checks and raises an error if any fail, so **the pipeline stops at the first broken step** and nothing downstream is built on bad data.
# MAGIC
# MAGIC | # | notebook | owner | writes (all in `workspace.datathon_2026`) |
# MAGIC |---|---|---|---|
# MAGIC | 1 | `Data Cleaning: Raw to Bronze` | Bronze teammate | `facilities_bronze`, `facilities_quarantine`, `pincode_reference`, `cleaning_audit` |
# MAGIC | 2 | `03_location_cleaning_and_uncertainty` | Location teammate | `pincode_clean`, `pin_lookup`, `place_lookup`, `facilities_clean`, `facilities_located` |
# MAGIC | 3 | `03_geo` | — | `pin_centers`, `geo_lookup` |
# MAGIC | 4 | `06_NFHS_Cleaning` | — | `nfhs_clean`, `nfhs_value_flags`, `nfhs_district_crosswalk`, `nfhs_quarantine` |
# MAGIC | 5 | `07_Specialty_Standardization` | — | `specialty_map` |
# MAGIC | 6 | `08_Silver_Facilities` | — | `facilities_silver` (uses `03_geo` for the location vote) |
# MAGIC | 7 | `09_Gold` | — | `gold_facility_search`, `gold_district_summary` |
# MAGIC | 8 | `04_travel_times_v1` | — | `travel_times` (uses Gold facilities) |
# MAGIC
# MAGIC The notebooks are **called, not copied** — a fix in any of them is picked up automatically on the next run.
# MAGIC Source tables are never modified. Typical run time: ~15–20 min on serverless.
# MAGIC
# MAGIC To re-run only part of the pipeline, set the `start_at` widget (e.g. `6` to rebuild Silver, Gold and travel times only).

# COMMAND ----------

# DBTITLE 1,Run the pipeline
import time
import pandas as pd

FOLDER = "/Workspace/Shared/Datathon 2026"
STEPS = [
    (1, "Data Cleaning: Raw to Bronze"),
    (2, "03_location_cleaning_and_uncertainty"),
    (3, "03_geo"),
    (4, "06_NFHS_Cleaning"),
    (5, "07_Specialty_Standardization"),
    (6, "08_Silver_Facilities"),
    (7, "09_Gold"),
    (8, "04_travel_times_v1"),
]
TIMEOUT_S = 3600

dbutils.widgets.dropdown("start_at", "1", [str(n) for n, _ in STEPS], "Start at step")
start_at = int(dbutils.widgets.get("start_at"))

log = []
for n, name in STEPS:
    if n < start_at:
        log.append({"step": n, "notebook": name, "status": "skipped", "minutes": 0.0, "error": ""})
        continue
    t0 = time.time()
    print(f"▶ {n}. {name} …")
    try:
        dbutils.notebook.run(f"{FOLDER}/{name}", TIMEOUT_S)
        log.append({"step": n, "notebook": name, "status": "PASS", "minutes": round((time.time() - t0) / 60, 1), "error": ""})
        print(f"  ✓ done in {log[-1]['minutes']} min")
    except Exception as e:
        log.append({"step": n, "notebook": name, "status": "FAIL", "minutes": round((time.time() - t0) / 60, 1),
                    "error": str(e)[:500]})
        display(pd.DataFrame(log))
        raise RuntimeError(f"Pipeline stopped at step {n} ({name}). Open that notebook's run to see which check failed.\n{e}") from e

display(pd.DataFrame(log))
print("Pipeline complete — all steps passed their validation checks.")

# COMMAND ----------

# DBTITLE 1,Final row counts
TGT = "workspace.datathon_2026"
counts = [(t, spark.read.table(f"{TGT}.{t}").count()) for t in
          ["facilities_bronze", "facilities_quarantine", "facilities_located", "nfhs_clean", "specialty_map",
           "pin_centers", "geo_lookup", "facilities_silver", "gold_facility_search", "gold_district_summary", "travel_times"]]
display(pd.DataFrame(counts, columns=["table", "rows"]))
