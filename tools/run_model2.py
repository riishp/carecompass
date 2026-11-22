"""Run the team's 'Model 2 - Facility Capability' notebook LOCALLY (pandas/sklearn, no Databricks compute), on top of
Model 1's output, and store the result for the app's Model 2 / Combined tabs.

Usage: .venv/bin/python tools/run_model2.py notebooks/model2_facility_capability.py <gold_facility_search_services.csv> <hospital_directory.csv>
The notebook's code runs unchanged; Model 1's results come from catalog/model1_levels.csv (tools/run_model1.py).
Writes catalog/model2_capability.parquet."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb, gold_csv, dir_csv = sys.argv[1:4]
SKIP = ("spark.table(MODEL1_TABLE)", "result = run(GOLD, DIRECTORY, MODEL1_CSV)", "fac = pd.read_csv(GOLD",
        "saveAsTable")
ns = {"__name__": "model2_local", "display": lambda *a, **k: None}
for c in Path(nb).read_text().split("# COMMAND ----------"):
    code = "\n".join(l for l in c.splitlines() if not l.startswith("# MAGIC"))
    if code.strip() and not any(k in code for k in SKIP):
        exec(compile(code, "model2_notebook", "exec"), ns)
res = ns["run"](gold_csv, dir_csv, str(ROOT / "catalog" / "model1_levels.csv"))
res.to_parquet(ROOT / "catalog" / "model2_capability.parquet", index=False)
print(f"\nwrote catalog/model2_capability.parquet: {len(res):,} rows")
