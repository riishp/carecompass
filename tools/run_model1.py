"""Run the team's 'Model 1 - Evidence Levels' notebook LOCALLY (pure pandas/sklearn, no Databricks compute)
and store its output for the app's 'From Model 1' card.

Usage: .venv/bin/python tools/run_model1.py <notebook.py> <gold_facility_search_services.csv> <hospital_directory.csv>
The notebook's code runs unchanged; only its two input paths are pointed at local copies.
Writes catalog/model1_levels.csv and catalog/model1_services.json (per-service checklist questions)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb, fac_csv, dir_csv = sys.argv[1:4]
cells = Path(nb).read_text().split("# COMMAND ----------")
ns = {"__name__": "model1_local", "display": lambda *a, **k: None}
for c in cells:
    code = "\n".join(l for l in c.splitlines() if not l.startswith("# MAGIC"))
    if not code.strip() or "result = run(FACILITIES, DIRECTORY)" in code or "fac = pd.read_csv(FACILITIES" in code:
        continue                                      # skip the notebook's own run/display cells
    exec(compile(code, "model1_notebook", "exec"), ns)
ns["FACILITIES"], ns["DIRECTORY"] = fac_csv, dir_csv
result = ns["run"](fac_csv, dir_csv)
keep = ["unique_id", "service", "evidence_level", "directory_confirm_pct", "quote", "quote_field", "reasons",
        "caps_applied", "rules_version", "pct_model_version"]
out = result[[c for c in keep if c in result.columns]]
(ROOT / "catalog").mkdir(exist_ok=True)
out.to_csv(ROOT / "catalog" / "model1_levels.csv", index=False)
(ROOT / "catalog" / "model1_services.json").write_text(json.dumps(
    {s: {"checklist": r.get("checklist", "")} for s, r in ns["SERVICE_TABLE"].items()}, indent=1))
print(f"model1_levels.csv: {len(out):,} facility x service rows, {out.unique_id.nunique():,} facilities")
print(out.evidence_level.value_counts().to_dict())
