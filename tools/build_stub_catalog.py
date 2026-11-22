"""Build catalog/capabilities.json from catalog/capabilities_stub_source.json, filling facility_count with
the number of verified facilities that LIST each capability (evidence LISTED or SPECIFIC), using the same
matcher the app uses. Only for the hand-built stub — the full catalog arrives with its own counts.
Usage: DATABRICKS_CONFIG_PROFILE=datathon .venv/bin/python tools/build_stub_catalog.py"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
src = json.loads((ROOT / "catalog/capabilities_stub_source.json").read_text())
for c in src:
    c["facility_count"] = None
(ROOT / "catalog/capabilities.json").write_text("[\n" + ",\n".join(" " + json.dumps(c, ensure_ascii=False) for c in src) + "\n]\n")
import capabilities as caps
caps.catalog.cache_clear()
from engine import find_care as fc
t = time.time(); fc.facilities(); print(f"matched all facilities in {time.time() - t:.1f}s")
for c in src:
    lv = fc.levels_for(c["capability_id"])
    c["facility_count"] = int(lv["level"].isin(["LISTED", "SPECIFIC"]).sum())   # facilities that LIST it
    print(f"  {c['capability_id']:20} {c['facility_count']:5}  " + "  ".join(f"{k}={v}" for k, v in lv["level"].value_counts().items() if k != "NONE"))
(ROOT / "catalog/capabilities.json").write_text("[\n" + ",\n".join(" " + json.dumps(c, ensure_ascii=False) for c in src) + "\n]\n")
print("wrote catalog/capabilities.json")
