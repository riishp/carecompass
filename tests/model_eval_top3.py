"""Accuracy test 2 — quality of the TOP 3 each version recommends (offline, no Databricks).

Runs N random districts × 10 services through all four versions:
  Current   the app's find_care() ranking code (engine score_candidates + pick_top3), run offline on local data.
            Offline limits: the 7 checked services use the generic matcher (the live app uses 04's hand-tuned
            evidence for them), and the search point is the district centre (no place-name resolution).
  Model 1 / Model 2 / Combined   versions.rank() on the same point and radius.
For each version: searches with any result, picks the government directory can check (matched hospitals with a
service list), share of those the directory CONFIRMS, and travel distance of the picks.

Usage: .venv/bin/python tests/model_eval_top3.py <gold_facility_search_services.csv> <hospital_directory.csv>
       <india_post_pincode_directory.csv> [n_districts=40]
Writes tests/results/model_eval_test2.csv
"""
import math
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
gold_csv, dir_csv, pin_csv = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 40
SERVICES = ["xray", "ultrasound", "ct_scan", "dialysis", "c_section", "nicu", "icu", "emergency_247", "cath_lab", "ent"]

# ---- load the engine's pure functions without a Databricks session
fake_spark = types.ModuleType("engine._spark")
fake_spark.spark = None
sys.modules["engine._spark"] = fake_spark
import engine.recommend_engine as eng  # noqa: E402  (only pure pandas functions are used below)
import capabilities as caps  # noqa: E402
from engine import find_care as fc  # noqa: E402
import versions  # noqa: E402

CHECKED_CATALOG = {"dialysis": dict(urgency="recurring", max_km=100), "c_section": dict(urgency="emergency", max_km=60),
                   "nicu": dict(urgency="emergency", max_km=100), "cardiac_surgery": dict(urgency="planned", max_km=300),
                   "angioplasty": dict(urgency="emergency", max_km=150), "mri": dict(urgency="planned", max_km=200),
                   "burn_care": dict(urgency="emergency", max_km=150)}
eng._catalog = lambda: CHECKED_CATALOG

# ---- facilities in the shape find_care expects, from the verified Gold CSV
g = pd.read_csv(gold_csv, dtype=str, keep_default_na=False)
sp = lambda x: [i for i in str(x).split("|") if i.strip()]
num = lambda s: pd.to_numeric(s.replace("", np.nan), errors="coerce")
fac = pd.DataFrame({
    "unique_id": g.unique_id, "name": g.name, "facility_type": g.facility_type, "operator_type": g.operator_type,
    "final_lat": num(g.latitude), "final_lon": num(g.longitude), "location_error_km": num(g.location_error_km),
    "uncertainty_tier": g.location_confidence.map({"high": "low", "medium": "medium"}).fillna("high"),
    "location_evidence": g.location_note, "location_warnings": g.location_warnings.map(sp),
    "possible_merged_record": g.location_warnings.str.contains("possible_merged_record"),
    "phone": g.phone, "phone_numbers": g.phone_numbers.map(sp), "website": g.website,
    "number_doctors": num(g.number_doctors), "capacity_beds": num(g.capacity_beds), "specialty_labels": g.specialty_labels.map(sp),
    "equipment": g.equipment.map(sp)})
fields = [{"specialties": sp(r.specialty_labels) + sp(r.specialties_other), "procedures": sp(r.procedures),
           "equipment": sp(r.equipment), "capabilities": sp(r.capabilities),
           "description": [r.description] if r.description else []} for r in g.itertuples()]
fac["cap_mentions"] = [caps.find_mentions(f) for f in fields]
dm = [caps.specialty_domains(sp(x)) for x in g.specialty_labels]
fac["main_domain"], fac["spec_domains"] = [d[0] for d in dm], [d[1] for d in dm]
fac["has_equipment"] = fac.equipment.map(bool)
fac = fac[fac.final_lat.notna()].reset_index(drop=True)
fc._CACHE["fac"] = fac
fc.CHECKED = {}                                   # offline: generic matcher for every service (see docstring)

# ---- answer key (Model 2's labels: directory-matched hospitals with a service list)
m2 = {"__name__": "m2", "display": lambda *a, **k: None}
for c in (ROOT / "notebooks" / "model2_facility_capability.py").read_text().split("# COMMAND ----------"):
    code = "\n".join(l for l in c.splitlines() if not l.startswith("# MAGIC"))
    if code.strip() and not any(k in code for k in ("spark.table", "result = run(", "fac = pd.read_csv(GOLD", "saveAsTable")):
        exec(compile(code, "model2", "exec"), m2)
key = m2["labels"](g, pd.read_csv(dir_csv, dtype=str, keep_default_na=False, low_memory=False))
truth = {(u, s): y for u, s, y in zip(key.unique_id, key.service, key.y)}

# ---- random districts (centre = median of India Post office coordinates)
p = pd.read_csv(pin_csv, dtype=str)
p["lat"], p["lon"] = pd.to_numeric(p.latitude, errors="coerce"), pd.to_numeric(p.longitude, errors="coerce")
p = p[p.lat.between(6, 37.6) & p.lon.between(68, 97.5)]
d = p.groupby(["district", "statename"])[["lat", "lon"]].median().reset_index()
districts = d.sample(N, random_state=7)


def place(lat, lon, label):
    return {"status": "resolved", "method": "coordinates", "lat": lat, "lon": lon, "radius_km": 0.0, "label": label,
            "district": label, "state": None, "options": [], "note": f"District centre of {label}."}


rows = []
for dd in districts.itertuples():
    eng.resolve_location = lambda text, _p=place(dd.lat, dd.lon, f"{dd.district}, {dd.statename}"): _p
    for cap in SERVICES:
        base = fc.find_care([cap], "district centre", route_checked=False)
        outs = {"Current (Find care)": base}
        for ver, name in [("model1", "Model 1"), ("model2", "Model 2"), ("combined", "Combined")]:
            outs[name] = versions.rank(ver, base, fc)
        svc = versions.MODEL2_SERVICE.get(cap, cap)
        for name, o in outs.items():
            picks = o.get("results") or []
            for r in picks:
                t = truth.get((r["unique_id"], svc))
                rows.append({"version": name, "district": dd.district, "service": cap, "unique_id": r["unique_id"],
                             "km": r["distance_km"], "verifiable": t is not None, "confirmed": t == 1})
            rows.append({"version": name, "district": dd.district, "service": cap, "unique_id": None,
                         "km": np.nan, "verifiable": False, "confirmed": False, "_search": True, "_n": len(picks)})
res = pd.DataFrame(rows)
(ROOT / "tests" / "results").mkdir(parents=True, exist_ok=True)
res.to_csv(ROOT / "tests" / "results" / "model_eval_test2.csv", index=False)
srch = res[res.get("_search", False) == True]  # noqa: E712
picks = res[res.unique_id.notna()]
summary = pd.DataFrame({
    "searches with ≥1 pick": srch.groupby("version")._n.apply(lambda s: f"{(s > 0).mean():.0%}"),
    "picks": picks.groupby("version").size(),
    "checkable by directory": picks.groupby("version").verifiable.apply(lambda s: f"{s.mean():.0%}"),
    "directory confirms (of checkable)": picks[picks.verifiable].groupby("version").confirmed.apply(lambda s: f"{s.mean():.0%}"),
    "confirmed n": picks[picks.verifiable].groupby("version").confirmed.sum(),
    "median km": picks.groupby("version").km.median().round(0),
    "share > 50 km": picks.groupby("version").km.apply(lambda s: f"{(s > 50).mean():.0%}"),
}).reindex(["Current (Find care)", "Model 1", "Model 2", "Combined"])
pd.set_option("display.width", 220)
print(f"{len(districts)} districts × {len(SERVICES)} services = {len(districts) * len(SERVICES)} searches per version\n")
print(summary.to_string())
