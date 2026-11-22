"""Model 1 / Model 2 / Combined versions of a search — separate ranking paths, for side-by-side testing.

The current algorithm (engine/find_care.py) is NOT changed. Each version re-ranks the facilities for the SAME search
(same capabilities, same resolved place, same radius as the Find care tab), using the precomputed model outputs:
  catalog/model1_levels.csv          tools/run_model1.py  (team notebook 'Model 1 - Evidence Levels')
  catalog/model2_capability.parquet  tools/run_model2.py  (team notebook 'Model 2 - Facility Capability')
Results come back in the same shape as find_care(), so the app shows them with the same call plan / slip / cards.
"""
from pathlib import Path

import pandas as pd

import capabilities as caps
import ui_results as ui

ROOT = Path(__file__).resolve().parent
# app capability_id → the models' service names (the extended Model 1 copy uses Model 2's names)
MODEL2_SERVICE = {"xray": "xray", "ultrasound": "ultrasound", "ct_scan": "ct_scan", "mri": "mri", "ecg": "cardiac_diagnostics",
                  "echo": "cardiac_diagnostics", "fracture_treatment": "orthopedic_surgery", "ortho_surgery": "orthopedic_surgery",
                  "emergency_247": "emergency_24x7", "icu": "icu", "nicu": "nicu", "blood_bank": "blood_bank",
                  "normal_delivery": "maternity", "c_section": "c_section", "dialysis": "dialysis",
                  "cath_lab": "cath_lab_angioplasty", "cardiac_surgery": "cardiac_surgery", "cataract_surgery": "eye_care",
                  "chemotherapy": "chemotherapy", "radiotherapy": "radiotherapy", "burn_care": "burn_care",
                  "ambulance": "ambulance", "physiotherapy": "physiotherapy_rehab", "ent": "ent", "dermatology": "dermatology",
                  "mental_health": "psychiatry", "lab_tests": "lab_pathology", "paediatrics": "pediatrics", "dental_xray": "dental"}
MODEL1_SERVICE = dict(MODEL2_SERVICE, anti_venom="anti_venom")   # Model 1 (all app services): every Find care service
M1_RANK = {"SPECIFIC": 2, "LISTED": 1, "PASSING": 0}
M2_RANK = {"LIKELY": 3, "POSSIBLE": 2, "UNKNOWN": 1, "UNLIKELY": 0}
LABEL_RANK = {"Strong match": 5, "Good match": 4, "Call to confirm": 3, "Conflict: call first": 2, "Doubtful": 1,
              "Not shown": 0, "Not mentioned": -1}
VERSION = {
    "model1": dict(title="Model 1", rule="Model 1 evidence level (Specific > Listed), then nearest, then directory-confirm %."),
    "model2": dict(title="Model 2", rule="Only facilities that mention the service. Model 2 capability (Likely > Possible), "
                                        "then nearest, then capability %."),
    "combined": dict(title="Combined", rule="Combined finder label (Strong match > Good match > Call to confirm), then "
                                            "nearest. 'Conflict', 'Doubtful' and 'Not shown' are never top picks."),
}
_D = {}


def available():
    return {"model1": (ROOT / "catalog" / "model1_levels.csv").exists(),
            "model2": (ROOT / "catalog" / "model2_capability.parquet").exists(),
            "combined": (ROOT / "catalog" / "model2_capability.parquet").exists()}


def _m1():
    if "m1" not in _D:
        _D["m1"] = pd.read_csv(ROOT / "catalog" / "model1_levels.csv", dtype={"unique_id": str})
    return _D["m1"]


def _m2():
    if "m2" not in _D:
        _D["m2"] = pd.read_parquet(ROOT / "catalog" / "model2_capability.parquet")
    return _D["m2"]


def _shell(base, version, cap_ids, notes):
    v = VERSION[version]
    names = " + ".join(caps.catalog()["by_id"][c]["name"] for c in cap_ids)
    how = dict(base.get("how_we_decided") or {})
    how["matching"] = f"{v['title']}: {v['rule']} A facility must cover every selected service it scores (AND)."
    how["widening"] = []
    return {"status": "no_results", "service_resolution": {"status": "matched", "display_name": names, "note":
            f"Searching for {names} — ranked by {v['title']}."}, "location_resolution": base["location_resolution"],
            "results": [], "fallback_results": [], "search_notes": notes, "how_we_decided": how,
            "capability_ids": cap_ids, "banner": base.get("banner"), "path": version}


def _result(row, km, role, level, label, colour, levels, note=None):
    return {"unique_id": row["unique_id"], "name": row["name"], "role": role, "distance_km": float(km),
            "lat": float(row["final_lat"]), "lon": float(row["final_lon"]), "overall_level": level,
            "display_label": label, "display_colour": colour, "capability_levels": levels,
            "phone": row.get("phone"), "website": row.get("website"), "why_location": row.get("location_evidence"),
            "warnings": [note] if note else [], "components": {}}


def rank(version, base, fc):
    """Re-rank Find care's search (`base` = find_care() output) with one model. Returns find_care()-shaped output."""
    if base.get("status") in ("needs_location_clarification", "needs_service_clarification", "unknown_location",
                              "unknown_service"):
        return base
    loc = base["location_resolution"]
    cap_ids = base.get("capability_ids") or []
    by_id = caps.catalog()["by_id"]
    radius = float((base.get("how_we_decided") or {}).get("final_search_radius_km") or 100.0)
    fac = fc.facilities()
    from engine import recommend_engine as eng
    km = pd.Series(eng._hav_np(loc["lat"], loc["lon"], fac["final_lat"].to_numpy(float), fac["final_lon"].to_numpy(float)),
                   index=fac["unique_id"].values)
    near = fac[fac["unique_id"].map(km).le(radius)].set_index("unique_id", drop=False)

    smap = MODEL1_SERVICE if version == "model1" else MODEL2_SERVICE
    if version == "combined":
        smap = MODEL2_SERVICE                     # needs both models; anti-venom has no Model 2 score
    covered = [c for c in cap_ids if c in smap]
    skipped = [by_id[c]["name"] for c in cap_ids if c not in smap]
    notes = [f"{VERSION[version]['title']} does not score {', '.join(skipped)} — ignored here."] if skipped else []
    out = _shell(base, version, cap_ids, notes)
    if not covered:
        out["search_notes"] = [f"{VERSION[version]['title']} does not score " + " + ".join(by_id[c]["name"] for c in cap_ids)
                               + " yet, so it has no ranking for this search."]
        return out

    rows = []
    if version == "model1":
        m = _m1()
        sub = m[m["service"].isin({smap[c] for c in covered}) & m["unique_id"].isin(near.index)]
        for uid, g in sub.groupby("unique_id"):
            got = {s: r for s, r in zip(g["service"], g.to_dict("records"))}
            if not all(smap[c] in got for c in covered):
                continue
            lv = {c: got[smap[c]] for c in covered}
            worst = min(lv.values(), key=lambda r: M1_RANK[r["evidence_level"]])["evidence_level"]
            pct = pd.to_numeric(pd.Series([r["directory_confirm_pct"] for r in lv.values()]), errors="coerce").mean()
            levels = [dict(capability_id=c, name=by_id[c]["name"], level=lv[c]["evidence_level"], quote=str(lv[c]["quote"]),
                           field=f"Model 1 · {lv[c]['quote_field']}", penalties=[]) for c in covered]
            label = ui.EVIDENCE_LABEL[worst] + (f" · directory confirm {pct:.0f}%" if pd.notna(pct) else "")
            rows.append((M1_RANK[worst], round(km[uid]), -(pct if pd.notna(pct) else 0), uid, worst, label,
                         ui.EVIDENCE_COLOUR[worst], levels, worst != "PASSING"))
    else:
        m = _m2()
        svcs = {smap[c] for c in covered}
        sub = m[m["service"].isin(svcs) & m["unique_id"].isin(near.index)]
        if version == "model2":                                    # decision: only facilities that mention the service
            mention = None
            for c in covered:
                ok = set(fc.levels_for(c).query("level != 'NONE'")["unique_id"])
                mention = ok if mention is None else mention & ok
            sub = sub[sub["unique_id"].isin(mention or set())]
        for uid, g in sub.groupby("unique_id"):
            got = {s: r for s, r in zip(g["service"], g.to_dict("records"))}
            if not all(smap[c] in got for c in covered):
                continue
            recs = {c: got[smap[c]] for c in covered}
            if version == "model2":
                assessed = {c: r for c, r in recs.items() if r["capability_band"] != "NOT_ASSESSED"}
                if not assessed:
                    continue
                worst = min((r["capability_band"] for r in assessed.values()), key=M2_RANK.get)
                pct = pd.to_numeric(pd.Series([r["capability_pct"] for r in assessed.values()]), errors="coerce").mean()
                lvl = {"LIKELY": "SPECIFIC", "POSSIBLE": "LISTED"}.get(worst, "PASSING")
                levels = [dict(capability_id=c, name=by_id[c]["name"], level=lvl,
                               quote=f"Capability {r['capability_band'].lower()}"
                                     + (f" ({r['capability_pct']:.0f}%)" if pd.notna(r["capability_pct"]) else "")
                                     + (f" — {r['drivers']}" if r["drivers"] else ""),
                               field="Model 2", penalties=[]) for c, r in assessed.items()]
                label = {"LIKELY": "Likely capable", "POSSIBLE": "Possibly capable", "UNKNOWN": "Capability unknown",
                         "UNLIKELY": "Unlikely capable"}[worst] + (f" · {pct:.0f}%" if pd.notna(pct) else "")
                colour = {"LIKELY": "#1B7F4C", "POSSIBLE": "#B7791F"}.get(worst, "#6B7280")
                rows.append((M2_RANK[worst], round(km[uid]), -(pct if pd.notna(pct) else 0), uid, lvl, label, colour,
                             levels, worst in ("LIKELY", "POSSIBLE")))
            else:                                                  # combined
                worst = min((r.get("finder_label") or "Not mentioned" for r in recs.values()), key=LABEL_RANK.get)
                pct = pd.to_numeric(pd.Series([r["capability_pct"] for r in recs.values()]), errors="coerce").mean()
                m1 = _m1()
                q = m1[(m1["unique_id"] == uid) & m1["service"].isin(svcs)]
                quote = str(q.iloc[0]["quote"]) if len(q) else ""
                lvl = "SPECIFIC" if LABEL_RANK[worst] >= 4 else "LISTED" if LABEL_RANK[worst] == 3 else "PASSING"
                levels = [dict(capability_id=c, name=by_id[c]["name"], level=lvl,
                               quote=f"{r.get('finder_label')} — Model 1 {r.get('evidence_level') or 'no evidence'}, "
                                     f"Model 2 {r['capability_band'].lower()}" + (f": “{quote[:140]}”" if quote else ""),
                               field="Combined", penalties=[]) for c, r in recs.items()]
                colour = {5: "#1B7F4C", 4: "#1B7F4C", 3: "#B7791F"}.get(LABEL_RANK[worst], "#6B7280")
                note = "Conflict: the record claims it in detail, but the facility's size/type makes it unlikely — call first." \
                    if worst == "Conflict: call first" else None
                rows.append((LABEL_RANK[worst], round(km[uid]), -(pct if pd.notna(pct) else 0), uid, lvl, worst, colour,
                             levels, LABEL_RANK[worst] >= 3, note))

    rows.sort(key=lambda t: (-t[0], t[1], t[2]))
    role = f"{VERSION[version]['title']} pick"
    top = [t for t in rows if t[8]][:3]
    rest = [t for t in rows if not t[8]][:3]
    mk = lambda t, rl: _result(near.loc[t[3]], km[t[3]], rl, t[4], t[5], t[6], t[7], t[9] if len(t) > 9 else None)
    out["results"] = [mk(t, role) for t in top]
    out["fallback_results"] = [mk(t, "Nothing better nearby") for t in rest] if len(top) < 3 else []
    out["status"] = "ok" if top else ("only_weak_results" if rest else "no_results")
    if not rows:
        out["search_notes"].append(f"{VERSION[version]['title']} has no facility within {radius:.0f} km for this.")
    return out
