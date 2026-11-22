"""find_care(capability_ids, location_text) — capability-based recommendations.

Matches facilities against ANY capability in catalog/capabilities.json (not just the 7 checked services),
then ranks with the same machinery as 06_recommend: resolve_location, score_candidates, pick_top3.

Rules
- Evidence level per facility × capability: SPECIFIC > LISTED > PASSING > NONE (capabilities.level_for),
  with penalties for 'has an equipment list without it' and 'mainly a different kind of facility'.
- The 7 checked services keep their hand-tuned rules: their level comes from facility_service_evidence
  (notebook 04 tiers incl. contradiction caps), and their max_km / urgency from service_catalog.
  A search for exactly ONE checked service goes through the original recommend() unchanged.
- Multi-select = AND: a facility's level is the LOWEST across the selected capabilities.
- Top picks need LISTED or better for every capability. PASSING / partial matches are shown only under
  "nothing better nearby", with a call-ahead checklist.
- Defaults for everything else: max_km 100, urgency planned unless the condition says emergency.
"""
import math

import numpy as np
import pandas as pd

import capabilities as caps
from engine import recommend_engine as eng

# capability_id → the 7 checked service_ids of 06_recommend (hand-tuned evidence, max_km, urgency)
CHECKED = {"dialysis": "dialysis", "c_section": "c_section", "nicu": "nicu", "cardiac_surgery": "cardiac_surgery",
           "cath_lab": "angioplasty", "mri": "mri", "burn_care": "burn_care"}
CHECKED_QUERY = {"dialysis": "dialysis", "c_section": "c-section", "nicu": "nicu", "cardiac_surgery": "heart surgery",
                 "angioplasty": "angioplasty", "mri": "mri", "burn_care": "burn care"}
TIER_TO_LEVEL = {"strong": "SPECIFIC", "moderate": "LISTED", "weak": "PASSING"}
LEVEL_TO_TIER = {"SPECIFIC": "strong", "LISTED": "moderate", "PASSING": "weak", "NONE": "weak"}
LEVEL_SCORE = {"SPECIFIC": 1.0, "LISTED": 0.6, "PASSING": 0.2, "NONE": 0.0}
DEFAULT_MAX_KM = 100.0
DOMAIN_MISMATCH_PENALTY = 0.08
URGENCY_ORDER = ["planned", "recurring", "emergency"]
_CACHE = {}


# ───────────────────────────── data ─────────────────────────────
def _items(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    try:
        return [str(x) for x in list(v) if x is not None and str(x).strip()]
    except TypeError:
        return []


def facilities():
    """Verified Gold facilities + their capability mentions (computed once per process)."""
    if "fac" not in _CACHE:
        fac = eng._base_facilities().toPandas()
        txt = (eng.spark.table(eng.FACILITY_TABLE)
               .select("unique_id", "facility_type", "specialty_labels", "specialties_other", "procedures",
                       "equipment", "capabilities", "description",
                       # display-only columns (trust chips, slip address, government fallback)
                       "operator_type", "city", "district", "state", "pincode", "directory_match",
                       "location_confidence").toPandas())
        bp = (eng.spark.table(f"{eng.SCHEMA}.service_fields_parsed")
              .select("unique_id", "possible_boilerplate").toPandas())
        fac = fac.drop(columns=["facility_type"]).merge(txt, on="unique_id").merge(bp, on="unique_id", how="left")
        fac["possible_boilerplate"] = fac["possible_boilerplate"].fillna(False).astype(bool)
        mentions, mains, doms = [], [], []
        for r in fac.itertuples(index=False):
            fields = {"specialties": _items(r.specialty_labels) + _items(r.specialties_other),
                      "procedures": _items(r.procedures), "equipment": _items(r.equipment),
                      "capabilities": _items(r.capabilities), "description": _items(r.description)}
            mentions.append(caps.find_mentions(fields))
            main, ds = caps.specialty_domains(_items(r.specialty_labels))
            mains.append(main)
            doms.append(ds)
        fac["cap_mentions"], fac["main_domain"], fac["spec_domains"] = mentions, mains, doms
        fac["has_equipment"] = fac["equipment"].map(lambda v: len(_items(v)) > 0)
        _CACHE["fac"] = fac
    return _CACHE["fac"]


def checked_evidence():
    if "ev" not in _CACHE:
        _CACHE["ev"] = (eng.spark.table(f"{eng.SCHEMA}.facility_service_evidence")
                        .select("unique_id", "service_id", "evidence_tier", "evidence_text", "contradiction")
                        .toPandas())
    return _CACHE["ev"]


def levels_for(capability_id):
    """DataFrame[unique_id, level, quote, field, penalties, domain_mismatch] for one capability."""
    key = ("lv", capability_id)
    if key in _CACHE:
        return _CACHE[key]
    cap = caps.catalog()["by_id"][capability_id]
    fac = facilities()
    if capability_id in CHECKED:
        ev = checked_evidence()
        ev = ev[ev["service_id"] == CHECKED[capability_id]].set_index("unique_id")
        rows = []
        for r in fac.itertuples(index=False):
            if r.unique_id in ev.index:
                e = ev.loc[r.unique_id]
                lv = TIER_TO_LEVEL.get(e["evidence_tier"], "NONE")
                rows.append((r.unique_id, lv, e["evidence_text"], "checked service evidence",
                             ["contradiction (capped)"] if bool(e["contradiction"]) else [], False))
            else:
                rows.append((r.unique_id, "NONE", None, None, [], False))
    else:
        rows = []
        for r in fac.itertuples(index=False):
            lv = caps.level_for(cap, r.cap_mentions.get(capability_id, []), {"equipment": r.has_equipment},
                                r.main_domain, r.spec_domains)
            rows.append((r.unique_id, lv["level"], lv["quote"], lv["field"], lv["penalties"], lv.get("domain_mismatch", False)))
    out = pd.DataFrame(rows, columns=["unique_id", "level", "quote", "field", "penalties", "domain_mismatch"])
    _CACHE[key] = out
    return out


def facility_count(capability_id):
    return int((levels_for(capability_id)["level"] != "NONE").sum())


# ───────────────────────────── service settings ─────────────────────────────
def _settings(capability_ids, urgency_hint=None):
    cat = eng._catalog()
    max_km, urg = [], [urgency_hint] if urgency_hint else []
    for cid in capability_ids:
        sid = CHECKED.get(cid)
        if sid and sid in cat:
            max_km.append(float(cat[sid]["max_km"]))
            urg.append(cat[sid]["urgency"])
        else:
            max_km.append(DEFAULT_MAX_KM)
    urgency = max(urg, key=URGENCY_ORDER.index) if urg else "planned"
    return max(max_km), urgency


# ───────────────────────────── explanations ─────────────────────────────
def _quote(q, n=110):
    q = " ".join(str(q or "").split())
    return q if len(q) <= n else q[: n - 1] + "…"


def _explain(pick, cap_ids, weights):
    r, role = pick["row"], pick["role"]
    names = caps.catalog()["by_id"]
    levels = []
    for cid in cap_ids:
        lv = r[f"lv_{cid}"]
        levels.append({"capability_id": cid, "name": names[cid]["name"], "level": lv["level"],
                       "quote": _quote(lv["quote"]) if lv["quote"] else None, "field": lv["field"],
                       "penalties": lv["penalties"]})
    why = "; ".join(
        f"{x['name']}: {caps.LEVEL_TEXT[x['level']]}" + (f" — '{x['quote']}' ({x['field']})" if x["quote"] else "")
        for x in levels)
    checklist = []
    for x in levels:
        if x["level"] in ("PASSING", "NONE"):
            checklist.append(f"Do you offer {x['name']}? (it is only {caps.LEVEL_TEXT[x['level']]})")
        elif x["level"] == "LISTED" and names[x["capability_id"]]["type"] == "equipment":
            checklist.append(f"Is the {x['name']} working and available today?")
    warnings = list(eng._arr(r.get("location_warnings")))
    for x in levels:
        for p in x["penalties"]:
            warnings.append(f"{x['name']}: {p}.")
    if bool(r.get("possible_merged_record")):
        warnings.append("This record may combine more than one facility — confirm the address when you call.")
    comps = {k: round(float(r[f"{k}_component"]), 2) for k in ("evidence", "access", "readiness", "trust")}
    top = max(comps, key=lambda k: weights[k] * comps[k])
    phone = r.get("phone") or (eng._arr(r.get("phone_numbers")) or [None])[0]
    return {
        "unique_id": r["unique_id"], "name": r["name"], "role": role,
        "distance_km": round(float(r["distance_km"]), 1), "final_score": round(float(r["final_score"]), 2),
        "evidence_tier": LEVEL_TO_TIER[r["overall_level"]], "overall_level": r["overall_level"],
        "lat": float(r["final_lat"]), "lon": float(r["final_lon"]), "phone": phone, "website": r.get("website"),
        "possible_merged_record": bool(r.get("possible_merged_record")), "components": comps,
        "headline": f"{r['name']} — {float(r['distance_km']):.0f} km — {role}",
        "why_service": why, "why_location": r.get("location_evidence"),
        "why_ranked": f"Ranked mainly on {eng.COMPONENT_PHRASES[top]}.",
        "contact": (f"Phone: {phone}. " if phone else "No phone number listed. ") + "Call ahead to confirm.",
        "warnings": warnings, "checklist": checklist, "capability_levels": levels,
        "caveat": "All service information is self-reported by the facility.",
    }


# ───────────────────────────── main entry ─────────────────────────────
def find_care(capability_ids, location_text, *, condition=None, urgency=None, route_checked=True):
    """Same output keys as recommend(): status, service_resolution, location_resolution, results,
    search_notes, how_we_decided — plus fallback_results, banner, capability_ids, path."""
    cat = caps.catalog()
    cap_ids = [c for c in dict.fromkeys(capability_ids) if c in cat["by_id"]]
    names = " + ".join(cat["by_id"][c]["name"] for c in cap_ids)
    svc = {"status": "matched" if cap_ids else "none", "display_name": names, "capability_ids": cap_ids,
           "note": f"Searching for {names}" + (f" (from '{condition}')" if condition else "") + "."}
    base = {"status": None, "service_resolution": svc, "location_resolution": None, "results": [],
            "fallback_results": [], "search_notes": [], "how_we_decided": {}, "capability_ids": cap_ids,
            "banner": None, "path": "capabilities"}
    cond = next((k for k in cat["conditions"] if condition and condition in (k["condition_id"], k["label"])), None)
    if cond is None and condition:
        hits = caps.suggest(condition)["conditions"]
        cond = next((k for k in cat["conditions"] if hits and k["condition_id"] == hits[0]["condition_id"]), None)
    if cond:
        urgency = urgency or cond["urgency"]
        if cond.get("banner"):
            base["banner"] = cond.get("banner_text") or caps.DEFAULT_BANNER
    if not cap_ids:
        return dict(base, status="unknown_service", search_notes=["No capability selected."])

    # exactly one checked service → the original, hand-tuned path, unchanged
    if route_checked and len(cap_ids) == 1 and cap_ids[0] in CHECKED:
        out = eng.recommend(CHECKED_QUERY[CHECKED[cap_ids[0]]], location_text)
        out = dict(out, fallback_results=[], capability_ids=cap_ids, banner=base["banner"], path="checked_service")
        out["service_resolution"] = dict(out["service_resolution"], display_name=names)
        return out

    loc = eng.resolve_location(location_text)
    base["location_resolution"] = loc
    if loc["status"] == "ambiguous":
        return dict(base, status="needs_location_clarification", search_notes=[loc["note"]])
    if loc["status"] != "resolved":
        return dict(base, status="unknown_location", search_notes=[loc["note"]])

    max_km, urg = _settings(cap_ids, urgency)
    fac = facilities()
    d = eng._hav_np(loc["lat"], loc["lon"], fac["final_lat"].to_numpy(float), fac["final_lon"].to_numpy(float))
    cand = fac.assign(distance_km=d)
    cand = cand[cand["final_lat"].notna()]
    overall = pd.Series("SPECIFIC", index=cand.index)
    any_hit = pd.Series(False, index=cand.index)
    mismatches = pd.Series(0, index=cand.index)
    ev_score = pd.Series(0.0, index=cand.index)
    for cid in cap_ids:
        lv = levels_for(cid).set_index("unique_id").reindex(cand["unique_id"])
        lv.index = cand.index
        cand[f"lv_{cid}"] = [dict(level=a, quote=b, field=c, penalties=p or [])
                             for a, b, c, p in zip(lv["level"].fillna("NONE"), lv["quote"], lv["field"], lv["penalties"])]
        lvl = lv["level"].fillna("NONE")
        overall = pd.Series([min(a, b, key=caps.LEVELS.index) for a, b in zip(overall, lvl)], index=cand.index)
        any_hit |= lvl != "NONE"
        mismatches += lv["domain_mismatch"].fillna(False).astype(int)
        ev_score += lvl.map(LEVEL_SCORE) / len(cap_ids)
    cand = cand.assign(overall_level=overall, _any=any_hit, _mismatch=mismatches, evidence_score=ev_score)
    cand = cand[cand["_any"]]
    cand["evidence_tier"] = cand["overall_level"].map(LEVEL_TO_TIER)

    # main pool: every capability LISTED or better; widen like 06 (1.5x, 3x) if fewer than 3
    good = cand[cand["overall_level"].isin(["LISTED", "SPECIFIC"])]
    radius, notes = max_km, []
    pool = good[good["distance_km"] <= radius]
    for mult in (1.5, 3.0):
        if len(pool) >= 3:
            break
        notes.append(f"Fewer than 3 facilities with {names} listed within {radius:.0f} km.")
        radius = max_km * mult
        pool = good[good["distance_km"] <= radius]
        notes.append(f"Widened the search to {radius:.0f} km.")

    results, fallback = [], []
    fitting = pool[pool["_mismatch"] == 0]
    if len(fitting) >= 3:          # 'rank lower': facilities of another main type only fill missing slots
        pool = fitting
    if len(pool):
        scored = eng.score_candidates(pool, urg)
        scored["final_score"] = scored["final_score"] - DOMAIN_MISMATCH_PENALTY * scored["_mismatch"]
        results = [_explain(p, cap_ids, eng.WEIGHTS[urg]) for p in eng.pick_top3(scored)]
    if len(results) < 3:
        rest = cand[~cand["unique_id"].isin({r["unique_id"] for r in results})
                    & (cand["distance_km"] <= radius)]
        if len(rest):
            scored = eng.score_candidates(rest, urg)
            scored["final_score"] = scored["final_score"] - DOMAIN_MISMATCH_PENALTY * scored["_mismatch"]
            scored = scored.sort_values("final_score", ascending=False).head(5)
            fallback = [_explain({"row": r, "role": "Nothing better nearby"}, cap_ids, eng.WEIGHTS[urg])
                        for _, r in scored.iterrows()]
            notes.append("Facilities that only mention these services in passing (or lack some of them) are listed "
                         "separately under 'nothing better nearby' — call to check before going.")
    status = "ok" if results else ("no_results" if not fallback else "only_weak_results")
    how = {"service": svc["note"],
           "location": loc["note"],
           "weights": f"Treated as a {urg} need — score weights " +
                      ", ".join(f"{k} {v}" for k, v in eng.WEIGHTS[urg].items()) + ".",
           "matching": "A facility counts only if it lists every selected service (multi-select = AND); its level "
                       "is the weakest of them. Services only mentioned in passing never make a top pick.",
           "widening": [n for n in notes if n.startswith("Widened")], "urgency": urg,
           "final_search_radius_km": radius}
    return dict(base, status=status, results=results, fallback_results=fallback, search_notes=notes,
                how_we_decided=how)


# ───────────────────────────── LLM fallback (Enter with no local match) ─────────────────────────────
LLM_PICK_FORMAT = "STRUCT<answer STRUCT<capability_ids ARRAY<STRING>>>"


def llm_capabilities(text):
    """Llama picks 3–6 capability_ids FROM THE CATALOG ONLY (validated); cached per normalised text.
    Returns (capability_ids, error_or_None) — errors are returned, never hidden."""
    key = ("llm", caps.norm(text))
    if key in _CACHE:
        return _CACHE[key], None
    try:
        parsed = eng._unwrap(eng._ai_query(caps.llm_prompt(text), LLM_PICK_FORMAT), "answer")
        ids = caps.validate_llm_ids((parsed or {}).get("capability_ids"))
    except Exception as e:
        return [], f"AI suggestion failed: {type(e).__name__}: {str(e)[:200]}"
    _CACHE[key] = ids
    return ids, None
