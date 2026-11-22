"""Turn an engine result (find_care() or recommend()) into a coordinator's call plan.

Display only — ranking is never changed here. Everything shown comes from columns the engine already
uses (Gold verified + 03b boilerplate flag + 04 evidence) or from the config/*.json templates.
"""
import html
import json
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

import capabilities as caps

CFG = Path(__file__).resolve().parent / "config"
CHECKLISTS = json.loads((CFG / "call_checklists.json").read_text())
BRING = json.loads((CFG / "what_to_bring.json").read_text())
SLIP = {"en": json.loads((CFG / "slip_en.json").read_text()), "hi": json.loads((CFG / "slip_hi.json").read_text())}
# DEMO ONLY: config/demo_phone.local.json (gitignored, never on GitHub) wins over the committed demo_phone.json (off).
_DEMO_FILE = next((f for f in (CFG / "demo_phone.local.json", CFG / "demo_phone.json") if f.exists()), None)
DEMO_PHONE = json.loads(_DEMO_FILE.read_text()) if _DEMO_FILE else {}
DEMO_TAG = {"en": "(demo number)", "hi": "(डेमो नंबर)"}

CAR_DETOUR, CAR_KMH = 1.4, 40.0                       # same assumptions as travel_times v1
LEVEL_FROM_TIER = {"strong": "SPECIFIC", "moderate": "LISTED", "weak": "PASSING"}
EVIDENCE_LABEL = {"SPECIFIC": "Specific evidence", "LISTED": "Listed, not detailed — call to confirm",
                  "PASSING": "Passing mention", "NONE": "Not listed"}
EVIDENCE_COLOUR = {"SPECIFIC": "#1B7F4C", "LISTED": "#B7791F", "PASSING": "#6B7280", "NONE": "#6B7280"}
ROLE_REASON = {"Best overall": "best overall match", "Closest credible option": "closest that lists it",
               "Strongest evidence": "strongest evidence", "Next best option": "next best option",
               "Nothing better nearby": "not confirmed — call to check",
               "Model 1 pick": "Model 1's best evidence nearby", "Model 2 pick": "most likely capable nearby",
               "Combined pick": "best combined label nearby"}
CALL_ORDER = ["Call first", "Call second", "Call third", "Call fourth", "Call fifth"]
FIELD_LABEL = {"equipment": "equipment list", "procedures": "procedures", "capabilities": "services",
               "specialties": "specialties", "description": "description",
               "checked service evidence": "service evidence"}
# reverse of engine.find_care.CHECKED: classic-path service_id → catalog capability_id
SERVICE_TO_CAP = {"dialysis": "dialysis", "c_section": "c_section", "nicu": "nicu", "cardiac_surgery": "cardiac_surgery",
                  "angioplasty": "cath_lab", "mri": "mri", "burn_care": "burn_care"}


# ───────────────────────────── small helpers ─────────────────────────────
def val(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, str) and x.strip().lower() in ("", "nan", "none", "null"):
        return None
    return x


def plain(text):
    """Engine text in plain words: no 'tier', 'score' or 'moderate'."""
    t = str(text or "")
    t = re.sub(r"\s*\(evidence tier: \w+\)", "", t)
    t = re.sub(r"evidence tier: (\w+)", lambda m: {"strong": "specific", "moderate": "listed", "weak": "passing"}.get(m.group(1), m.group(1)) + " evidence", t)
    t = re.sub(r"\bmoderate\b", "listed", t)
    t = re.sub(r"\bstrong or listed\b", "specific or listed", t)
    t = re.sub(r"\bscore weights?\b", "ranking weights", t)
    return t


def travel_min(km):
    return max(1, int(round(float(km) * CAR_DETOUR / CAR_KMH * 60)))


def by_car(minutes, lang="en"):
    T = SLIP[lang]
    return T["by_car_short"] if minutes < 5 else T["by_car"].format(min=minutes)


def phone_info(r, fac_row):
    raw = val(r.get("phone"))
    if not raw and fac_row is not None:
        nums = fac_row.get("phone_numbers")
        nums = [n for n in (list(nums) if nums is not None and not isinstance(nums, str) else []) if val(n)]
        raw = nums[0] if nums else None
    if not raw:
        return None, None, False
    digits = re.sub(r"\D", "", str(raw))
    if digits.startswith("91") and len(digits) == 12:
        local = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        local = digits[1:]
    else:
        local = digits
    valid = len(local) == 10 and (local[0] in "6789" or local[0] in "12345")
    shown = f"+91 {local[:5]} {local[5:]}" if valid else str(raw)
    return shown, ("tel:+91" + local) if valid else "tel:" + digits, valid


def address(row):
    if row is None:
        return ""
    parts = []
    for c in ("city", "district", "state"):
        v = val(row.get(c))
        if v and str(v).lower() not in [p.lower() for p in parts]:
            parts.append(str(v))
    pin = val(row.get("pincode"))
    return ", ".join(parts) + (f" {pin}" if pin else "")


def bold_matches(text, cap_ids):
    """Bold the catalog terms that matched (x-ray / X Ray / xray all match the variant 'xray')."""
    t = html.escape(str(text))
    cat = caps.catalog()["by_id"]
    pats = set()
    for cid in cap_ids:
        for v in cat[cid]["_variants"]:
            chars = [re.escape(c) for c in v.replace(" ", "")]
            if len(chars) >= 3:
                pats.add(r"[\W_]*".join(chars))
    if not pats:
        return t
    rx = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(pats, key=len, reverse=True)) + r")s?(?![A-Za-z])", re.I)
    return rx.sub(lambda m: f"<b>{m.group(0)}</b>", t)


# ───────────────────────────── per-result view ─────────────────────────────
def _cap_ids_for(out):
    ids = out.get("capability_ids") or []
    if ids:
        return ids
    sid = (out.get("service_resolution") or {}).get("service_id")
    return [SERVICE_TO_CAP[sid]] if sid in SERVICE_TO_CAP else []


def _level(r):
    return r.get("overall_level") or LEVEL_FROM_TIER.get(r.get("evidence_tier"), "PASSING")


def trust_chips(r, row, urgency, fc):
    chips = []
    if row is None:
        return chips
    src = str(row.get("location_source") or "")
    conf = str(row.get("location_confidence") or "")
    if src.startswith("directory") or "matches directory" in src:
        chips.append(("ok", "Location verified (government record)"))
    elif conf == "high":
        chips.append(("ok", "Location confirmed"))
    else:
        chips.append(("warn", "Location approximate" if conf == "medium" else "Location uncertain"))
    _, _, valid = phone_info(r, row)
    chips.append(("ok", "Phone number valid") if valid else ("warn", "No valid phone number"))
    dm = str(row.get("directory_match") or "NONE")
    if dm == "STRONG":
        chips.append(("ok", "In government hospital directory"))
    elif dm == "UNVERIFIED":
        chips.append(("warn", "Government directory match unverified"))
    chips.append(("warn", "Services text looks like a shared template") if bool(row.get("possible_boilerplate"))
                 else ("ok", "Branch-specific service details"))
    if urgency == "emergency":
        lv = fc.levels_for("emergency_247").set_index("unique_id")["level"].get(row["unique_id"], "NONE")
        chips.append(("ok", "24/7 emergency listed") if lv in ("LISTED", "SPECIFIC") else ("warn", "24/7 emergency not listed"))
    op = str(row.get("operator_type") or "unknown")
    chips.append(("info", "Government (public)" if op == "public" else "Private" if op == "private" else "Ownership unknown"))
    return chips


def evidence_view(r, cap_ids, row):
    """Main quote (matched words bold) + other evidence bullets tagged by source field."""
    levels = r.get("capability_levels") or []
    main, main_field = None, None
    for x in levels:
        if x.get("quote"):
            main, main_field = x["quote"], x.get("field")
            break
    if main is None:
        main, main_field = plain(r.get("why_service")), "service evidence"
    bullets, seen = [], {str(main).strip().lower()}
    if row is not None:
        ments = row.get("cap_mentions") or {}
        for cid in cap_ids:
            for field, item, _ in ments.get(cid, []):
                k = str(item).strip().lower()
                if k not in seen:
                    seen.add(k)
                    bullets.append((FIELD_LABEL.get(field, field), item))
    return {"main_html": bold_matches(main, cap_ids), "main_field": FIELD_LABEL.get(main_field, main_field or ""),
            "bullets": [(f, bold_matches(i, cap_ids)) for f, i in bullets[:12]]}


def questions(r, cap_ids, urgency):
    qs = []
    if urgency == "emergency":
        qs += CHECKLISTS["_emergency"]
    for cid in cap_ids[:2]:
        qs += CHECKLISTS.get(cid, [])[:2]
    names = caps.catalog()["by_id"]
    for x in r.get("capability_levels") or []:
        if x["level"] == "PASSING":
            qs.append(CHECKLISTS["_weak"].format(name=names[x["capability_id"]]["name"]))
        elif x["level"] == "NONE":
            qs.append(CHECKLISTS["_missing"].format(name=names[x["capability_id"]]["name"]))
    if not cap_ids:
        qs += CHECKLISTS["_default"]
    return list(dict.fromkeys(qs))[:4]


def _reason_phrase(card):
    lv = card["level"]
    if lv == "SPECIFIC":
        return f"{card['first_cap']} with specific details in its {card['evidence']['main_field'] or 'record'}"
    if lv == "LISTED":
        return f"{card['first_cap']} listed in its {card['evidence']['main_field'] or 'record'}"
    return f"{card['first_cap']} only mentioned in passing"


def build_view(out, fc):
    """→ dict(summary, cards, fallback_cards, plan, govt, why_not, urgency, ranking_rows)"""
    fac = fc.facilities().set_index("unique_id", drop=False)
    cap_ids = _cap_ids_for(out)
    names = caps.catalog()["by_id"]
    how = out.get("how_we_decided") or {}
    urgency = how.get("urgency") or "planned"

    def card(r, i, role_label):
        row = fac.loc[r["unique_id"]] if r["unique_id"] in fac.index else None
        shown, tel, valid = phone_info(r, row)
        lv = _level(r)
        c = {"n": i + 1, "unique_id": r["unique_id"], "name": r["name"], "role": role_label,
             "reason": ROLE_REASON.get(r.get("role"), "option"), "km": float(r["distance_km"]),
             "minutes": travel_min(r["distance_km"]), "level": lv,
             "level_label": r.get("display_label") or EVIDENCE_LABEL[lv],          # model tabs bring their own wording
             "level_colour": r.get("display_colour") or EVIDENCE_COLOUR[lv], "phone": shown, "tel": tel, "address": address(row),
             "lat": r.get("lat"), "lon": r.get("lon"), "chips": trust_chips(r, row, urgency, fc),
             "evidence": evidence_view(r, cap_ids, row), "questions": questions(r, cap_ids, urgency),
             "warnings": [plain(w) for w in (r.get("warnings") or [])],
             "location_note": plain(r.get("why_location")), "components": r.get("components") or {},
             "first_cap": names[cap_ids[0]]["name"] if cap_ids else (out["service_resolution"].get("display_name") or "Service"),
             "website": val(r.get("website"))}
        return c

    cards = [card(r, i, CALL_ORDER[i]) for i, r in enumerate(out.get("results") or [])]
    fb = [card(r, i, "Possible option") for i, r in enumerate(out.get("fallback_results") or [])]
    shown_ids = {c["unique_id"] for c in cards + fb}
    loc = out.get("location_resolution") or {}
    radius = float(how.get("final_search_radius_km") or 100.0)
    govt = public_fallback(loc, radius, shown_ids, fc) if loc.get("lat") is not None else None
    if govt and govt.get("name"):
        govt["cap_ids"] = cap_ids
        govt["listed_caps"] = [c for c in cap_ids
                               if fc.levels_for(c).set_index("unique_id")["level"].get(govt["unique_id"], "NONE") != "NONE"]
    why = why_not(cap_ids, loc, radius, shown_ids, fc, out) if loc.get("lat") is not None and cap_ids else []

    plan_cards = cards or fb[:3]
    if DEMO_PHONE.get("enabled") and plan_cards:            # DEMO ONLY — see config/demo_phone.json
        plan_cards[0]["phone"], plan_cards[0]["tel"] = DEMO_PHONE["phone_display"], DEMO_PHONE["tel"]
        plan_cards[0]["demo_phone"] = True
    if plan_cards:
        far = max(c["minutes"] for c in plan_cards)
        summary = (f"{len(plan_cards)} option{'s' if len(plan_cards) != 1 else ''} within ~{far} min by car (estimated). "
                   f"Call {plan_cards[0]['name']} first: {_reason_phrase(plan_cards[0])}.")
        if not cards:
            summary = "No facility nearby clearly lists this. " + summary + " None of these is confirmed — ask before going."
    else:
        summary = "No facility in our data lists this within reach."
    return {"summary": summary, "cards": cards, "fallback_cards": fb, "plan": plan_cards[:3], "govt": govt,
            "why_not": why, "urgency": urgency, "cap_ids": cap_ids, "radius": radius}


# ───────────────────────────── fallback + exclusions ─────────────────────────────
_GOVT_GENERAL = re.compile(r"district hospital|sadar|civil hospital|general hospital|medical college|community health|"
                           r"\bchc\b|sub.?divisional|referral hospital|\bfru\b|aiims|esic|zilla|jilla|"
                           r"government hospital|govt\.? hospital|primary health", re.I)
_GOVT_SPECIALTY = re.compile(r"maternity|women|mahila|eye|dental|mental|psychiatr|\btb\b|chest|cancer|ayurved|homo?eo|"
                             r"unani|leprosy|ent\b|children|child|paediatric|pediatric", re.I)


def govt_line(g, lang="en"):
    """Honest one-liner for the government fallback: does its record list any selected service? Never says confirmed."""
    T = SLIP[lang]
    by_id = caps.catalog()["by_id"]
    name = (lambda c: SLIP["hi"]["capabilities"].get(c, by_id[c]["name"])) if lang == "hi" else (lambda c: by_id[c]["name"])
    listed, wanted = g.get("listed_caps") or [], g.get("cap_ids") or []
    if listed:
        return T["govt_listed"].format(service=" + ".join(name(c) for c in listed))
    return T["govt_not_listed"].format(service=" + ".join(name(c) for c in wanted) if wanted else
                                       ("इस सेवा" if lang == "hi" else "this service"))


def public_fallback(loc, radius, exclude, fc):
    fac = fc.facilities()
    pub = fac[(fac["operator_type"] == "public") & (fac["facility_type"].str.lower() == "hospital")
              & fac["final_lat"].notna() & ~fac["unique_id"].isin(exclude)].copy()
    if pub.empty:
        return None
    from engine import recommend_engine as eng
    pub["km"] = eng._hav_np(loc["lat"], loc["lon"], pub["final_lat"].to_numpy(float), pub["final_lon"].to_numpy(float))
    pub = pub[pub["km"] <= max(radius, 50.0)]
    if pub.empty:
        return {"none_within_km": int(max(radius, 50.0))}
    # general government hospitals first; specialty-named ones (maternity, eye, TB…) only as a last resort
    special = pub["name"].str.contains(_GOVT_SPECIALTY)
    pub["rank"] = np.where(pub["name"].str.contains(_GOVT_GENERAL) & ~special, 0,
                           np.where(~special & (pub["main_domain"] == "general"), 1, 2))
    best = pub.sort_values(["rank", "km"]).iloc[0]
    shown, tel, _ = phone_info({"phone": best.get("phone")}, best)
    return {"unique_id": best["unique_id"], "name": best["name"], "km": float(best["km"]), "minutes": travel_min(best["km"]), "phone": shown, "tel": tel,
            "address": address(best), "lat": float(best["final_lat"]), "lon": float(best["final_lon"])}


def why_not(cap_ids, loc, radius, shown_ids, fc, out):
    """2–3 nearest facilities that mention the service but were left out, each with a REAL reason taken from
    the flags the engine used. Never invents a reason: candidates without a recorded reason are skipped."""
    from engine import recommend_engine as eng
    fac = fc.facilities()
    names = caps.catalog()["by_id"]
    lv = {cid: fc.levels_for(cid).set_index("unique_id") for cid in cap_ids}
    mentioned = pd.Series(False, index=fac.index)
    for cid in cap_ids:
        mentioned |= fac["unique_id"].map(lv[cid]["level"]).fillna("NONE").ne("NONE")
    cand = fac[mentioned & ~fac["unique_id"].isin(shown_ids) & fac["final_lat"].notna()].copy()
    if cand.empty:
        return []
    cand["km"] = eng._hav_np(loc["lat"], loc["lon"], cand["final_lat"].to_numpy(float), cand["final_lon"].to_numpy(float))
    cand = cand[cand["km"] <= radius].sort_values("km")
    rows = []
    for _, f in cand.head(40).iterrows():
        reasons = []
        for cid in cap_ids:
            x = lv[cid].loc[f["unique_id"]]
            if x["domain_mismatch"]:
                reasons.append(f"Specialist clinic in a different field (mainly {str(f['main_domain']).replace('_', ' ')})")
            elif x["level"] == "NONE":
                reasons.append(f"Does not list {names[cid]['name']}")
            elif x["level"] == "PASSING":
                q = str(x["quote"] or "").lower()
                reasons.append(f"Mentions {names[cid]['name']} only as a referral" if "refer" in q
                               else f"Only mentions {names[cid]['name']} in passing ({FIELD_LABEL.get(x['field'], x['field'])})")
            if x["penalties"] and any("contradiction" in p for p in x["penalties"]):
                reasons.append(f"Facility type or size does not fit {names[cid]['name']}")
        if bool(f.get("possible_boilerplate")):
            reasons.append("Services text looks like a shared template, not specific to this branch")
        if bool(f.get("possible_merged_record")):
            reasons.append("Possible duplicate record (may combine several facilities)")
        if not phone_info({"phone": f.get("phone")}, f)[2]:
            reasons.append("No working phone number listed")
        if reasons:
            rows.append({"name": f["name"], "km": float(f["km"]), "minutes": travel_min(f["km"]), "reason": reasons[0],
                         "more": reasons[1:3]})
        if len(rows) >= 3:
            break
    return rows


# ───────────────────────────── the slip ─────────────────────────────
def bring_list(cap_ids, urgency, lang):
    items = list(BRING["_default"])
    if urgency == "emergency":
        items += BRING["_emergency"]
    for cid in cap_ids:
        items += BRING.get(cid, [])
    seen, out = set(), []
    for it in items:
        if it["en"] not in seen:
            seen.add(it["en"])
            out.append(it[lang])
    return out[:6]


def need_text(cap_ids, typed, lang):
    names = caps.catalog()["by_id"]
    if lang == "hi":
        parts = [SLIP["hi"]["capabilities"].get(c, names[c]["name"]) for c in cap_ids]
    else:
        parts = [names[c]["name"] for c in cap_ids]
    t = " + ".join(parts)
    if typed:
        t += " " + SLIP[lang]["from_words"].format(typed=typed)
    return t


def slip_text(view, typed, near_label, lang="en", need_label=None):
    T = SLIP[lang]
    need = need_text(view["cap_ids"], typed, lang) if view["cap_ids"] else (need_label or "")
    lines = [T["title"], f"{T['need']}: {need}", f"{T['near']}: {near_label}", ""]
    if view["urgency"] == "emergency":
        lines += [T["emergency"], ""]
    lines.append(T["call_order"])
    for c in view["plan"]:
        phone = (f"{T['phone']}: {c['phone']}" + (f" {DEMO_TAG[lang]}" if c.get("demo_phone") else "")) if c["phone"] else T["no_phone"]
        lines.append(f"{c['n']}. {c['name']} — {phone} — {c['address']} — {by_car(c['minutes'], lang)} — {T['confirm']}")
    g = view["govt"]
    lines.append("")
    if g and g.get("name"):
        phone = f"{T['phone']}: {g['phone']}" if g["phone"] else T["no_phone"]
        lines.append(f"{T['if_none']} {g['name']} — {govt_line(g, lang)} — {phone} — {g['address']} — "
                     f"{by_car(g['minutes'], lang)}")
    elif g:
        lines.append(T["no_govt"].format(km=g["none_within_km"]))
    lines += ["", T["bring"]] + [f"- {b}" for b in bring_list(view["cap_ids"], view["urgency"], lang)]
    lines += ["", T["footer"]]
    return "\n".join(lines)


def whatsapp_url(text):
    return "https://wa.me/?text=" + quote(text, safe="")


# ───────────────────────────── Model 1 (testing) ─────────────────────────────
# One extra "From Model 1" card. Reads catalog/model1_levels.csv, produced by tools/run_model1.py from the team's
# 'Model 1 - Evidence Levels' notebook. Display only: it does not touch the ranking, call plan or slip.
MODEL1_SERVICE = {"dialysis": "dialysis", "icu": "icu", "nicu": "nicu", "c_section": "c_section",
                  "cardiac_surgery": "cardiac_surgery", "cath_lab": "cath_lab_angioplasty", "mri": "mri",
                  "ct_scan": "ct_scan", "emergency_247": "emergency_24x7", "burn_care": "burn_care",
                  "chemotherapy": "chemotherapy", "radiotherapy": "radiotherapy", "blood_bank": "blood_bank"}
_M1_RANK = {"SPECIFIC": 2, "LISTED": 1, "PASSING": 0}
_M1 = {}


def _model1():
    if "levels" not in _M1:
        root = Path(__file__).resolve().parent / "catalog"
        lv = root / "model1_levels.csv"
        _M1["levels"] = pd.read_csv(lv, dtype={"unique_id": str}) if lv.exists() else None
        sj = root / "model1_services.json"
        _M1["services"] = json.loads(sj.read_text()) if sj.exists() else {}
    return _M1["levels"], _M1["services"]


def model1_pick(view, out, fc):
    """→ card dict for Model 1's best facility in range, or {'note': ...} when Model 1 can't answer."""
    levels, services = _model1()
    loc = out.get("location_resolution") or {}
    if levels is None or loc.get("lat") is None:
        return None
    names = caps.catalog()["by_id"]
    svcs = [MODEL1_SERVICE[c] for c in view["cap_ids"] if c in MODEL1_SERVICE]
    if not svcs:
        return {"note": "Model 1 does not score " + " + ".join(names[c]["name"] for c in view["cap_ids"]) + " yet."}
    sub = levels[levels["service"].isin(svcs)]
    per = sub.groupby("unique_id").agg(n=("service", "nunique"),
                                       rank=("evidence_level", lambda s: min(_M1_RANK[x] for x in s)),
                                       pct=("directory_confirm_pct", "mean"))
    per = per[per["n"] == len(set(svcs))]                      # must cover every selected Model 1 service
    fac = fc.facilities().set_index("unique_id", drop=False)
    per = per[per.index.isin(fac.index)]
    if per.empty:
        return {"note": "Model 1 found no facility covering this."}
    from engine import recommend_engine as eng
    f = fac.loc[per.index]
    per["km"] = eng._hav_np(loc["lat"], loc["lon"], f["final_lat"].to_numpy(float), f["final_lon"].to_numpy(float))
    per = per[per["km"] <= view["radius"]]
    if per.empty:
        return {"note": f"Model 1 has no facility within {view['radius']:.0f} km for this."}
    # Model 1's best evidence level first, then the NEAREST such facility, then directory-confirm % as tie-break
    best_id = per.assign(km_r=per["km"].round(0)).sort_values(["rank", "km_r", "pct"], ascending=[False, True, False]).index[0]
    row, b = fac.loc[best_id], per.loc[best_id]
    lv = {2: "SPECIFIC", 1: "LISTED", 0: "PASSING"}[int(b["rank"])]
    m = sub[sub["unique_id"] == best_id]
    first = m.sort_values("evidence_level", key=lambda s: -s.map(_M1_RANK)).iloc[0]
    shown, tel, _ = phone_info({"phone": row.get("phone")}, row)
    return {"n": "M1", "unique_id": best_id, "name": row["name"], "role": "From Model 1 (testing)",
            "reason": "Model 1's best-rated facility nearby" + (
                " (scores " + " + ".join(names[c]["name"] for c in view["cap_ids"] if c in MODEL1_SERVICE) + " only)"
                if any(c not in MODEL1_SERVICE for c in view["cap_ids"]) else ""), "km": float(b["km"]), "minutes": travel_min(b["km"]),
            "level": lv, "level_label": EVIDENCE_LABEL[lv], "level_colour": EVIDENCE_COLOUR[lv], "phone": shown,
            "tel": tel, "address": address(row),
            "pct": None if pd.isna(b["pct"]) else int(round(float(b["pct"]))),
            "questions": [services.get(s, {}).get("checklist") for s in svcs if services.get(s, {}).get("checklist")],
            "quote": str(first["quote"]), "quote_field": str(first["quote_field"]),
            "reasons": [r.strip() for r in str(first["reasons"]).split("|") if r.strip() and r.strip() != "nan"]}


# ───────────────────────────── "Tell the family" (WhatsApp) ─────────────────────────────
# Template only (no LLM, no patient name). The family's number is used only to build the wa.me link:
# nothing here stores, prints or logs it.
FAMILY_BRING = json.loads((CFG / "family_bring.json").read_text())
FAMILY_MAX_CHARS = 900


def normalise_whatsapp(raw):
    """Indian mobile → '91XXXXXXXXXX'. Returns (number_or_None, hint)."""
    d = re.sub(r"\D", "", str(raw or ""))
    if d.startswith("0") and len(d) == 11:
        d = d[1:]
    if len(d) == 10:
        d = "91" + d
    if len(d) == 12 and d.startswith("91") and d[2] in "6789":
        return d, ""
    return None, "Enter the family's 10-digit Indian mobile number, e.g. 98765 43210."


def family_targets(view):
    """The 3 call-plan facilities + the government fallback, in call order."""
    out = [{"label": f"{c['role']}: {c['name']}", "name": c["name"], "address": c["address"], "lat": c.get("lat"),
            "lon": c.get("lon"), "minutes": c["minutes"], "phone": c["phone"], "demo_phone": c.get("demo_phone", False)}
           for c in view["plan"]]
    g = view.get("govt")
    if g and g.get("name"):
        out.append({"label": f"Government fallback: {g['name']}", "name": g["name"], "address": g["address"],
                    "lat": g.get("lat"), "lon": g.get("lon"), "minutes": g["minutes"], "phone": g["phone"],
                    "govt": True, "cap_ids": g.get("cap_ids") or [], "listed_caps": g.get("listed_caps") or []})
    return out


def _family_bring(cap_ids, lang):
    items = list(FAMILY_BRING["_default"])
    for group, ids in FAMILY_BRING["_groups"].items():
        if set(ids) & set(cap_ids or []):
            items += FAMILY_BRING[group]
    seen, out = set(), []
    for it in items:
        if it["en"] not in seen:
            seen.add(it["en"])
            out.append(it[lang])
    return out


def family_message(target, need, confirmed, sender, urgency, cap_ids, lang="en"):
    """WhatsApp text for the patient's family, under 900 characters. Names, numbers and links are never translated."""
    T = SLIP[lang]["family"]
    bring = _family_bring(cap_ids, lang)

    def build(items, address):
        lines = [T["title"], target["name"]]
        if address:
            lines.append(address)
        if target.get("lat") is not None and target.get("lon") is not None:
            lines.append(f"{T['map']}: https://maps.google.com/?q={float(target['lat']):.5f},{float(target['lon']):.5f}")
        lines.append(by_car(target["minutes"], lang))
        lines.append(f"{T['need']}: {need}")
        if target.get("govt"):
            lines.append(govt_line(target, lang))
        lines.append(T["confirmed_yes"] if confirmed else T["confirmed_no"])
        if target.get("phone"):
            lines.append(f"{T['hospital_phone']}: {target['phone']}" + (f" {DEMO_TAG[lang]}" if target.get("demo_phone") else ""))
        if items:
            lines.append(f"{T['please_bring']}: " + ", ".join(items))
        if urgency == "emergency":
            lines.append(T["emergency"])
        s = str(sender or "").strip()
        lines.append(T["sent_by"].format(sender=s[:40]) if s else T["sent_anon"])
        return "\n".join(lines)

    msg, address = build(bring, target.get("address") or ""), target.get("address") or ""
    while len(msg) > FAMILY_MAX_CHARS and bring:           # trim the bring list first, then the address
        bring = bring[:-1]
        msg = build(bring, address)
    if len(msg) > FAMILY_MAX_CHARS:
        msg = build(bring, address[:60])
    return msg[:FAMILY_MAX_CHARS]


def whatsapp_to(number, text):
    return f"https://wa.me/{number}?text=" + quote(text, safe="")
