# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Model 1 — Service Evidence Levels + percentage
# MAGIC For every facility × key service: an evidence level (**SPECIFIC / LISTED / PASSING**), the quote behind it, the reasons, and **directory_confirm_pct** —
# MAGIC the chance the government hospital directory also lists that service, learned from how the service is presented.
# MAGIC It is *not* the chance the service is real. See "PRD: Model 1 — Service Evidence Levels".
# MAGIC
# MAGIC **Run all.** Nothing is saved as a table unless the last cell is un-commented.

# COMMAND ----------

# MAGIC %pip install rapidfuzz scikit-learn --quiet

# COMMAND ----------

FACILITIES = "/Volumes/workspace/datathon_2026/csv_data/gold_facility_search_services.csv"
DIRECTORY  = "/Volumes/workspace/datathon_2026/csv_data/hospital_directory.csv"

# COMMAND ----------

# MAGIC %md ## 1. Service-term rules (which words mean which service)

# COMMAND ----------

"""Collapse the raw service terms into a clean, grouped list for human review.

  python3 normalize_services.py services_top.csv

Input:  services_top.csv  (term, facility_count_any_field, fields, checked_service, example_raw_text)
Output: service_terms_mapped.csv   one row per raw term: cleaned form, group, kind, service, reason
        service_groups_review.csv  one row per group: what a reviewer approves or fixes
        service_mapping_changes.csv where this mapping differs from the existing checked_service column

Nothing here decides whether a facility REALLY offers a service. It only decides which words
count as talking about which service. Evidence strength is Model 1's job.
"""

import re
import sys

import pandas as pd
from rapidfuzz import fuzz, process

# --------------------------------------------------------------------------- services
# include = words that mean the service; exclude = look-alikes that mean something else.
SERVICES = {
    "dialysis":        (r"dialys|haemodial|hemodial", r"peritoneal|capd"),
    "peritoneal_dialysis": (r"peritoneal dialys|\bcapd\b", r"$^"),
    "icu":             (r"\bicu\b|intensive care|critical care unit|\bccu\b|\bmicu\b|\bsicu\b",
                        r"neonatal|nicu|picu|pediatric|paediatric|cardiac care unit"),
    "nicu":            (r"\bnicu\b|neonatal intensive|newborn intensive|level ii+ neonat", r"$^"),
    "picu":            (r"\bpicu\b|paediatric intensive|pediatric intensive", r"$^"),
    "c_section":       (r"caesar|cesar|\blscs\b|c[- ]section", r"$^"),
    "cardiac_surgery": (r"cabg|coronary artery bypass|open heart|cardiac surg|heart surg|valve (replacement|repair)"
                        r"|cardiothoracic|\bctvs\b|heart bypass|bypass surg", r"gastric|bariatric|weight|obesity"),
    "cath_lab_angioplasty": (r"cath ?lab|angioplast|\bptca\b|coronary stent|cardiac cath|angiograph",
                             r"bile|biliary|ureter|esophag|oesoph|pancrea|colon|renal artery|carotid|peripheral|cerebral|\bct\b|\bmr\b|mri|fluorescein|ocular|retina"),
    "mri":             (r"\bmri\b|magnetic resonance", r"$^"),
    "pet_ct":          (r"pet[- ]?ct|\bpet scan", r"$^"),
    "ct_scan":         (r"\bct\b|computed tomograph|cat scan", r"pet[- ]?ct|\bct guided|cbct|cone beam"),
    "trauma_center":   (r"trauma (care|centre|center|unit|surgery|service)|^(emergency (and|&) )?trauma$|polytrauma|accident (and|&) emergency",
                        r"eye|ocular|ophthal|plastic|burn|dental|dentist|tooth|teeth|maxillofacial|ortho|fracture|psych|emotional"),
    "emergency_24x7":  (r"emergency", r"eye|dental|vet|pet"),
    "burn_care":       (r"\bburns?\b (unit|ward|care|management|treatment|centre|center)|burns? management|^burns?$", r"$^"),
    "chemotherapy":    (r"chemo", r"$^"),
    "radiotherapy":    (r"radiotherap|radiation therap|linac|linear accelerator|brachytherap|cobalt", r"$^"),
    "blood_bank":      (r"blood bank|blood storage", r"$^"),
    "ventilator":      (r"ventilator", r"$^"),
}

# Specialty codes that point at a service, but only weakly (a specialty is not a unit).
SPECIALTY_HINTS = {
    "nephrology": "dialysis", "neonatologyperinatalmedicine": "nicu", "criticalcaremedicine": "icu",
    "cardiothoracicsurgery": "cardiac_surgery", "interventionalcardiology": "cath_lab_angioplasty",
    "medicaloncology": "chemotherapy", "radiationoncology": "radiotherapy",
    "emergencymedicine": "emergency_24x7", "gynecologyandobstetrics": "c_section",
}
# Near-universal for their facility type in this data -> carry no evidence.
DEFAULT_LABELS = {"internalmedicine", "familymedicine"}

# Facility attributes: real information, but not a medical service.
ATTRIBUTE = re.compile(
    r"open 24|operates 24|24 ?hours|24 ?hrs|24/7$|wheel ?chair|parking|private hospital|public hospital|"
    r"multi ?special|super ?special|inpatient and outpatient|in-patient|out-patient|outpatient services|"
    r"pharmacy|canteen|cafeteria|wifi|wi-fi|insurance|cashless|tpa|restroom|toilet|appointment|online|"
    r"lgbtq|gender[- ]neutral|accepts|credit card|debit card|upi|air[- ]condition|lift|elevator|waiting area|"
    r"home visit|teleconsult|tele[- ]?medicine|ambulance service|^hospital$|^clinic$|nursing home$|"
    r"government hospital|trust hospital|charitable")

LEAD = re.compile(r"^(?:we |it |the hospital |the clinic |hospital |clinic )?"
                  r"(?:provides?|offers?|offering|performs?|performing|has(?: an?| its own)?|have|having|"
                  r"specializ(?:es|ing) in|specialis(?:es|ing) in|equipped with|features?|available|"
                  r"conducts?|does|treats?|includes?|with)\s+")
TRAIL = re.compile(r"\s+(?:available|facility|facilities|services?|treatments?|on[- ]site|in[- ]house|"
                   r"department|dept|centre|center|machine|machines)$")


def clean(term: str) -> str:
    t = re.sub(r"[()\[\]\"',.;:]+", " ", str(term).lower())
    t = re.sub(r"\s+", " ", t).strip()
    for _ in range(3):                      # strip several layers: "provides dialysis services available"
        t2 = TRAIL.sub("", LEAD.sub("", t)).strip()
        if t2 == t:
            break
        t = t2 or t
    t = re.sub(r"\bscanner\b", "scan", t)
    t = re.sub(r"\bscans\b", "scan", t)
    return t


def classify(raw: str, cleaned: str, fields: str):
    """-> (kind, service, reason)"""
    if fields == "specialties" or raw in DEFAULT_LABELS or raw in SPECIALTY_HINTS:
        if raw in DEFAULT_LABELS:
            return "default_label", None, "listed for nearly every facility of its type - no evidence"
        if raw in SPECIALTY_HINTS:
            return "specialty", SPECIALTY_HINTS[raw], "specialty only hints at the service (weak)"
        if fields == "specialties":
            return "specialty", None, ""
    lookalike = None
    for svc, (inc, exc) in SERVICES.items():
        if re.search(inc, cleaned) and not re.search(exc, cleaned):
            return "service_mention", svc, ""          # a more specific rule (e.g. NICU) can still win
        if re.search(inc, cleaned) and lookalike is None:
            hit = re.search(exc, cleaned).group(0)
            lookalike = f"looks like {svc} but '{hit}' means something else"
    if lookalike:
        return "lookalike_excluded", None, lookalike
    if ATTRIBUTE.search(cleaned):
        return "facility_attribute", None, "describes the facility, not a medical service"
    return "other_service_text", None, ""


def group_terms(df: pd.DataFrame, threshold=92) -> pd.Series:
    """Greedy grouping of near-identical cleaned terms (most common term becomes the group name)."""
    order = df.groupby("cleaned")["facility_count_any_field"].sum().sort_values(ascending=False)
    names = list(order.index)
    group_of, heads = {}, []
    for n in names:
        if heads:
            m = process.extractOne(n, heads, scorer=fuzz.token_sort_ratio, score_cutoff=threshold)
            if m:
                group_of[n] = m[0]
                continue
        heads.append(n)
        group_of[n] = n
    return df["cleaned"].map(group_of)


def main_normalize_services(path):
    d = pd.read_csv(path)
    d["cleaned"] = d["term"].map(clean)
    out = d.apply(lambda r: classify(r["term"], r["cleaned"], r["fields"]), axis=1, result_type="expand")
    d[["kind", "service", "reason"]] = out
    d["group"] = group_terms(d)
    d.loc[d["service"].notna() & (d["kind"] == "service_mention"), "group"] = d["service"]

    d[["term", "cleaned", "group", "kind", "service", "reason", "facility_count_any_field", "fields",
       "checked_service", "example_raw_text"]].sort_values("facility_count_any_field", ascending=False) \
        .to_csv("service_terms_mapped.csv", index=False)

    g = (d.groupby(["group", "kind"], dropna=False)
           .agg(service=("service", lambda s: s.dropna().iloc[0] if s.notna().any() else ""),
                total_mentions=("facility_count_any_field", "sum"), n_variants=("term", "size"),
                examples=("example_raw_text", lambda s: " | ".join(s.astype(str).head(5))))
           .reset_index().sort_values("total_mentions", ascending=False))
    g["reviewer_ok"] = ""        # reviewer writes Y / N
    g["reviewer_fix"] = ""       # reviewer writes the correct service/group if N
    g.to_csv("service_groups_review.csv", index=False)

    old = d["checked_service"].fillna("")
    new = d["service"].where(d["kind"] == "service_mention", "").fillna("")
    norm_old = old.replace({"angioplasty": "cath_lab_angioplasty", "trauma": "trauma_center",
                            "cardiac_surgery|angioplasty": "cardiac_surgery"})
    changes = d[(norm_old != new) & ((old != "") | (new.isin(["dialysis", "nicu", "c_section", "cardiac_surgery",
                                                                "cath_lab_angioplasty", "mri", "trauma_center",
                                                                "burn_care"])))]
    changes[["term", "facility_count_any_field", "checked_service", "service", "kind", "reason"]] \
        .rename(columns={"checked_service": "old_mapping", "service": "new_mapping"}) \
        .sort_values("facility_count_any_field", ascending=False).to_csv("service_mapping_changes.csv", index=False)

    # ---- summary
    print(f"raw terms: {len(d)}  ->  cleaned forms: {d['cleaned'].nunique()}  ->  groups: {d['group'].nunique()}")
    print("\nby kind (terms / mentions):")
    print(d.groupby("kind").agg(terms=("term", "size"), mentions=("facility_count_any_field", "sum"))
            .sort_values("mentions", ascending=False).to_string())
    print("\nkey services (terms / mentions):")
    s = d[d["kind"] == "service_mention"].groupby("service").agg(
        terms=("term", "size"), mentions=("facility_count_any_field", "sum")).sort_values("mentions", ascending=False)
    print(s.to_string())
    print(f"\nmapping changes vs existing checked_service: {len(changes)} terms")

# COMMAND ----------

# MAGIC %md ## 2. Service table — DRAFT, needs review

# COMMAND ----------

"""Model 1 service table (DRAFT, step 2 of the PRD - Teammate A to review with someone medical).

For each of the 18 key services:
  prereqs        groups of things a facility needs before the claim is believable.
                 Every group must be found somewhere in the record (AND); inside a group any
                 pattern is enough (OR). Example: cardiac surgery needs (cath lab OR cardiothoracic
                 surgeon) AND (an ICU).
  allowed_types  facility types that can plausibly run the service. Anything else is capped.
  min_beds       if the bed count is known and below this, the claim is capped. None = no rule.
  detail         words that count as "specific detail" for this service (named procedures,
                 machine types). Counts like "20 machines" / "6-bed" are detected for every service.
  checklist      what the coordinator asks on the call.
  equipment_is_prereq  for machine-type services, being listed in the equipment field counts as the prerequisite.

Every value here is a starting guess. Changing a value here is a rules change: bump
RULES_VERSION in model1_evidence.py so old and new outputs are never mixed up.
"""

HOSPITALS = {"hospital", "nursing_home"}
HOSPITALS_AND_CLINICS = {"hospital", "nursing_home", "clinic"}

SERVICE_TABLE = {
    "dialysis": dict(
        equipment_is_prereq=True,
        prereqs=[("dialysis machines or nephrology",
                  r"dialysis (?:machine|equipment)|haemodialysis machine|hemodialysis machine|dialy[sz]ers?\b|dialysis unit|"
                  r"dialysis cent|\bro plant|water treatment|nephrolog|fresenius|nipro|nikkiso|b\.? ?braun|surdial|sundial")],
        allowed_types=HOSPITALS_AND_CLINICS, min_beds=None,
        detail=r"\bhdf\b|haemodiafiltration|hemodiafiltration|\bsled\b|\bcrrt\b|\bcapd\b|av fistula|bicarbonate",
        checklist="Is the dialysis unit running this week? How many machines? Walk-in or referral only? Cost per session?"),
    "peritoneal_dialysis": dict(
        prereqs=[("nephrology", r"nephrolog")],
        allowed_types=HOSPITALS_AND_CLINICS, min_beds=None, detail=r"\bcapd\b|\bapd\b|automated peritoneal",
        checklist="Do you start and follow up peritoneal dialysis (CAPD) patients? Is a nephrologist on staff?"),
    "icu": dict(
        prereqs=[("ICU equipment or intensivist",
                  r"ventilator|multipara|monitor|defibrillat|intensivist|intensive care specialist|critical care|anaesthe|anesthe|crash cart")],
        allowed_types=HOSPITALS, min_beds=10, detail=r"\bmicu\b|\bsicu\b|\bccu\b|intensivist|level (?:ii|iii|2|3)\b",
        checklist="How many ICU beds, and are any free now? Is there a doctor in the ICU at night? Ventilators?"),
    "nicu": dict(
        prereqs=[("neonatologist, paediatrics or newborn equipment",
                  r"neonatolog|incubator|radiant warmer|phototherapy|bubble cpap|\bcpap\b|paediatric|pediatric")],
        allowed_types=HOSPITALS, min_beds=10, detail=r"level (?:ii|iii|2|3)\b|incubator|radiant warmer|phototherapy|bubble cpap|surfactant",
        checklist="What NICU level? Can you take a baby under 1.5 kg? Is a neonatologist or paediatrician there 24 hours?"),
    "picu": dict(
        prereqs=[("paediatrics", r"paediatric|pediatric")],
        allowed_types=HOSPITALS, min_beds=10, detail=r"ventilator|intensivist",
        checklist="Do you admit children to an intensive care unit? Paediatric ventilators?"),
    "c_section": dict(
        prereqs=[("obstetrics or operating theatre",
                  r"obstetric|gynaec|gynec|maternity|operation theat|operating theat|\bot\b|labou?r room|anaesthe|anesthe")],
        allowed_types=HOSPITALS_AND_CLINICS, min_beds=None, detail=r"\blscs\b|emergency (?:c[- ]section|caesar)|blood bank",
        checklist="Can you do an emergency C-section at night? Is an anaesthetist available 24 hours? Blood on site?"),
    "cardiac_surgery": dict(
        prereqs=[("cath lab or cardiothoracic surgeon",
                  r"cath ?lab|cardiothoracic|\bctvs\b|heart[- ]lung machine|perfusion|cardiac surgeon"),
                 ("ICU", r"\bicu\b|intensive care|critical care|\bccu\b|\bcticu\b")],
        allowed_types=HOSPITALS, min_beds=30,
        detail=r"\bcabg\b|valve (?:replacement|repair)|\biabp\b|intra[- ]aortic|\becmo\b|heart[- ]lung machine|beating heart|\bopcab\b",
        checklist="Which heart operations were done here in the last month? Is the cardiac surgeon on staff or visiting?"),
    "cath_lab_angioplasty": dict(
        prereqs=[("cardiology", r"cardiolog")],
        allowed_types=HOSPITALS, min_beds=20,
        detail=r"\bptca\b|primary angioplasty|flat panel|biplane|\bivus\b|\boct\b|rotablat|drug[- ]eluting|stent",
        checklist="Is the cath lab open 24 hours for heart attacks? Is the interventional cardiologist on staff?"),
    "mri": dict(
        equipment_is_prereq=True,
        prereqs=[("radiology or MRI machine", r"radiolog|mri (?:machine|scanner)|magnetic resonance")],
        allowed_types=HOSPITALS_AND_CLINICS | {"unknown"}, min_beds=None,
        detail=r"\btesla\b|\b\d(?:\.\d)?\s?t\b|siemens|philips|\bge\b|wipro|toshiba|canon",
        checklist="Is the MRI working today? Waiting time? Is a radiologist reporting on site?"),
    "pet_ct": dict(
        equipment_is_prereq=True,
        prereqs=[("nuclear medicine or PET machine", r"nuclear medicine|pet[- ]?ct (?:machine|scanner)|cyclotron")],
        allowed_types=HOSPITALS_AND_CLINICS | {"unknown"}, min_beds=None, detail=r"siemens|philips|\bge\b|\bpsma\b|\bfdg\b",
        checklist="Is the PET-CT working? Which scans (FDG, PSMA)? Waiting time?"),
    "ct_scan": dict(
        equipment_is_prereq=True,
        prereqs=[("radiology or CT machine", r"radiolog|ct (?:machine|scanner)|computed tomograph")],
        allowed_types=HOSPITALS_AND_CLINICS | {"unknown"}, min_beds=None,
        detail=r"slice|siemens|philips|\bge\b|toshiba|canon|contrast",
        checklist="Is the CT scanner working now? Open at night? How fast is the report?"),
    "trauma_center": dict(
        prereqs=[("emergency", r"emergency|casualty"),
                 ("surgery", r"orthopa?ed|ortho\b|neurosurg|general surg|operation theat|operating theat|\bot\b")],
        allowed_types=HOSPITALS, min_beds=30, detail=r"level (?:i|ii|1|2)\b|polytrauma|\batls\b|golden hour",
        checklist="Do you take road-accident patients 24 hours? Is a surgeon and an orthopaedic surgeon there at night? CT on site?"),
    "emergency_24x7": dict(
        prereqs=[("24-hour cover, casualty or ambulance", r"24 ?(?:x|/|hours|hrs|7)|round[- ]the[- ]clock|casualty|ambulance|emergencymedicine")],
        allowed_types=HOSPITALS_AND_CLINICS, min_beds=None, detail=r"triage|\bacls\b|\batls\b|resuscitation|casualty medical officer",
        checklist="Is a doctor in the emergency room at night? Do you have an ambulance?"),
    "burn_care": dict(
        prereqs=[("plastic surgery or burns ward", r"plastic surg|burns? (?:ward|unit)")],
        allowed_types=HOSPITALS, min_beds=20, detail=r"\b\d+\s?%|skin graft|burns? icu|dressing",
        checklist="Do you admit burn patients? Is there a burns ward or ICU? Plastic surgeon on staff?"),
    "chemotherapy": dict(
        prereqs=[("oncology", r"oncolog|cancer")],
        allowed_types=HOSPITALS_AND_CLINICS, min_beds=None, detail=r"day ?care|biosafety|laminar|immunotherap|targeted therap",
        checklist="Is a medical oncologist on staff? Can treatment start this week? Day-care chairs?"),
    "radiotherapy": dict(
        prereqs=[("radiation oncology or a treatment machine",
                  r"radiation oncolog|radiotherap|radiation therap|linac|linear accelerator|cobalt|brachy|varian|elekta|tomotherapy|cyberknife")],
        allowed_types=HOSPITALS_AND_CLINICS, min_beds=None,
        detail=r"\bimrt\b|\bigrt\b|\bsbrt\b|\bsrs\b|\bvmat\b|linac|linear accelerator|varian|elekta|tomotherapy|cyberknife|brachy",
        checklist="Which radiotherapy machine is working? Waiting list length? Radiation oncologist on staff?"),
    "blood_bank": dict(
        prereqs=[("pathology or transfusion", r"patholog|transfusion|blood component|laborator|\blab\b")],
        allowed_types=HOSPITALS_AND_CLINICS, min_beds=None, detail=r"component|apheresis|licen[cs]ed|\bnat\b|platelet|plasma",
        checklist="Is the blood bank on site and licensed? Is the needed blood group usually in stock?"),
    "ventilator": dict(
        equipment_is_prereq=True,
        prereqs=[("ICU or anaesthesia", r"\bicu\b|intensive care|critical care|anaesthe|anesthe")],
        allowed_types=HOSPITALS, min_beds=None, detail=r"\b\d+\s?ventilators?\b|\bniv\b|bipap|hamilton|dr[aä]ger|mindray",
        checklist="How many ventilators, and are any free now? Who manages ventilated patients at night?"),
}


# Single-specialty facilities, recognised by name. They are typed "hospital" but cannot back heavy
# services outside their field (an eye hospital doing heart valve surgery). Value = services they CAN back.
SINGLE_SPECIALTY = {
    r"\beye\b|netra|nethra|ophthalm|vision|lasik": {"emergency_24x7"},
    r"dental|dentist|\bteeth\b|\btooth\b|orthodont": set(),
    r"\bskin\b|derma|\bhair\b|cosmetic|aesthetic": set(),
    r"\bent\b|ear,? nose": {"emergency_24x7", "ct_scan"},
    r"\bivf\b|fertility|test tube": {"nicu", "c_section", "icu", "blood_bank"},
}
# ...unless the name also says it is a general / mixed facility.
MIXED_FACILITY = (r"multi|super|general|children|child\b|women|maternity|nursing home|medical college|"
                  r"\bhospitals? (?:and|&)|(?:and|&) (?:\w+ )?hospital")

# COMMAND ----------

# MAGIC %md ## 3. Evidence-level rules

# COMMAND ----------

"""Model 1 - Service Evidence Levels (PRD step 3).

For every facility x key service that is mentioned anywhere, give an evidence level
(SPECIFIC / LISTED / PASSING), the quote behind it, and the plain-English reasons.
It does NOT decide whether the facility really offers the service; it measures how
strongly the record supports the claim.

  python3 model1_evidence.py gold_facility_search_services.csv            # writes model1_evidence_levels.csv
  python3 model1_evidence.py gold_facility_search_services.csv --out x.csv

In a Databricks notebook (pandas runs fine on serverless):
  import model1_evidence as m1
  levels = m1.score_file("/Volumes/workspace/datathon_2026/csv_data/gold_facility_search_services.csv")
  # the master-notebook owner decides whether/where to save `levels`

How a level is decided (same order as the PRD):
  1. Collect every mention: structured fields (procedures / equipment / capabilities, using the
     service-term rules in normalize_services.py), description sentences, and specialty hints.
  2. Throw out weak mentions: keyword lists (5+ items, no detail), "done elsewhere" wording
     (visiting, referral, tie-up, camp), copied text (same long sentence at 20+ different
     organisations), negated text ("no ICU").
  3. Start level:  no usable structured mention -> PASSING;  otherwise LISTED.
  4. Size / type cap: facility type or bed count can't support the service -> PASSING.
  5. Raise to SPECIFIC only if 2+ of: mentioned in 2+ structured fields / specific detail
     (counts, named machines or procedures) / an official website (not JustDial, Practo...).
  6. Cap at LISTED: no prerequisite found anywhere in the record, or marketing filler with no detail.
"""

import argparse
import collections
import re
import sys
from pathlib import Path

import pandas as pd



RULES_VERSION = "v1"

# ----------------------------------------------------------------------------- thresholds (tune once)
LIST_ITEMS = 5          # a sentence naming this many things with no detail = keyword list
COPY_ORGS = 20          # the same long sentence at this many different organisations = copied text
COPY_MIN_WORDS = 8      # shorter phrases ("ct scanner") repeat naturally and are not "copied"

FIELDS = ["procedures", "equipment", "capabilities"]
LEVELS = ["SPECIFIC", "LISTED", "PASSING"]
SENT = re.compile(r"(?<=[.!?])\s+|\n+")

# ----------------------------------------------------------------------------- wording patterns
COUNT_DETAIL = re.compile(
    r"\b\d+\s?(?:-\s?)?(?:[a-z]+[\s-]){0,2}(?:bed(?:ded|s)?|machines?|units?|stations?|chairs?|slices?|incubators?|"
    r"ventilators?|warmers?|cots?|tesla|cath ?labs?|operation theatres?|ots?)\b", re.I)
ELSEWHERE = re.compile(
    r"visiting|\bon[- ]call basis\b|referr(?:al|ed|ing)? (?:to|for)\b|refers? (?:patients )?(?:to|for)\b|"
    r"tie[- ]?ups?\b|tied up|\bcamps?\b|in association with|in collaboration with|"
    r"partner(?:ed|ship)? (?:with|hospital)|arranged (?:at|through)|outsourc|through (?:a |our )?partner", re.I)
FILLER = re.compile(
    r"world[- ]class|state[- ]of[- ]the[- ]art|all (?:types|kinds) of|\bbest\b|top[- ]notch|ultra[- ]?modern|"
    r"cutting[- ]edge|\blatest\b|renowned|\bpremier\b|\bleading\b|excellence|complete range", re.I)
# scraper notes about its own extraction ("does not list ... in the provided content") are not claims
META = re.compile(r"provided (?:content|text|information)|(?:does|do) not (?:list|mention|specify|state)|"
                  r"not (?:specified|mentioned|stated|listed)|no (?:information|details|mention)|unclear whether", re.I)
NEGATION = re.compile(
    r"\b(?:no(?! other| extra| additional)|not available|not offered|not provided|does not|doesn't|do not|unavailable|lacks?|without)\b[^.]{0,30}$", re.I)

AGGREGATORS = re.compile(
    r"justdial|indiamart|practo|sulekha|lybrate|credihealth|medindia|eka\.care|business\.site|g\.page|"
    r"goo\.gl|bit\.ly|linktr\.ee|facebook|instagram|youtube|twitter|x\.com|linkedin|wa\.me|google\.|"
    r"quikr|tripadvisor|healthgrades|apollo247|1mg|mfine|docprime|clinicspots|vaidam|joonsquare", re.I)


def domain(website: str) -> str:
    w = re.sub(r'[\[\]"\' ]', "", str(website).lower()).split(",")[0]
    w = re.sub(r"^https?://", "", w)
    return re.sub(r"^www\.", "", w).split("/")[0]


def is_keyword_list(text: str) -> bool:
    parts = [p for p in re.split(r",|;|/|\band\b|\|", text) if p.strip()]
    return len(parts) >= LIST_ITEMS


def has_detail(svc: str, text: str) -> bool:
    return bool(COUNT_DETAIL.search(text) or re.search(SERVICE_TABLE[svc]["detail"], text, re.I))


def is_negated(svc: str, text: str) -> bool:
    inc = SERVICES[svc][0]
    low = text.lower()
    return any(NEGATION.search(low[:m.start()]) for m in re.finditer(inc, low))


def norm_text(t: str) -> str:
    return re.sub(r"\W+", " ", str(t).lower()).strip()


# ----------------------------------------------------------------------------- copied text
def copy_counts(df: pd.DataFrame) -> dict[str, int]:
    """Long sentence -> number of different organisations (website domain, else name) using it."""
    owners: dict[str, set] = collections.defaultdict(set)
    for r in df.to_dict("records"):
        owner = domain(r.get("website", "")) or norm_text(r.get("name", ""))
        pieces = []
        for f in FIELDS:
            pieces += [x for x in str(r.get(f, "")).split("|")]
        pieces += SENT.split(str(r.get("description", "")))
        for p in pieces:
            n = norm_text(p)
            if len(n.split()) >= COPY_MIN_WORDS:
                owners[n].add(owner)
    return {k: len(v) for k, v in owners.items() if len(v) >= COPY_ORGS}


# ----------------------------------------------------------------------------- mentions
def collect_mentions(row: dict) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = collections.defaultdict(list)
    for field in FIELDS:
        for item in filter(None, (x.strip() for x in str(row.get(field, "")).split("|"))):
            kind, svc, _ = classify(item.lower(), clean(item), field)
            if kind == "service_mention" and svc in SERVICE_TABLE and not META.search(item):
                found[svc].append({"field": field, "text": item})
    for sent in SENT.split(str(row.get("description", ""))):
        s = sent.lower()
        if not s.strip() or META.search(s):
            continue
        for svc, (inc, exc) in SERVICES.items():
            if re.search(inc, s) and not re.search(exc, s):
                found[svc].append({"field": "description", "text": sent.strip()})
    return found


def specialty_hints(row: dict) -> dict[str, str]:
    out = {}
    for c in str(row.get("specialty_codes", "")).split("|"):
        c = c.strip().lower()
        if c in SPECIALTY_HINTS:
            out.setdefault(SPECIALTY_HINTS[c], c)
    return out


def prereqs(svc: str, record_text: str) -> tuple[list[str], list[str]]:
    found, missing = [], []
    for label, pattern in SERVICE_TABLE[svc]["prereqs"]:
        (found if re.search(pattern, record_text, re.I) else missing).append(label)
    return found, missing


def to_number(x) -> float | None:
    try:
        v = float(str(x).strip())
        return v if v > 0 else None
    except ValueError:
        return None


# ----------------------------------------------------------------------------- scoring
def score_one(svc: str, mentions: list[dict], hint: str | None, row: dict, copied: dict[str, int]) -> dict:
    reasons, caps = [], []
    for m in mentions:
        m["list"] = m["field"] != "equipment" and is_keyword_list(m["text"]) and not has_detail(svc, m["text"])
        m["elsewhere"] = bool(ELSEWHERE.search(m["text"]))
        m["copied"] = norm_text(m["text"]) in copied
        m["negated"] = is_negated(svc, m["text"])
        m["detail"] = has_detail(svc, m["text"])
        m["filler"] = bool(FILLER.search(m["text"]))
        m["ok"] = not (m["list"] or m["elsewhere"] or m["copied"] or m["negated"])

    struct = [m for m in mentions if m["field"] in FIELDS]
    good_struct = [m for m in struct if m["ok"]]
    good_any = [m for m in mentions if m["ok"]]
    n_fields = len({m["field"] for m in mentions})

    record_text = " ".join(str(row.get(f, "")) for f in
                           FIELDS + ["description", "specialty_codes", "specialty_labels", "name"])
    pre_found, pre_missing = prereqs(svc, record_text)
    if pre_missing and SERVICE_TABLE[svc].get("equipment_is_prereq") and any(m["field"] == "equipment" for m in good_struct):
        pre_found, pre_missing = pre_found + ["listed as equipment"], []

    # -- step 3: starting level
    if not mentions:
        level = "PASSING"
        caps.append("specialty_hint_only")
        reasons.append(f"only implied by the specialty '{hint}', never named")
    elif not struct:
        level = "PASSING"
        caps.append("description_only")
        reasons.append("named only in the free-text description, not in any structured field")
    elif not good_struct:
        level = "PASSING"
        for flag, cap, why in [("list", "keyword_list", f"only inside a list of {LIST_ITEMS}+ services with no detail"),
                               ("elsewhere", "done_elsewhere", "wording suggests it is done elsewhere (visiting / referral / tie-up / camp)"),
                               ("copied", "copied_text", f"the only structured text is copied word-for-word at {COPY_ORGS}+ other organisations"),
                               ("negated", "contradicted", "the record says it is NOT available")]:
            if any(m[flag] for m in struct):
                caps.append(cap)
                reasons.append(why)
    else:
        level = "LISTED"
        reasons.append(f"named in structured field(s): {', '.join(sorted({m['field'] for m in good_struct}))}")

    if any(m["negated"] for m in mentions) and "contradicted" not in caps:
        level = "PASSING"
        caps.append("contradicted")
        neg = next(m for m in mentions if m["negated"])
        reasons.append(f"part of the record says it is NOT available ({neg['field']}: \"{neg['text'][:120]}\")")

    # -- step 4: facility type / size
    rule = SERVICE_TABLE[svc]
    ftype = str(row.get("facility_type", "")).strip().lower() or "unknown"
    beds = to_number(row.get("capacity_beds"))
    if ftype not in rule["allowed_types"]:
        level = "PASSING"
        caps.append("facility_type_cant_support")
        reasons.append(f"a {ftype} is not the kind of facility that runs {svc.replace('_', ' ')}")
    elif not re.search(MIXED_FACILITY, str(row.get("name", "")), re.I) and any(
            re.search(pat, str(row.get("name", "")), re.I) and svc not in ok
             for pat, ok in SINGLE_SPECIALTY.items()):
        level = "PASSING"
        caps.append("single_specialty_facility")
        reasons.append(f"the name says it is a single-specialty facility; {svc.replace('_', ' ')} is outside that field")
    elif rule["min_beds"] and beds is not None and beds < rule["min_beds"]:
        level = "PASSING"
        caps.append("too_few_beds")
        reasons.append(f"{int(beds)} beds is below the {rule['min_beds']} needed for {svc.replace('_', ' ')}")

    # -- steps 5-6: raise to SPECIFIC, or cap at LISTED
    if level == "LISTED":
        multi = len({m["field"] for m in good_struct}) >= 2
        detail = any(m["detail"] for m in good_any)
        site = domain(row.get("website", ""))
        official = bool(site) and not AGGREGATORS.search(site)
        signals = [s for s, on in [("in 2+ structured fields", multi), ("specific detail", detail),
                                   (f"official website ({site})", official)] if on]
        if signals:
            reasons.append("raise signals: " + "; ".join(signals))
        if pre_missing:
            caps.append("no_prerequisite")
            reasons.append("no sign anywhere of: " + "; ".join(pre_missing))
        elif all(m["filler"] for m in good_struct) and not detail:
            caps.append("marketing_filler")
            reasons.append("marketing wording with no concrete detail behind it")
        elif len(signals) >= 2:
            level = "SPECIFIC"
        else:
            reasons.append("needs 2 raise signals for Specific")

    # -- best quote: detailed structured > equipment > procedures > capabilities > description
    order = {"equipment": 0, "procedures": 1, "capabilities": 2, "description": 3}
    pool = good_any or mentions
    if pool:
        best = sorted(pool, key=lambda m: (not m["detail"], m["field"] not in FIELDS, order[m["field"]], -len(m["text"])))[0]
        quote, quote_field = best["text"][:300], best["field"]
    else:
        quote, quote_field = f"specialty: {hint}", "specialty_codes"

    # raw wording signals, for the probability model (model1_probability.py)
    site = domain(row.get("website", ""))
    feats = {
        "f_n_mentions": len(mentions),
        "f_n_good_struct_fields": len({m["field"] for m in good_struct}),
        "f_in_description": any(m["field"] == "description" for m in mentions),
        "f_description_only": bool(mentions) and not struct,
        "f_detail": any(m["detail"] for m in good_any),
        "f_official_site": bool(site) and not AGGREGATORS.search(site),
        "f_prereq_ok": not pre_missing,
        "f_keyword_list": any(m["list"] for m in mentions),
        "f_done_elsewhere": any(m["elsewhere"] for m in mentions),
        "f_copied": any(m["copied"] for m in mentions),
        "f_contradicted": any(m["negated"] for m in mentions),
        "f_filler_only": bool(good_struct) and all(m["filler"] for m in good_struct),
        "f_cant_support": bool({"facility_type_cant_support", "single_specialty_facility", "too_few_beds"} & set(caps)),
        "f_specialty_hint_too": hint is not None,
        "f_beds_known": beds is not None,
    }
    return {"unique_id": row.get("unique_id"), "service": svc, "evidence_level": level,
            "quote": quote, "quote_field": quote_field, "n_fields": n_fields,
            "prereqs_found": "; ".join(pre_found), "caps_applied": "|".join(dict.fromkeys(caps)),
            "reasons": " | ".join(reasons), "rules_version": RULES_VERSION, **feats}


def score_frame(df: pd.DataFrame, with_features: bool = False) -> pd.DataFrame:
    df = df.fillna("").astype(str)
    copied = copy_counts(df)
    rows = []
    for r in df.to_dict("records"):
        found = collect_mentions(r)
        hints = specialty_hints(r)
        for svc in SERVICE_TABLE:
            if found.get(svc) or svc in hints:
                rows.append(score_one(svc, found.get(svc, []), hints.get(svc), r, copied))
    cols = ["unique_id", "service", "evidence_level", "quote", "quote_field", "n_fields",
            "prereqs_found", "caps_applied", "reasons", "rules_version"]
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=cols)
    out["f_services_at_facility"] = out.groupby("unique_id").service.transform("size")
    return out if with_features else out[cols]


def score_file(path: str, with_features: bool = False) -> pd.DataFrame:
    return score_frame(pd.read_csv(path, dtype=str, keep_default_na=False), with_features)


def summary(levels: pd.DataFrame) -> str:
    t = pd.crosstab(levels.service, levels.evidence_level).reindex(columns=LEVELS, fill_value=0)
    t["total"] = t.sum(axis=1)
    caps = levels.caps_applied[levels.caps_applied != ""].str.split("|").explode().value_counts()
    return (f"{len(levels)} facility x service rows, {levels.unique_id.nunique()} facilities\n\n"
            f"{t.sort_values('total', ascending=False).to_string()}\n\n"
            f"Overall: {levels.evidence_level.value_counts().reindex(LEVELS).to_dict()}\n\n"
            f"Caps fired:\n{caps.to_string()}")


def main_model1_evidence():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("--out", default="model1_evidence_levels.csv")
    a = ap.parse_args()
    if Path(a.out).resolve() == Path(a.input).resolve():
        raise SystemExit("refusing to overwrite the input file")
    levels = score_file(a.input)
    levels.to_csv(a.out, index=False)
    print(f"Wrote {a.out}  (rules {RULES_VERSION})\n")
    print(summary(levels))

# COMMAND ----------

# MAGIC %md ## 4. Directory check helpers

# COMMAND ----------

"""PRD step 5 result: confirmation rate per evidence level, with a 95% range.

  python3 validation_report.py validation_calls.csv validation_key.csv

Only "confirmed" and "not_offered" count. "cant_tell" (no answer, wrong number) is reported
separately and left out of the rate. The 95% range is the Wilson interval, which behaves well
for small samples like 40.
Success (PRD): SPECIFIC confirms clearly more often than PASSING and the two ranges don't overlap.
"""

import math
import sys

import pandas as pd


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def main_validation_report(calls_path, key_path):
    calls = pd.read_csv(calls_path, dtype=str, keep_default_na=False)
    key = pd.read_csv(key_path, dtype=str)
    rcol = [c for c in calls.columns if c.startswith("result")][0]
    df = key.merge(calls[["check_id", rcol]], on="check_id")
    df["result"] = df[rcol].str.strip().str.lower().str.replace(" ", "_")

    lines, ranges = [], {}
    for level in ["SPECIFIC", "LISTED", "PASSING"]:
        g = df[df.evidence_level == level]
        if g.empty:
            continue
        k = (g.result == "confirmed").sum()
        n = k + (g.result == "not_offered").sum()
        unk = len(g) - n
        lo, hi = wilson(k, n)
        ranges[level] = (lo, hi)
        lines.append(f"{level:<9} {k}/{n} confirmed = {k / n:.0%} (95% range {lo:.0%}-{hi:.0%})   can't tell / not done: {unk}"
                     if n else f"{level:<9} no answered checks yet ({unk} pending)")
    print("\n".join(lines))
    if "SPECIFIC" in ranges and "PASSING" in ranges:
        sep = ranges["SPECIFIC"][0] > ranges["PASSING"][1]
        print("\nRanges separate: YES - the levels mean something." if sep else
              "\nRanges overlap: fix the rules once (bump RULES_VERSION), then check a fresh sample "
              "(make_validation_sample.py --exclude validation_key.csv).")

# COMMAND ----------

"""Automatic validation against the government hospital directory (PRD validation source #2).

  python3 directory_check.py model1_evidence_levels.csv gold_facility_search_services.csv hospital_directory.csv

For facilities matched to the directory (STRONG or PROBABLE), checks whether the directory's own
Specialties / Facilities text names each service. The scraper never used the directory, so it is an
independent source. Writes directory_check_rows.csv and prints the agreement rate per level.

Read the result carefully: "not in the directory" does NOT mean "not offered" (directory lists
are short), so this measures agreement, not truth. The pattern that matters is whether
SPECIFIC agrees more often than PASSING.
"""
import re
import sys

import pandas as pd


# what the directory would write for each service (specialty names count)
DIR_TERMS = {
    "dialysis": r"dialys|nephrolog", "peritoneal_dialysis": r"dialys|nephrolog",
    "icu": r"\bicu\b|intensive care|critical care|\bmicu\b|\bsicu\b|\biccu\b",
    "nicu": r"nicu|neo ?-?natolog|neonat", "picu": r"picu|p(?:a)?ediatric intensive|neonat",
    "c_section": r"obstet|gyn(?:a)?ecolog|maternity|caesar|lscs",
    "cardiac_surgery": r"cardio ?-?thoracic|\bctvs\b|\bcvts\b|cardiac surg|heart surg|bypass",
    "cath_lab_angioplasty": r"cardiolog|cath|angioplast", "mri": r"\bmri\b|magnetic|radiolog|imaging",
    "ct_scan": r"\bct\b|computed tomog|radiolog|imaging", "pet_ct": r"\bpet\b|nuclear medicine",
    "trauma_center": r"trauma|accident|casualty", "emergency_24x7": r"emergency|casualty|24 ?hours|trauma|accident",
    "burn_care": r"burn|plastic", "chemotherapy": r"oncolog|chemo|cancer",
    "radiotherapy": r"radiation|radiotherap|linac", "blood_bank": r"blood", "ventilator": r"ventilat|\bicu\b|intensive|critical care",
}


def main_directory_check(levels_p, fac_p, dir_p):
    d = pd.read_csv(dir_p, dtype=str, keep_default_na=False, low_memory=False)
    d["dir_services"] = (d.Specialties.where(d.Specialties != "0", "") + " ; " +
                         d.Facilities.where(d.Facilities != "0", "")).str.replace(r"\\n", " ", regex=True).str.strip(" ;")
    d = d[d.dir_services != ""][["Sr_No", "Hospital_Name", "dir_services"]]
    g = pd.read_csv(fac_p, dtype=str, keep_default_na=False)
    g = g[g.directory_match.isin(["STRONG", "PROBABLE"])][["unique_id", "name", "directory_id", "directory_match"]]
    g = g.merge(d, left_on="directory_id", right_on="Sr_No")
    lv = pd.read_csv(levels_p, dtype=str, keep_default_na=False)
    x = lv[~lv.caps_applied.str.contains("specialty_hint_only")].merge(g, on="unique_id")
    x["directory_lists_it"] = [bool(re.search(DIR_TERMS[s], t, re.I)) for s, t in zip(x.service, x.dir_services)]
    x[["unique_id", "name", "Hospital_Name", "directory_match", "service", "evidence_level", "quote",
       "directory_lists_it", "dir_services"]].to_csv("/tmp/directory_check_rows.csv", index=False)

    print(f"{x.unique_id.nunique()} facilities, {len(x)} facility x service claims checked against the directory\n")
    for level in ["SPECIFIC", "LISTED", "PASSING"]:
        s = x[x.evidence_level == level].directory_lists_it
        k, n = int(s.sum()), len(s)
        lo, hi = wilson(k, n)
        print(f"{level:<9} directory agrees {k}/{n} = {k / n:.0%}  (95% range {lo:.0%}-{hi:.0%})")
    print("\nBy service (agree rate SPECIFIC vs LISTED, n):")
    t = x.groupby(["service", "evidence_level"]).directory_lists_it.agg(["mean", "size"]).unstack()
    print(t.round(2).to_string())

# COMMAND ----------

# MAGIC %md ## 5. Percentage model

# COMMAND ----------

"""Model 1, part 2: turn how a service is presented into a percentage.

  python3 model1_probability.py gold_facility_search_services.csv hospital_directory.csv

What the percentage means (say it exactly like this):
  "Chance that the government hospital directory also lists this service for this hospital,
   given how the service is presented in our data."
It is NOT the chance the hospital truly offers it: directory lists are short, so the numbers
run low for everyone. Use it to compare and rank hospitals, not as a promise.

How it works:
  1. Model 1's rules read every mention and record its wording signals (how many fields name it,
     concrete detail, keyword list, "visiting/referral", prerequisites, copied text, ...).
  2. Answer key: for the hospitals that match the government directory, does the directory's own
     service list name the service? (~850 claims). The scraper never used the directory.
  3. A logistic regression (a standard statistics model that turns weighted signals into a
     0-100% chance) learns which signals go with "the directory lists it".
  4. Fair test: 5 rounds, each time hiding a fifth of the hospitals, training on the rest and
     scoring the hidden ones. Reported against the plain Specific/Listed/Passing levels.
  5. Trained on all the labelled claims, then applied to every claim -> directory_confirm_pct.
     Rows that are only a specialty code (never a real claim) get no percentage.

Writes model1_evidence_with_pct.csv. Never changes the input files.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


MODEL_VERSION = "p1"
LEVEL_RANK = {"PASSING": 0, "LISTED": 1, "SPECIFIC": 2}

# plain-English names for the printout
NAMES = {
    "f_n_good_struct_fields": "named in more structured fields", "f_detail": "concrete detail (counts, machines, procedures)",
    "f_official_site": "own website (not JustDial/Practo)", "f_prereq_ok": "prerequisites present",
    "f_keyword_list": "buried in a keyword list", "f_done_elsewhere": "visiting / referral / tie-up wording",
    "f_copied": "copied text", "f_contradicted": "contradicted elsewhere", "f_filler_only": "marketing filler only",
    "f_cant_support": "facility type/size can't support it", "f_description_only": "description only",
    "f_in_description": "also in the description", "f_specialty_hint_too": "matching specialty code too",
    "f_beds_known": "bed count published", "log_mentions": "more mentions", "log_services": "facility claims many services",
}


def directory_labels(fac: pd.DataFrame, directory: pd.DataFrame) -> pd.DataFrame:
    d = directory.copy()
    d["dir_services"] = (d.Specialties.where(d.Specialties != "0", "") + " ; " +
                         d.Facilities.where(d.Facilities != "0", "")).str.replace(r"\\n", " ", regex=True).str.strip(" ;")
    d = d[d.dir_services != ""][["Sr_No", "dir_services"]]
    g = fac[fac.directory_match.isin(["STRONG", "PROBABLE"])][["unique_id", "directory_id"]]
    return g.merge(d, left_on="directory_id", right_on="Sr_No")[["unique_id", "dir_services"]]


def design(rows: pd.DataFrame, services: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=rows.index)
    for c in [c for c in rows.columns if c.startswith("f_") and c not in ("f_n_mentions", "f_services_at_facility")]:
        X[c] = rows[c].astype(float)
    X["log_mentions"] = np.log1p(rows.f_n_mentions.astype(float))
    X["log_services"] = np.log1p(rows.f_services_at_facility.astype(float))
    for s in services[1:]:                                   # one column per service (first = baseline)
        X[f"svc_{s}"] = (rows.service == s).astype(float)
    return X


def run(facilities_path: str, directory_path: str, out: str | None = None) -> pd.DataFrame:
    """Score every claim and add directory_confirm_pct. Writes `out` only if given."""
    fac = pd.read_csv(facilities_path, dtype=str, keep_default_na=False)
    directory = pd.read_csv(directory_path, dtype=str, keep_default_na=False, low_memory=False)
    rows = score_frame(fac, with_features=True)
    claims = rows[~rows.caps_applied.str.contains("specialty_hint_only")].copy()
    services = sorted(SERVICE_TABLE)

    lab = claims.merge(directory_labels(fac, directory), on="unique_id")
    lab["y"] = [int(bool(re.search(DIR_TERMS[s], t, re.I))) for s, t in zip(lab.service, lab.dir_services)]
    X, y, groups = design(lab, services), lab.y.values, lab.unique_id.values
    print(f"Answer key: {len(lab)} claims at {lab.unique_id.nunique()} hospitals; directory lists the service for {y.mean():.0%}\n")

    # ---- fair test: hospitals in the test fold are never seen in training
    pred = np.zeros(len(lab))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        clf = LogisticRegression(C=0.5, max_iter=2000).fit(X.iloc[tr], y[tr])
        pred[te] = clf.predict_proba(X.iloc[te])[:, 1]
    rank = lab.evidence_level.map(LEVEL_RANK).values
    print("Fair test on hospitals the model never saw (AUC: 0.5 = coin flip, 1.0 = perfect ranking):")
    print(f"  plain levels (Specific > Listed > Passing): AUC {roc_auc_score(y, rank):.2f}")
    print(f"  percentage model:                           AUC {roc_auc_score(y, pred):.2f}   "
          f"(Brier {brier_score_loss(y, pred):.3f} vs {brier_score_loss(y, np.full(len(y), y.mean())):.3f} for guessing the average)")
    lab["pred"] = pred
    lab["band"] = pd.cut(lab.pred, [0, .3, .45, .6, 1], labels=["under 30%", "30-45%", "45-60%", "over 60%"])
    cal = lab.groupby("band", observed=True).agg(claims=("y", "size"), predicted=("pred", "mean"), actually_listed=("y", "mean"))
    print("\nDoes the percentage mean what it says? (held-out)")
    print(cal.assign(predicted=lambda t: (t.predicted * 100).round().astype(int).astype(str) + "%",
                     actually_listed=lambda t: (t.actually_listed * 100).round().astype(int).astype(str) + "%").to_string())

    # ---- final model on all labelled claims, applied to every claim
    clf = LogisticRegression(C=0.5, max_iter=2000).fit(X, y)
    coefs = pd.Series(clf.coef_[0], index=X.columns)
    wording = coefs[[c for c in coefs.index if not c.startswith("svc_")]].sort_values()
    print("\nWhat moves the percentage (wording signals; + raises, - lowers):")
    for c, v in wording.items():
        print(f"  {v:+.2f}  {NAMES.get(c, c)}")

    rows["directory_confirm_pct"] = pd.NA
    is_claim = ~rows.caps_applied.str.contains("specialty_hint_only")
    rows.loc[is_claim, "directory_confirm_pct"] = np.round(
        clf.predict_proba(design(rows[is_claim], services))[:, 1] * 100).astype(int)
    rows["pct_model_version"] = MODEL_VERSION
    keep = ["unique_id", "service", "evidence_level", "directory_confirm_pct", "quote", "quote_field", "n_fields",
            "prereqs_found", "caps_applied", "reasons", "rules_version", "pct_model_version"]
    result = rows[keep]
    if out:
        if Path(out).resolve() in (Path(facilities_path).resolve(), Path(directory_path).resolve()):
            raise SystemExit("refusing to overwrite an input file")
        result.to_csv(out, index=False)
    s = rows[is_claim].groupby("evidence_level").directory_confirm_pct.agg(lambda v: f"median {int(np.median(v.astype(int)))}%, range {int(v.min())}-{int(v.max())}%")
    print(f"\n{'Wrote ' + out + ': ' if out else ''}{is_claim.sum()} claims with a percentage ({(~is_claim).sum()} specialty-code-only rows left blank)")
    print(s.reindex(["SPECIFIC", "LISTED", "PASSING"]).to_string())
    return result


def main_model1_probability():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("facilities")
    ap.add_argument("directory")
    ap.add_argument("--out", default="model1_evidence_with_pct.csv")
    a = ap.parse_args()
    run(a.facilities, a.directory, a.out)

# COMMAND ----------

# MAGIC %md ## 6. Run it

# COMMAND ----------

result = run(FACILITIES, DIRECTORY)
display(result)

# COMMAND ----------

# MAGIC %md ## 7. Example: best-supported dialysis claims

# COMMAND ----------

fac = pd.read_csv(FACILITIES, dtype=str, keep_default_na=False)[["unique_id", "name", "city", "state", "phone_numbers"]]
show = result.merge(fac, on="unique_id")
display(show[show.service == "dialysis"].sort_values("directory_confirm_pct", ascending=False)[["name", "city", "evidence_level", "directory_confirm_pct", "quote", "reasons"]].head(20))

# COMMAND ----------

# MAGIC %md ## 8. (Master-notebook owner only) save the result

# COMMAND ----------

# spark.createDataFrame(result.astype({"directory_confirm_pct": "float"})).write.mode("overwrite").saveAsTable("workspace.datathon_2026.model1_evidence_levels")