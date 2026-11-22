# Databricks notebook source
# MAGIC %md
# MAGIC # Model 2 — Facility Capability (54 services, model c1)
# MAGIC Estimates whether a facility is the **kind** that runs a service, from its size, staff, type and specialties — never from how the service is worded.
# MAGIC Learns from the national government hospital directory (all India): for facilities matched to it, the directory's own service lists are the answer key. Services with too few examples are marked NOT_ASSESSED.
# MAGIC See "PRD: Model 2 — Facility Capability".
# MAGIC
# MAGIC Runs **after** the Model 1 notebook and on top of it: if Model 1's results table exists, this notebook compares against it and adds the combined `finder_label`.
# MAGIC Model 1 is never changed. **Run all.** Nothing is saved unless the last cell is un-commented.

# COMMAND ----------

# MAGIC %pip install scikit-learn --quiet

# COMMAND ----------

GOLD   = "/Volumes/workspace/datathon_2026/csv_data/gold_facility_search_services.csv"
DIRECTORY = "/Volumes/workspace/datathon_2026/csv_data/hospital_directory.csv"
MODEL1_TABLE = "workspace.datathon_2026.model1_evidence_levels"   # saved by the Model 1 notebook

# COMMAND ----------

# MAGIC %md ## 1. Model 2 code

# COMMAND ----------

"""Model 2 - Facility Capability (see "PRD: Model 2 - Facility Capability").

Estimates whether a facility is the KIND that runs a service, from what the facility is
(size, staff, type, specialties) and never from how the service is worded.

  python3 model2_capability.py gold_facility_search_services.csv hospital_directory.csv \
          [--model1 model1_evidence_with_pct.csv] [--out model2_capability.csv]

  gold file       our 9,969 facilities (the same file Model 1 reads)
  directory file  the national government hospital directory (all India). Its own Specialties /
                  Facilities lists are the answer key, for facilities matched to it (STRONG/PROBABLE).
  --model1        optional: Model 1's output. Only used to compare against Model 1 and to add the
                  combined finder label. Model 1 itself is never changed or retrained.

Output: one row per facility x service for all 54 services, with capability_band
(LIKELY / POSSIBLE / UNLIKELY / UNKNOWN; NOT_ASSESSED where the directory gives too few examples
to learn from), capability_pct, drivers, info_level.
"""

import argparse
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

MODEL_VERSION = "c1"
C_REG = 0.3                      # smaller = stronger regularisation (95 hospitals: keep it strong)
BANDS = (0.30, 0.60)
MIN_POS = 8                      # a service is assessed only with at least this many yes AND no examples             # UNLIKELY < 30% <= POSSIBLE < 60% <= LIKELY

# service -> specialty codes (from our specialty_codes) that relate to it. Services not listed have no related code.
RELATED = {
    "cardiac_surgery":      r"^(cardiacsurgery|cardiothoracicsurgery)$",
    "cath_lab_angioplasty": r"^(cardiology|interventionalcardiology)$",
    "chemotherapy":         r"^(medicaloncology|oncology|hematology|pediatrichematologyoncology)$",
    "radiotherapy":         r"^(radiationoncology)$",
    "burn_care":            r"^(plasticsurgery|reconstructiveplasticsurgery|burnandtraumaplasticsurgery)$",
    "nicu":                 r"^(neonatologyperinatalmedicine|pediatriccriticalcaremedicine)$",
    "trauma_center":        r"^(orthopedicsurgery|emergencymedicine|criticalcaremedicine|neurosurgery)$",
    "c_section":            r"^(gynecologyandobstetrics|obstetricsandmaternitycare)$",
    "emergency_24x7":       r"^(emergencymedicine|criticalcaremedicine)$",
    "icu":                  r"^(criticalcaremedicine|anesthesia|pulmonology)$",
    "picu":                 r"^(pediatriccriticalcaremedicine|neonatologyperinatalmedicine)$",
    "dialysis":             r"^(nephrology)$", "peritoneal_dialysis": r"^(nephrology)$",
    "general_surgery":      r"^(generalsurgery|colorectalsurgery)$",
    "laparoscopic_surgery": r"^(generalsurgery|minimallyinvasivesurgeryandendourology|bariatricsurgery)$",
    "orthopedic_surgery":   r"^(orthopedicsurgery|spineneurosurgery|shoulderandelboworthopedicsurgery|orthopedicspinesurgery)$",
    "joint_replacement":    r"^(jointreconstructionsurgery|orthopedicsurgery)$",
    "neurosurgery":         r"^(neurosurgery|spineneurosurgery)$",
    "urology":              r"^(urology|minimallyinvasivesurgeryandendourology|andrologyandmalefertility)$",
    "ent":                  r"^(otolaryngology)$",
    "eye_care":             r"^(ophthalmology|cataractandanteriorsegmentsurgery|retinaandvitreoretinalophthalmology|glaucomaophthalmology|corneaophthalmology)$",
    "plastic_surgery":      r"^(plasticsurgery|reconstructiveplasticsurgery|burnandtraumaplasticsurgery)$",
    "vascular_surgery":     r"^(vascularsurgery|cardiothoracicsurgery)$",
    "transplant":           r"^(transplantsurgery)$",
    "pediatric_surgery":    r"^(pediatricsurgery)$",
    "surgical_oncology":    r"^(surgicaloncology|gynecologicaloncology|urologiconcology)$",
    "cardiology":           r"^(cardiology|interventionalcardiology)$",
    "psychiatry":           r"^(psychiatry)$",
    "maternity":            r"^(gynecologyandobstetrics|obstetricsandmaternitycare|maternalfetalmedicineorperinatology)$",
    "pediatrics":           r"^(pediatrics|neonatologyperinatalmedicine)$",
    "neurology":            r"^(neurology)$", "gastroenterology": r"^(gastroenterology)$", "pulmonology": r"^(pulmonology)$",
    "endocrinology_diabetes": r"^(endocrinologyanddiabetesandmetabolism)$", "dermatology": r"^(dermatology|cosmeticdermatology)$",
    "hematology":           r"^(hematology)$", "dental": r"^(dentistry|generaldentistry|endodontics|orthodontics|prosthodontics|periodontics)$",
    "ivf_fertility":        r"^(reproductiveendocrinologyandinfertility)$", "physiotherapy_rehab": r"^(physicalmedicineandrehabilitation)$",
    "endoscopy":            r"^(gastroenterology)$", "cardiac_diagnostics": r"^(cardiology)$", "lab_pathology": r"^(pathology|clinicalpathology)$",
    "xray":                 r"^(radiology)$", "ultrasound": r"^(radiology)$", "mammography": r"^(radiology)$", "ct_scan": r"^(radiology)$",
    "mri":                  r"^(radiology)$", "pet_ct": r"^(radiology|nuclearmedicine)$", "bariatric_surgery": r"^(bariatricsurgery)$",
}
# service -> words in the government directory's Specialties / Facilities text that confirm it (the answer key).
# Same terms Model 1's directory check uses.
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
    "xray": r"x-?ray|radiolog|imaging", "ultrasound": r"ultrasound|\busg\b|sonograph|radiolog|imaging", "mammography": r"mammograph|radiolog",
    "endoscopy": r"endoscop|gastroenterolog", "cardiac_diagnostics": r"\becg\b|echo|cardiolog", "lab_pathology": r"patholog|laborator|\blab\b",
    "general_surgery": r"general surg", "laparoscopic_surgery": r"laparoscop|general surg", "orthopedic_surgery": r"orthop",
    "joint_replacement": r"joint replace|orthop", "neurosurgery": r"neuro ?-?surg", "urology": r"urolog", "ent": r"\bent\b|otorhino|otolaryng|\bear\b",
    "eye_care": r"ophthalm|opthalm|\beye\b", "plastic_surgery": r"plastic", "bariatric_surgery": r"bariatric", "vascular_surgery": r"vascular",
    "transplant": r"transplant", "pediatric_surgery": r"pa?ediatric surg", "surgical_oncology": r"surgical oncolog|oncolog|cancer",
    "cardiology": r"cardiolog", "neurology": r"neurolog", "gastroenterology": r"gastroenterolog|gastro", "pulmonology": r"pulmonolog|chest|respirat|\btb\b",
    "endocrinology_diabetes": r"endocrin|diabet", "psychiatry": r"psychiatr|mental", "physiotherapy_rehab": r"physiotherap|rehabilit|physical medicine",
    "dermatology": r"dermatolog|skin", "hematology": r"hematolog|haematolog", "palliative_care": r"palliative", "vaccination": r"vaccin|immuni",
    "maternity": r"obstet|gyn(?:a)?ecolog|maternity", "ivf_fertility": r"\bivf\b|infertil|fertility", "pediatrics": r"pa?ediatric|child",
    "dental": r"dental|dentist|\bdent", "ambulance": r"ambulance",
}
ALL_SERVICES = ["dialysis", "peritoneal_dialysis", "icu", "nicu", "picu", "c_section", "cardiac_surgery", "cath_lab_angioplasty",
                "mri", "pet_ct", "ct_scan", "trauma_center", "emergency_24x7", "burn_care", "chemotherapy", "radiotherapy",
                "blood_bank", "ventilator", "xray", "ultrasound", "mammography", "endoscopy", "cardiac_diagnostics", "lab_pathology",
                "general_surgery", "laparoscopic_surgery", "orthopedic_surgery", "joint_replacement", "neurosurgery", "urology", "ent",
                "eye_care", "plastic_surgery", "bariatric_surgery", "vascular_surgery", "transplant", "pediatric_surgery",
                "surgical_oncology", "cardiology", "neurology", "gastroenterology", "pulmonology", "endocrinology_diabetes",
                "psychiatry", "physiotherapy_rehab", "dermatology", "hematology", "palliative_care", "vaccination", "maternity",
                "ivf_fertility", "pediatrics", "dental", "ambulance"]
SERVICES = ALL_SERVICES                      # every Model 1 service gets a row
DEFAULTS = {"internalmedicine", "familymedicine"}           # listed by nearly everyone: no information
SINGLE_SPECIALTY = re.compile(r"\beye\b|netra|nethra|ophthalm|dental|dentist|\bskin\b|derma|\bivf\b|fertility|\bent\b", re.I)
MIXED = re.compile(r"multi|super|general|children|women|maternity|medical college|\bhospitals? (?:and|&)|(?:and|&) (?:\w+ )?hospital", re.I)
# Model 2 changes the finder label only for services where, on held-out hospitals, it ranked better than Model 1.
# This list is worked out on every run (needs Model 1's results); it is printed and saved in `used_in_finder`.
PLAIN = {"log_beds": "beds", "log_doctors": "doctors", "beds_known": "bed count published",
         "doctors_known": "doctor count published", "is_hospital": "is a hospital", "is_clinic": "is a clinic",
         "is_public": "public facility", "related_specialty": "lists a related specialty",
         "log_n_specialties": "number of specialties", "log_equipment": "amount of equipment listed",
         "single_specialty_name": "single-specialty name", "age_known": "founding year published",
         "log_age": "years in operation"}


def num(x):
    try:
        v = float(str(x).strip().replace(",", ""))
        return v if v > 0 else np.nan
    except ValueError:
        return np.nan


# ----------------------------------------------------------------------------- features (no service wording)
def facility_features(g: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame({"unique_id": g.unique_id.values})
    beds, docs = g.capacity_beds.map(num).values, g.number_doctors.map(num).values
    year = g.year_established.map(num).values
    age = np.where((year > 1850) & (year <= 2026), 2026 - year, np.nan)
    codes = g.specialty_codes.map(lambda s: {c.strip().lower() for c in s.split("|") if c.strip()} - DEFAULTS)
    f["beds_known"], f["log_beds"] = ~np.isnan(beds), np.log1p(np.nan_to_num(beds))
    f["doctors_known"], f["log_doctors"] = ~np.isnan(docs), np.log1p(np.nan_to_num(docs))
    f["age_known"], f["log_age"] = ~np.isnan(age), np.log1p(np.nan_to_num(age))
    f["is_hospital"] = (g.facility_type.values == "hospital")
    f["is_clinic"] = (g.facility_type.values == "clinic")
    f["is_public"] = (g.operator_type.values == "public")
    f["log_n_specialties"] = np.log1p(codes.map(len).values)
    f["log_equipment"] = np.log1p(g.equipment.map(lambda s: len([x for x in s.split("|") if x.strip()])).values)
    names = g.name.fillna("")
    f["single_specialty_name"] = names.map(lambda n: bool(SINGLE_SPECIALTY.search(n)) and not MIXED.search(n)).values
    f["_codes"] = codes.values
    return f


def long_rows(fac: pd.DataFrame) -> pd.DataFrame:
    """facility x service rows; adds the service-specific 'related specialty' flag."""
    parts = []
    for svc in SERVICES:
        r = fac.copy()
        r["service"] = svc
        pat = RELATED.get(svc)
        r["related_specialty"] = r._codes.map(lambda cs: bool(pat) and any(re.match(pat, c) for c in cs))
        parts.append(r)
    out = pd.concat(parts, ignore_index=True)
    known = out.beds_known.astype(int) + out.doctors_known.astype(int)
    out["info_level"] = np.where(known == 2, "rich", np.where(known == 1, "partial", "thin"))
    out["unknown"] = out.is_hospital & ~out.beds_known & ~out.doctors_known & ~out.related_specialty
    return out


def design(rows: pd.DataFrame) -> pd.DataFrame:
    cols = list(PLAIN)
    X = rows[cols].astype(float)
    for s in SERVICES[1:]:
        X[f"svc_is_{s}"] = (rows.service == s).astype(float)
    return X


# ----------------------------------------------------------------------------- answer key
def labels(g: pd.DataFrame, directory: pd.DataFrame) -> pd.DataFrame:
    """1 if the government directory's own service text for the matched hospital names the service, else 0."""
    d = directory.copy()
    d["dir_services"] = (d.Specialties.where(d.Specialties != "0", "") + " ; " +
                         d.Facilities.where(d.Facilities != "0", "")).str.replace(r"\\n", " ", regex=True).str.strip(" ;")
    d = d[d.dir_services != ""][["Sr_No", "dir_services"]]
    m = g[g.directory_match.isin(["STRONG", "PROBABLE"])][["unique_id", "directory_id"]].merge(
        d, left_on="directory_id", right_on="Sr_No")
    out = [{"unique_id": r.unique_id, "service": svc, "y": int(bool(re.search(pat, r.dir_services, re.I)))}
           for r in m.itertuples() for svc, pat in DIR_TERMS.items()]
    return pd.DataFrame(out).drop_duplicates(["unique_id", "service"])


def band(p, unknown):
    return np.where(unknown, "UNKNOWN", np.where(p >= BANDS[1], "LIKELY", np.where(p >= BANDS[0], "POSSIBLE", "UNLIKELY")))


DRIVER_TEXT = {  # (text when the value is high/true, text when low/false)
    "log_beds": ("many beds", "few or no listed beds"), "log_doctors": ("many doctors", "few or no listed doctors"),
    "beds_known": ("bed count published", "no bed count"), "doctors_known": ("doctor count published", "no doctor count"),
    "is_hospital": ("is a hospital", "not a hospital"), "is_clinic": ("is a clinic", "not a clinic"),
    "is_public": ("public facility", "private facility"), "related_specialty": ("lists a related specialty", "no related specialty"),
    "log_n_specialties": ("many specialties", "few specialties"), "log_equipment": ("lots of equipment listed", "little equipment listed"),
    "single_specialty_name": ("single-specialty name (eye, dental, IVF...)", "not a single-specialty name"),
    "age_known": ("founding year published", "no founding year"), "log_age": ("long-established", "recently founded"),
}


def drivers(clf, X: pd.DataFrame, top=3) -> list[str]:
    """The inputs that moved each row's percentage most vs an average facility, as 'raises/lowers: fact'."""
    keys = list(PLAIN)
    w = pd.Series(clf.coef_[0], index=X.columns)[keys].values
    V = X[keys].values
    base = V.mean(axis=0)
    C = (V - base) * w
    order = np.argsort(-np.abs(C), axis=1)[:, :top]
    out = []
    for i in range(len(V)):
        parts = []
        for j in order[i]:
            if abs(C[i, j]) < 0.05:
                continue
            fact = DRIVER_TEXT[keys[j]][0] if V[i, j] > base[j] else DRIVER_TEXT[keys[j]][1]
            parts.append(f"{'raises' if C[i, j] > 0 else 'lowers'}: {fact}")
        out.append("; ".join(parts))
    return out


# ----------------------------------------------------------------------------- combined finder label (lookup, no retraining)
LOOKUP = {
    ("SPECIFIC", "LIKELY"): "Strong match", ("SPECIFIC", "POSSIBLE"): "Good match",
    ("SPECIFIC", "UNLIKELY"): "Conflict: call first", ("SPECIFIC", "UNKNOWN"): "Good match",
    ("LISTED", "LIKELY"): "Good match", ("LISTED", "POSSIBLE"): "Call to confirm",
    ("LISTED", "UNLIKELY"): "Doubtful", ("LISTED", "UNKNOWN"): "Call to confirm",
    ("PASSING", "LIKELY"): "Call to confirm", ("PASSING", "POSSIBLE"): "Doubtful",
    ("PASSING", "UNLIKELY"): "Not shown", ("PASSING", "UNKNOWN"): "Doubtful",
}
LABEL_RANK = {"Strong match": 5, "Good match": 4, "Call to confirm": 3, "Conflict: call first": 2,
              "Doubtful": 1, "Not shown": 0, "Not mentioned": -1}


def run(gold_path, directory_path, model1_path=None, out=None) -> pd.DataFrame:
    g = pd.read_csv(gold_path, dtype=str, keep_default_na=False)
    directory = pd.read_csv(directory_path, dtype=str, keep_default_na=False, low_memory=False)
    rows = long_rows(facility_features(g))
    lab = labels(g, directory).merge(rows, on=["unique_id", "service"])
    pos, neg = lab.groupby("service").y.sum(), lab.groupby("service").y.apply(lambda v: (v == 0).sum())
    assessed = [s for s in SERVICES if pos.get(s, 0) >= MIN_POS and neg.get(s, 0) >= MIN_POS]
    skipped = [s for s in SERVICES if s not in assessed]
    print(f"Assessed (>= {MIN_POS} yes and >= {MIN_POS} no in the directory answer key): {assessed}")
    print(f"Not assessed (too few examples to learn from): {skipped}")
    lab = lab[lab.service.isin(assessed)].reset_index(drop=True)
    X, y, groups = design(lab), lab.y.values, lab.unique_id.values
    print(f"Answer key: {lab.unique_id.nunique()} hospitals matched to the government directory x {len(assessed)} services "
          f"= {len(lab)} rows; directory lists the service for {y.sum()} ({y.mean():.0%})\n")

    # ---- fair test: hospitals held out together
    pred = np.zeros(len(lab))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        pred[te] = LogisticRegression(C=C_REG, max_iter=3000).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    lab["m2_pct"], lab["m2_band"] = pred, band(pred, lab.unknown.values)
    print(f"Held-out (hospitals never seen in training)\n  Model 2 alone: AUC {roc_auc_score(y, pred):.2f}")
    print("  Share the directory lists, by band:")
    print(lab.groupby("m2_band").y.agg(["size", "mean"]).reindex(["LIKELY", "POSSIBLE", "UNLIKELY", "UNKNOWN"])
          .dropna().rename(columns={"size": "rows", "mean": "dir_lists"}).assign(dir_lists=lambda t: (t.dir_lists * 100).round().astype(int).astype(str) + "%").to_string())

    print("  (pooled AUC across services is inflated by different approval rates per service; judge by the per-service table)")
    if model1_path:
        m1 = pd.read_csv(model1_path, dtype=str, keep_default_na=False)[["unique_id", "service", "evidence_level", "directory_confirm_pct"]]
        cmp_ = lab.merge(m1, on=["unique_id", "service"], how="left")
        lv = cmp_.evidence_level.fillna("").values
        # Model 1's % was trained on this same directory, so comparing with it would be unfair: use its rule-based level.
        m1score = np.array([{"SPECIFIC": 3, "LISTED": 2, "PASSING": 1}.get(l, 0) for l in lv], dtype=float)
        # 1) per-service: which services does Model 2 rank better than Model 1?
        per = []
        for svc in assessed:
            k = (cmp_.service == svc).values
            if cmp_.y[k].sum() >= MIN_POS and (k.sum() - cmp_.y[k].sum()) >= MIN_POS:
                per.append((svc, int(cmp_.y[k].sum()), roc_auc_score(cmp_.y[k], m1score[k]), roc_auc_score(cmp_.y[k], cmp_.m2_pct.values[k])))
        def combined_rank(services_using_m2):
            used = cmp_.service.isin(services_using_m2).values
            bands_used = np.where(used, cmp_.m2_band, "UNKNOWN")
            labels_ = [LOOKUP.get((l, b), "Not mentioned") for l, b in zip(lv, bands_used)]
            tie = np.where(used, cmp_.m2_pct.values, 0.0)
            return labels_, np.array([LABEL_RANK[c] for c in labels_]) + tie * 0.01   # tie-break inside a label

        # keep Model 2 only where the COMBINED ranking beats Model 1 alone on held-out hospitals
        _, trial = combined_rank({svc for svc, *_ in per})
        use_in_finder = {svc for svc, _, a1, _ in per
                         if roc_auc_score(cmp_.y[(cmp_.service == svc).values], trial[(cmp_.service == svc).values]) > a1}
        comb, comb_rank = combined_rank(use_in_finder)
        t = pd.DataFrame([(svc, n, a1, a2, roc_auc_score(cmp_.y[(cmp_.service == svc).values], comb_rank[(cmp_.service == svc).values]))
                          for svc, n, a1, a2 in per], columns=["service", "dir_lists", "Model 1", "Model 2", "Combined"]).round(2)
        print(f"\n  Within each service (fair comparison), held-out AUC. Model 1 = its Specific/Listed/Passing level:")
        print(t.sort_values("dir_lists", ascending=False).to_string(index=False))
        print("  average:", t[["Model 1", "Model 2", "Combined"]].mean().round(2).to_dict(),
              "  PRD success = Combined beats Model 1 by >= 0.05")
        print(f"  Model 2 changes the finder label for {len(use_in_finder)} services: {sorted(use_in_finder)}")
        print("  (this choice is made on the same held-out results, so the Combined average is slightly optimistic)")
        cmp_["combined"] = comb
        print("  Share the directory lists, by combined label:")
        print(cmp_.groupby("combined").y.agg(["size", "mean"]).assign(r=lambda q: q.index.map(LABEL_RANK))
              .sort_values("r", ascending=False).drop(columns="r").rename(columns={"size": "rows", "mean": "dir_lists"})
              .assign(dir_lists=lambda q: (q.dir_lists * 100).round().astype(int).astype(str) + "%").to_string())

    # ---- final model on all labelled rows -> every facility
    clf = LogisticRegression(C=C_REG, max_iter=3000).fit(X, y)
    w = pd.Series(clf.coef_[0], index=X.columns)
    print("\nWhat moves capability (+ raises, - lowers):")
    for k, v in w[list(PLAIN)].sort_values().items():
        print(f"  {v:+.2f}  {PLAIN[k]}")

    Xall = design(rows)
    p = clf.predict_proba(Xall)[:, 1]
    ok = rows.service.isin(assessed).values
    res = pd.DataFrame({"unique_id": rows.unique_id, "service": rows.service,
                        "capability_band": np.where(ok, band(p, rows.unknown.values), "NOT_ASSESSED"),
                        "capability_pct": np.where(ok & ~rows.unknown.values, np.round(p * 100), np.nan),
                        "drivers": np.where(ok, drivers(clf, Xall), ""), "info_level": rows.info_level,
                        "model_version": MODEL_VERSION})
    if model1_path:
        m1 = pd.read_csv(model1_path, dtype=str, keep_default_na=False)[["unique_id", "service", "evidence_level"]]
        res = res.merge(m1, on=["unique_id", "service"], how="left")
        res["finder_label"] = [LOOKUP.get((lv, b if svc in use_in_finder else "UNKNOWN"), "Not mentioned")
                               if isinstance(lv, str) else "Not mentioned"
                               for lv, b, svc in zip(res.evidence_level, res.capability_band, res.service)]
        res["used_in_finder"] = res.service.isin(use_in_finder)
    print(f"\nScored {res.unique_id.nunique()} facilities x {len(SERVICES)} services = {len(res)} rows")
    print(res.capability_band.value_counts().to_string())
    if out:
        res.to_csv(out, index=False)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold")
    ap.add_argument("directory")
    ap.add_argument("--model1")
    ap.add_argument("--out", default="model2_capability.csv")
    a = ap.parse_args()
    run(a.gold, a.directory, a.model1, a.out)

# COMMAND ----------

# MAGIC %md ## 2. Use Model 1's results if they exist

# COMMAND ----------

MODEL1_CSV = None
try:
    spark.table(MODEL1_TABLE).toPandas().to_csv("/tmp/model1_for_model2.csv", index=False)
    MODEL1_CSV = "/tmp/model1_for_model2.csv"
    print("Using Model 1 results from", MODEL1_TABLE)
except Exception as e:
    print("Model 1 table not found - running Model 2 on its own (no comparison, no combined label).")

# COMMAND ----------

# MAGIC %md ## 3. Train, test on held-out hospitals, score every facility

# COMMAND ----------

result = run(GOLD, DIRECTORY, MODEL1_CSV)
display(result)

# COMMAND ----------

# MAGIC %md ## 4. Conflicts: specific claims the facility's size or type doesn't support

# COMMAND ----------

fac = pd.read_csv(GOLD, dtype=str, keep_default_na=False)[["unique_id", "name", "city", "state", "capacity_beds", "facility_type"]]
if "finder_label" in result:
    display(result[result.finder_label == "Conflict: call first"].merge(fac, on="unique_id")
            [["name", "city", "state", "facility_type", "capacity_beds", "service", "capability_pct", "drivers"]])

# COMMAND ----------

# MAGIC %md ## 5. (Master-notebook owner only) save the result

# COMMAND ----------

# spark.createDataFrame(result.astype(str)).write.mode("overwrite").saveAsTable("workspace.datathon_2026.model2_capability")
