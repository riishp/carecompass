# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 07 — Specialty Standardization
# MAGIC
# MAGIC **Problem** (`03_Data_Quality_Audit` §4h): the `specialties` lists hold **2,933 different strings**, but most are the same specialty written differently —
# MAGIC `generalSurgery` vs `General Surgery`, `pediatricNephrology` vs `paediatricNephrology`, `gynecologyAndObstetrics` vs `Obstetrics & Gynaecology`, `ENT/ Otorhinolaryngologist` vs `otolaryngology`.
# MAGIC 150 codes (each used by ≥ 50 facilities) already cover 95% of all tags.
# MAGIC
# MAGIC **What this notebook does:** builds a lookup from every raw specialty string → one **standard code** + a readable **label**. It does not change facility rows; Silver applies the lookup.
# MAGIC
# MAGIC **Input (read-only):** `databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities` (`specialties` column)
# MAGIC **Output:** `workspace.datathon_2026.specialty_map` — `raw_specialty, specialty_code, specialty_label, match_type, n_facilities`
# MAGIC
# MAGIC **Matching order (first hit wins):**
# MAGIC 1. `standard` — already one of the standard codes
# MAGIC 2. `format_variant` — same words after fixing case, spacing, `&`, British spelling (paediatric → pediatric), word order
# MAGIC 3. `synonym` — hand-checked synonyms (ENT → otolaryngology, IVF → infertility, Obstetrics & Gynaecology → gynecologyAndObstetrics…)
# MAGIC 4. `profession` — the doctor's title instead of the field (gastroenterologist → gastroenterology)
# MAGIC 5. `rolled_up_to_parent` — a sub-specialty with no own code goes to its broader field (interventionalCardiology → cardiology)
# MAGIC 6. `fuzzy` — ≥ 92% similar spelling to a standard code (typos: opthalmology → ophthalmology)
# MAGIC 7. `other` — genuinely unusual (e.g. *earwax removal*); kept with a cleaned label, not thrown away
# MAGIC
# MAGIC **Cleaning only — no analysis.** Section 3 runs validation checks; all must PASS.

# COMMAND ----------

# DBTITLE 1,0. Setup
import re, difflib
from datetime import datetime, timezone
import pandas as pd
from pyspark.sql import functions as F

SRC = "databricks_virtue_foundation_dataset_dais_2026.virtue_foundation_dataset.facilities"
TGT = "workspace.datathon_2026"
SPEC_MAP = f"{TGT}.specialty_map"
AUDIT = f"{TGT}.cleaning_audit"
STANDARD_MIN_FACILITIES = 50   # a code used by this many facilities counts as a standard code

RUN_ID = "spec-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
audit_rows = []
def log(step, rule, n, note=""):
    audit_rows.append((RUN_ID, f"spec-{step}", rule, int(n), note))
    print(f"[{step}] {rule}: {n:,} {note}")

counts = (spark.read.table(SRC)
          .select("unique_id", F.explode(F.array_distinct(F.from_json("specialties", "array<string>"))).alias("raw"))
          .where("raw IS NOT NULL AND trim(raw) != '' AND lower(trim(raw)) NOT IN ('null','none','n/a')")
          .select("unique_id", F.trim("raw").alias("raw")).distinct()
          .groupBy("raw").agg(F.count("*").alias("n_facilities"))
          .toPandas().sort_values("n_facilities", ascending=False))
log(0, "distinct raw specialty strings", len(counts))

# COMMAND ----------

# DBTITLE 1,1. Normalizer, synonyms and standard codes
BRITISH = [("paediatric", "pediatric"), ("paediatr", "pediatr"), ("gynaecolog", "gynecolog"), ("haemato", "hemato"),
           ("anaesthe", "anesthe"), ("orthopaedic", "orthopedic"), ("oesophag", "esophag"), ("oncosurg", "surgical oncolog"),
           ("foetal", "fetal"), ("anaemia", "anemia"), ("haem", "hem")]
STOP = {"and", "of", "the", "or", "in", "for", "dept", "department", "clinic", "services", "service", "treatment", "care", "unit", "speciality", "specialty", "sciences", "science"}

def words(s):
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)              # camelCase -> words
    s = s.lower().replace("&", " and ")
    for a, b in BRITISH:
        s = s.replace(a, b)
    return [w for w in re.split(r"[^a-z]+", s) if w and w not in STOP]

def key(s):          # order-insensitive key: "Obstetrics & Gynaecology" == "gynecologyAndObstetrics"
    return " ".join(sorted(words(s)))

def joined(s):       # spacing-insensitive key: "Micro Biology" == "microbiology"
    return "".join(words(s))

standard = counts[(counts.n_facilities >= STANDARD_MIN_FACILITIES) & counts.raw.str.match(r"^[a-z][A-Za-z–\-]*$")]
STANDARD = list(standard.raw)
BY_KEY = {}
for code in STANDARD:                       # most-used code wins a shared key
    BY_KEY.setdefault(key(code), code)
BY_JOINED = {}
for code in STANDARD:
    BY_JOINED.setdefault(joined(code), code)
log(1, "standard codes (camelCase, >= 50 facilities)", len(STANDARD))

# Hand-checked synonyms: normalized key of the raw text -> standard code
SYNONYMS_RAW = {
    "ENT": "otolaryngology", "ENT/ Otorhinolaryngologist": "otolaryngology", "Otorhinolaryngology": "otolaryngology",
    "Ear Nose Throat": "otolaryngology",
    "IVF": "reproductiveEndocrinologyAndInfertility", "IVF Treatment": "reproductiveEndocrinologyAndInfertility",
    "Infertility": "reproductiveEndocrinologyAndInfertility", "Fertility": "reproductiveEndocrinologyAndInfertility",
    "Obstetrics & Gynaecology": "gynecologyAndObstetrics", "Obstetrics and Gynecology": "gynecologyAndObstetrics",
    "OBG": "gynecologyAndObstetrics", "Obstetrics": "obstetricsAndMaternityCare",
    "General Medicine": "generalMedicine", "Neuro Sciences": "neurology", "Neurosciences": "neurology",
    "Cancer Care": "oncology", "Cancer": "oncology", "Oncosurgery": "surgicalOncology",
    "Cardio Thoracic Surgery": "cardiothoracicSurgery", "CTVS": "cardiothoracicSurgery",
    "General & Laparoscopic Surgery": "generalSurgery", "Laparoscopic Surgery": "minimallyInvasiveSurgeryAndEndourology",
    "GI Surgery": "gastroenterology", "Surgical Gastroenterology": "gastroenterology",
    "Physiotherapy": "physiotherapy", "Physical Therapy": "physiotherapy",
    "Diabetes": "diabetology", "Diabetic Retinopathy": "retinaAndVitreoretinalOphthalmology",
    "Retina": "retinaAndVitreoretinalOphthalmology", "Cataract": "cataractAndAnteriorSegmentSurgery",
    "Dialysis": "nephrology", "Kidney Transplant": "renalTransplantationUrology",
    "Joint Replacement": "jointReconstructionSurgery", "Knee Replacement": "jointReconstructionSurgery",
    "Hip Replacement": "jointReconstructionSurgery", "Spine Surgery": "orthopedicSpineSurgery",
    "Dental": "dentistry", "Dental Implants": "dentalImplants", "Root Canal": "endodontics",
    "Paediatrics & Neonatology": "pediatrics", "Neonatology": "neonatologyPerinatalMedicine", "NICU": "neonatologyPerinatalMedicine",
    "ICU": "criticalCareMedicine", "Critical Care": "criticalCareMedicine", "Emergency": "emergencyMedicine",
    "Trauma": "emergencyMedicine", "Casualty": "emergencyMedicine",
    "Psychology": "psychiatry", "Mental Health": "psychiatry", "De-addiction": "addictionPsychiatry",
    "Pulmonary Medicine": "pulmonology", "Chest Medicine": "pulmonology", "Respiratory Medicine": "pulmonology",
    "Skin": "dermatology", "Cosmetology": "cosmeticDermatology", "Hair Transplant": "hairAndNailDisorders",
    "Pathology Lab": "pathology", "Laboratory": "pathology", "Diagnostics": "pathology", "Imaging": "radiology",
    "Anaesthesiology": "anesthesiology", "Nuclear Medicine": "nuclearMedicineAndMolecularImaging",
    "Palliative Care": "hospiceAndPalliativeInternalMedicine", "Geriatrics": "geriatricsInternalMedicine",
    "Plastic & Reconstructive Surgery": "reconstructivePlasticSurgery", "Cosmetic Surgery": "aestheticAndCosmeticSurgery",
    "Maxillofacial Surgery": "oralAndMaxillofacialSurgery", "Maxillo Facial Surgery": "oralAndMaxillofacialSurgery",
    "Oral Surgery": "oralAndMaxillofacialSurgery", "Pedodontics": "pediatricDentistry",
    # common names with no exact standard code (checked against the 'other' list)
    "orthopedics": "orthopedicSurgery", "orthopaedics": "orthopedicSurgery", "andrology": "andrologyAndMaleFertility",
    "jointReplacementSurgery": "jointReconstructionSurgery", "hepatology": "gastroenterology",
    "gastrointestinalSurgery": "gastroenterology", "gastroSurgery": "gastroenterology", "upperGIAndForegutSurgery": "gastroenterology",
    "laryngology": "otolaryngology", "rhinology": "otolaryngology", "neurotologyAndOtology": "otolaryngology", "audiology": "otolaryngology",
    "handAndUpperExtremitiesSurgery": "handOrUpperExtremityAndPeripheralNerveSurgery",
    "handOrUpperExtremitiesSurgery": "handOrUpperExtremityAndPeripheralNerveSurgery",
    "generalThoracicSurgery": "cardiothoracicSurgery", "gynecologicalSurgery": "gynecology",
    "oculoplasty": "oculoplasticsAndReconstructiveOrbitalSurgery", "neonatalPerinatalMedicine": "neonatologyPerinatalMedicine",
    "reproductiveMedicine": "reproductiveEndocrinologyAndInfertility", "paediatricOrthopaedics": "pediatricOrthopedicSurgery",
    "endourology": "minimallyInvasiveSurgeryAndEndourology", "implantology": "dentalImplants", "periodontology": "periodontics",
    "laboratoryMedicine": "pathology", "biochemistry": "pathology", "clinicalBiochemistry": "pathology", "serology": "pathology",
    "cytology": "pathology", "cytologyLaboratory": "pathology", "histopathologyLaboratory": "pathology",
    "neurocriticalCare": "criticalCareMedicine", "sportsMedicine": "sportsMedicinePMR", "painManagement": "painMedicinePMR",
    "femaleUrology": "urology", "neuroUrology": "urology", "genitourinarySurgery": "urology",
    "genitourinaryReconstructiveSurgery": "urology", "proctology": "colorectalSurgery",
    "TMJSurgery": "oralAndMaxillofacialSurgery", "orthognathicSurgery": "oralAndMaxillofacialSurgery",
    "boneMarrowTransplant": "hematology", "oralPathology": "oralMedicine", "oralMedicineAndRadiology": "oralMedicine",
}
SYNONYMS = {key(k): v for k, v in SYNONYMS_RAW.items()}
bad_targets = sorted({v for v in SYNONYMS_RAW.values() if v not in STANDARD})
assert not bad_targets, f"Synonym targets that aren't standard codes: {bad_targets}"

PROFESSION = [(r"ologists?$", "ology"), (r"iatrists?$", "iatry"), (r"icians?$", "ics"), (r"geons?$", "gery"),
              (r"dontists?$", "dontics"), (r"ists?$", "y")]

def label(code):
    """'endocrinologyAndDiabetesAndMetabolism' -> 'Endocrinology and Diabetes and Metabolism'"""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", code)          # "GIAnd" -> "GI And"
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s).replace("–", "-")
    s = " ".join(w if w.isupper() and len(w) > 1 else w.lower() for w in s.split())
    s = re.sub(r"\b(and|or|of)\b", lambda m: m.group(0).lower(), s)
    return s[:1].upper() + s[1:]

# COMMAND ----------

# DBTITLE 1,2. Map every raw string
def classify(raw):
    if raw in STANDARD:
        return raw, "standard"
    k, j = key(raw), joined(raw)
    if not k:
        return None, "empty"
    if k in BY_KEY:
        return BY_KEY[k], "format_variant"
    if j in BY_JOINED:
        return BY_JOINED[j], "format_variant"
    if k in SYNONYMS:
        return SYNONYMS[k], "synonym"
    for pat, rep in PROFESSION:
        w = words(raw)
        if w and re.search(pat, w[-1]):
            k2 = " ".join(sorted(w[:-1] + [re.sub(pat, rep, w[-1])]))
            if k2 in BY_KEY:
                return BY_KEY[k2], "profession"
    # junk prefix like "orth orthopedicSurgery": a standard code is the tail of the string
    # a standard code is the tail of the string: either junk in front ("orth orthopedicSurgery") or a
    # sub-specialty rolled up to its broader field ("interventionalCardiology" -> cardiology)
    for code in sorted(STANDARD, key=len, reverse=True):
        if len(joined(code)) >= 8 and j.endswith(joined(code)):
            return code, ("format_variant" if raw.split()[-1] == code else "rolled_up_to_parent")
    m = difflib.get_close_matches(j, list(BY_JOINED), n=1, cutoff=0.92)
    if m:
        return BY_JOINED[m[0]], "fuzzy"
    return None, "other"

res = counts.copy()
res[["specialty_code", "match_type"]] = res.raw.apply(lambda r: pd.Series(classify(r)))
# 'other': keep a tidy label of the raw text so nothing is lost
res["specialty_label"] = [label(c) if isinstance(c, str) else (" ".join(words(r)).capitalize() or r) for r, c in zip(res.raw, res.specialty_code)]
res["is_standard"] = res.specialty_code.notna()
res = res.rename(columns={"raw": "raw_specialty"})[["raw_specialty", "specialty_code", "specialty_label", "match_type", "is_standard", "n_facilities"]]

summary = res.groupby("match_type").agg(strings=("raw_specialty", "count"), facility_tags=("n_facilities", "sum")).reset_index()
display(summary)
for _, r in summary.iterrows():
    log(2, f"match_type = {r.match_type}", int(r.strings), f"{int(r.facility_tags)} facility tags")
display(res[res.match_type.isin(["synonym", "profession", "fuzzy"])].sort_values(["match_type", "n_facilities"], ascending=[True, False]))
display(res[res.match_type == "format_variant"].sort_values("n_facilities", ascending=False).head(60))

spark.createDataFrame(res).withColumn("cleaned_run_id", F.lit(RUN_ID)) \
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(SPEC_MAP)
(spark.createDataFrame(audit_rows, "run_id string, step string, rule string, rows_affected long, note string")
      .withColumn("logged_at", F.current_timestamp())
      .write.mode("append").option("mergeSchema", "true").saveAsTable(AUDIT))

# COMMAND ----------

# DBTITLE 1,3. Validation
m = spark.read.table(SPEC_MAP).toPandas()
checks = []
def check(name, ok, detail=""):
    checks.append((name, "PASS" if ok else "FAIL", str(detail)))

check("Every raw string appears exactly once", m.raw_specialty.is_unique and len(m) == len(counts), f"{len(m)} rows, {len(counts)} raw strings")
codes_bad = set(m.specialty_code.dropna()) - set(STANDARD)
check("Every specialty_code is a standard code", not codes_bad, sorted(codes_bad)[:10])
check("Every row has a label", m.specialty_label.notna().all() and (m.specialty_label.str.len() > 0).all())
tag_share = m[m.is_standard].n_facilities.sum() / m.n_facilities.sum()
check("≥ 96% of facility tags map to a standard code", tag_share >= 0.96, f"{tag_share:.1%}")
n_codes = m.specialty_code.nunique()
check("Standard vocabulary is ≤ 200 codes", n_codes <= 200, f"{n_codes} codes")
# Spot checks — known variants must land on the right code
EXPECT = {"General Surgery": "generalSurgery", "Obstetrics & Gynaecology": "gynecologyAndObstetrics",
          "ENT/ Otorhinolaryngologist": "otolaryngology", "paediatricNephrology": "pediatricNephrology",
          "Internal Medicine": "internalMedicine", "orth orthopedicSurgery": "orthopedicSurgery",
          "obstetricsAndGynaecology": "gynecologyAndObstetrics", "Micro Biology": "microbiology"}
look = dict(zip(m.raw_specialty, m.specialty_code))
wrong = {k: look.get(k) for k, v in EXPECT.items() if k in look and look[k] != v}
check("Spot checks map correctly", not wrong, wrong or "ok")

result = spark.createDataFrame(checks, "check string, result string, detail string")
display(result)
failed = [x for x in checks if x[1] == "FAIL"]
if failed:
    raise AssertionError(f"{len(failed)} specialty validation check(s) FAILED: " + "; ".join(x[0] for x in failed))
print(f"All {len(checks)} specialty validation checks PASSED. {tag_share:.1%} of facility tags standardized into {n_codes} codes.")
