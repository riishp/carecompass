"""Accuracy test 1 — which version agrees most with the government hospital directory? (offline, no Databricks)

Answer key: for our facilities matched to the government directory (STRONG/PROBABLE) that have a service list there,
does the directory list the service? (Model 2's DIR_TERMS — the same terms the models' own tests use.)

Each version scores every labelled (facility, service) pair; we report, per service, the AUC
(0.5 = coin flip, 1.0 = perfect: how well the version's scores put directory-listed services above unlisted ones),
then the average across services (pooled AUC would be inflated by different base rates per service).

  Current   the app's Find care evidence level (capabilities.level_for: NONE < PASSING < LISTED < SPECIFIC).
            NOTE: for the 7 checked services the live app uses 04's hand-tuned evidence instead; this offline test
            uses the generic matcher for every service.
  Model 1   its rule-based level (the extended 'all app services' copy). Its % is NOT used: it was trained on this
            same directory, so it would be scored on its own answer key.
  Model 2   HELD-OUT capability %: re-created with the notebook's own 5-fold GroupKFold, so every hospital is scored
            by a model that never saw it.
  Combined  the notebook's finder-label lookup (Model 1 level × held-out Model 2 band), Model 2 % as tie-break.

Usage: .venv/bin/python tests/model_eval.py <gold_facility_search_services.csv> <hospital_directory.csv>
Writes tests/results/model_eval_test1.csv
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import capabilities as caps  # noqa: E402

gold_csv, dir_csv = sys.argv[1:3]

# ---- Model 2's code, unchanged (for its features, answer key and lookup)
m2 = {"__name__": "m2", "display": lambda *a, **k: None}
for c in (ROOT / "notebooks" / "model2_facility_capability.py").read_text().split("# COMMAND ----------"):
    code = "\n".join(l for l in c.splitlines() if not l.startswith("# MAGIC"))
    if code.strip() and not any(k in code for k in ("spark.table", "result = run(", "fac = pd.read_csv(GOLD", "saveAsTable")):
        exec(compile(code, "model2", "exec"), m2)

g = pd.read_csv(gold_csv, dtype=str, keep_default_na=False)
directory = pd.read_csv(dir_csv, dtype=str, keep_default_na=False, low_memory=False)
rows = m2["long_rows"](m2["facility_features"](g))
lab = m2["labels"](g, directory).merge(rows, on=["unique_id", "service"])

# app capability → model service (versions.py); keep services that some app capability maps to
sys.modules.setdefault("ui_results", type(sys)("ui_results"))  # versions imports ui only for labels
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("versions_src", ROOT / "versions.py")
src = (ROOT / "versions.py").read_text()
MODEL2_SERVICE = eval(re.search(r"MODEL2_SERVICE = (\{.*?\})\n", src, re.S).group(1))
svc_caps = {}
for cap, svc in MODEL2_SERVICE.items():
    svc_caps.setdefault(svc, []).append(cap)
lab = lab[lab.service.isin(svc_caps)].reset_index(drop=True)
pos, neg = lab.groupby("service").y.sum(), lab.groupby("service").y.apply(lambda v: (v == 0).sum())
assessed = [s for s in svc_caps if pos.get(s, 0) >= m2["MIN_POS"] and neg.get(s, 0) >= m2["MIN_POS"]]
lab = lab[lab.service.isin(assessed)].reset_index(drop=True)
print(f"Answer key: {lab.unique_id.nunique()} directory-matched hospitals × {len(assessed)} services = {len(lab)} pairs "
      f"(directory lists {lab.y.mean():.0%})")

# ---- Model 2 held-out % (same CV as the notebook, but only on the app's services)
X, y, groups = m2["design"](lab), lab.y.values, lab.unique_id.values
pred = np.zeros(len(lab))
for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
    pred[te] = LogisticRegression(C=m2["C_REG"], max_iter=3000).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
lab["m2"] = pred
lab["m2_band"] = m2["band"](pred, lab.unknown.values)

# ---- Model 1 (extended) levels
m1 = pd.read_csv(ROOT / "catalog" / "model1_levels.csv", dtype={"unique_id": str})[["unique_id", "service", "evidence_level"]]
lab = lab.merge(m1, on=["unique_id", "service"], how="left")
lab["m1"] = lab.evidence_level.map({"SPECIFIC": 3, "LISTED": 2, "PASSING": 1}).fillna(0)

# ---- Combined: the notebook's lookup on (Model 1 level, held-out Model 2 band)
LOOKUP, LABEL_RANK = m2["LOOKUP"], m2["LABEL_RANK"]
lab["combined_label"] = [LOOKUP.get((l, b), "Not mentioned") if isinstance(l, str) else "Not mentioned"
                         for l, b in zip(lab.evidence_level, lab.m2_band)]
lab["combined"] = lab.combined_label.map(LABEL_RANK) + lab.m2 * 0.01

# ---- Current: the app's evidence level (generic matcher) for the app capabilities mapped to each service
fac = g.set_index("unique_id")
sp = lambda x: [i for i in str(x).split("|") if i.strip()]
LV = {"NONE": 0, "PASSING": 1, "LISTED": 2, "SPECIFIC": 3}
cur = {}
for uid in lab.unique_id.unique():
    r = fac.loc[uid]
    fields = {"specialties": sp(r.specialty_labels) + sp(r.specialties_other), "procedures": sp(r.procedures),
              "equipment": sp(r.equipment), "capabilities": sp(r.capabilities),
              "description": [r.description] if r.description else []}
    mentions = caps.find_mentions(fields)
    main, doms = caps.specialty_domains(sp(r.specialty_labels))
    for svc, cids in svc_caps.items():
        cur[(uid, svc)] = max(LV[caps.level_for(caps.catalog()["by_id"][c], mentions.get(c, []),
                                                {"equipment": bool(fields["equipment"])}, main, doms)["level"]] for c in cids)
lab["current"] = [cur[(u, s)] for u, s in zip(lab.unique_id, lab.service)]

# ---- per-service AUC
VERS = ["current", "m1", "m2", "combined"]
NAMES = {"current": "Current (Find care)", "m1": "Model 1", "m2": "Model 2", "combined": "Combined"}
out = []
for svc in assessed:
    k = lab.service == svc
    row = {"service": svc, "pairs": int(k.sum()), "dir_lists": int(lab.y[k].sum())}
    for v in VERS:
        row[NAMES[v]] = roc_auc_score(lab.y[k], lab[v][k]) if lab[v][k].nunique() > 1 else 0.5
    row["current_covers"] = f"{(lab.current[k] > 0).mean():.0%}"
    row["model1_covers"] = f"{(lab.m1[k] > 0).mean():.0%}"
    out.append(row)
t = pd.DataFrame(out).sort_values("pairs", ascending=False)
(ROOT / "tests" / "results").mkdir(parents=True, exist_ok=True)
t.to_csv(ROOT / "tests" / "results" / "model_eval_test1.csv", index=False)
pd.set_option("display.width", 200)
print(t.round(2).to_string(index=False))
avg = t[[NAMES[v] for v in VERS]].mean().round(3)
print("\nAverage AUC across services:", avg.to_dict())
print("Best:", avg.idxmax())
