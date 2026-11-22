# CareCompass

**CareCompass turns "what does the patient need, and where are they?" into a ranked call plan of nearby facilities, with the evidence behind each pick.**

Built for health workers and NGO coordinators in India finding care for a patient.

**Results:** confidently wrong location answers fell from 81 to 1 of 1,517 test inputs, and of the top-3 picks we could check against the government directory, 70% were confirmed.

It is a Streamlit app. You enter what the patient needs and where they are, and CareCompass gives back a **call plan**:

- three facilities to call, in order, with what to ask each one
- a government fallback
- a referral slip in English or Hindi
- a "Tell the family" WhatsApp message

It was built for the CMU Datathon Fall 2026 (CMU Data Science Club × Databricks) on Virtue Foundation Indian healthcare facility data.

See [OVERVIEW.md](OVERVIEW.md) for features and a code map, [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) for the system design, and [CLEANING_README.md](CLEANING_README.md) for the data pipeline.

## Results

**Location understanding** (`tests/location_eval.py`, 1,517 generated inputs: district names, typos, old names, PINs, states, "Town, State"):

| | Correct | Asked to clarify | Confidently wrong | Not found |
|---|---|---|---|---|
| Before | 1,309 | 10 | **81** | 117 |
| After | 1,156 | 358 | **1** | 2 |

Confidently wrong answers fell from 81 to 1. "Correct" fell because typos now get a "did you mean…?" question instead of a silent guess.

**Scoring facilities** (`tests/model_eval.py`, 241 hospitals matched to the government directory × 27 services, average AUC, 0.5 = coin flip): Current 0.60 · Model 1 0.62 · Model 2 0.66 · Combined 0.67.

**The top 3 users see** (`tests/model_eval_top3.py`, 40 random districts × 10 services; share of checkable picks the government directory confirms): Current **70%** (43/61) · Model 1 57% (30/53) · Model 2 59% (44/75) · Combined 60% (46/77).

Our hand-tuned evidence rules are strict: for C-section they find evidence for only 10% of hospitals the government directory lists, versus 73% for Model 1. We favour fewer, better-supported picks and rely on the call plan and government fallback to cover the gap.

The combined model scores individual facilities best, but the current ranking gives the most practical top 3 because it also weighs distance, so the app ships the current ranking. The model rankings can be compared in Judge view. Caveats: only ~5% of picks can be checked against the directory, the directory lists coarse departments only, and both models learned from that same directory, so these comparisons are suggestive, not conclusive.

## What we recommend to Virtue Foundation

1. **Build verification into the workflow.** Coordinators already call facilities before sending patients. Record each call's outcome (confirmed / not offered / no answer) and feed it back into the data. That's the only way to get real ground truth.
2. **Collect branch-specific, structured service data.** Many records share identical template sentences (one appears in ~489 facilities), so chain-wide or directory text is mistaken for what a single branch offers. Ask for per-branch service lists with concrete details (equipment, counts, hours).
3. **Don't let websites decide who looks capable.** Public hospitals rarely publish detailed websites, so text-based scoring underrates free government care. Use government service standards (IPHS / First Referral Unit status) as a baseline and confirm by phone.
4. **Fix location data at the source.** Filler coordinates, swapped lat/lon and wrong district labels are common in the source data; validating coordinates against PIN code areas would prevent patients being routed to the wrong region.

## What you need to run it

The app reads its facility data from **Databricks Unity Catalog** tables in `workspace.datathon_2026`. These are built by the notebooks in `notebooks/` (`00_Full_Cleaning_Pipeline` → … → `09b_Gold_Verified`). Without a Databricks workspace that has those tables, search will not work.

| Part | Runs without Databricks? |
|---|---|
| `tests/test_suggestions.py`, `tests/test_family.py` | yes |
| `tests/model_eval.py`, `tests/model_eval_top3.py`, `tools/run_model*.py` | yes, given the datathon CSVs (paths are passed as arguments) |
| The app's search, `tests/test_matching.py`, `tests/location_eval.py` | no: these need Databricks Connect and the tables |

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt pytest scikit-learn
```

### Run locally against a Databricks workspace

1. Configure a CLI profile for your workspace: `databricks auth login --host <workspace-url> --profile datathon`.
2. Build the tables by running the pipeline notebooks in `notebooks/` in that workspace.
3. Start the app:

```bash
DATABRICKS_CONFIG_PROFILE=datathon DATABRICKS_WAREHOUSE_ID=<sql-warehouse-id> .venv/bin/streamlit run app.py
```

### Deploy as a Databricks App

`tools/deploy.sh` syncs the engine from `notebooks/06_recommend.py`, uploads the app and deploys it. It assumes the profile `datathon`, an app named `carecompass` and the workspace path `/Workspace/Shared/Datathon 2026/app`, so edit those first. The app must be running before you deploy.

## Limitations

- Facility data is self-reported. Every result says to call first.
- Distances are straight-line, and travel times are estimates from distance.
- Ranking weights are hand-set (sensitivity tested: ±20% weights keep ~73% of top-3s).
- Strict evidence rules miss some providers.
- Under 5% of picks can be checked against the government directory.
- Hindi template text is machine-translated.

## Notes

- **Where to edit the engine:** `engine/recommend_engine.py` is generated from `notebooks/06_recommend.py`. Edit the notebook, not the generated file.
- **Facility data is self-reported** and may be wrong. Always confirm by phone before a patient travels.
- **App tabs:** The app opens on Find care. Turn on the **Judge view** toggle (top-right) to see why other facilities were left out, how the ranking was decided, and the Model 1 / Model 2 / Combined comparison tabs.
- **Notebooks:** `notebooks/` holds every notebook the pipeline (`00_Full_Cleaning_Pipeline` and its 8 steps), the engine (`03b`, `04`, `05`, `06_recommend`, `09b`) and the models (`model1_evidence_levels.py`, `model1_all_services.py`, `model2_facility_capability.py`) depend on. Not included: the government-directory matching step, which a teammate did outside the workspace (`09b_Gold_Verified` reads its output CSV, `gold_facility_search_directory_verified.csv`), and exploratory notebooks the app doesn't use (`01_Verify_Virtue_Dataset`, `02_Facilities_EDA`). `06_07_Recommend_Google_Driving_Combined.py` (recommendation with Google driving times; in progress, not used by the app yet) is included; it reads the Google Maps key from `GOOGLE_MAPS_API_KEY` or a Databricks secret.
- **Demo phone number:** if `config/demo_phone.local.json` exists (it is gitignored), every "Call first" shows its number, labelled "Demo number". Otherwise the app reads `config/demo_phone.json`, which is off, and shows real numbers from the data.
