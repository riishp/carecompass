# CareCompass

A Streamlit app that helps a health coordinator in India find care for a patient. You enter what the patient needs and where they are. CareCompass gives back a **call plan**:

- three facilities to call, in order, with what to ask each one
- a government fallback
- a referral slip in English or Hindi
- a "Tell the family" WhatsApp message

It was built for the Databricks DAIS 2026 datathon on Virtue Foundation Indian healthcare facility data.

See [OVERVIEW.md](OVERVIEW.md) for features and a code map, [SYSTEM_OVERVIEW.md](SYSTEM_OVERVIEW.md) for the system design, and [CLEANING_README.md](CLEANING_README.md) for the data pipeline.

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

## Notes

- **Where to edit the engine:** `engine/recommend_engine.py` is generated from `notebooks/06_recommend.py`. Edit the notebook, not the generated file.
- **Facility data is self-reported** and may be wrong. Always confirm by phone before a patient travels.
- **Demo phone number:** `config/demo_phone.json` can make every "Call first" show one demo number. It is off by default.
