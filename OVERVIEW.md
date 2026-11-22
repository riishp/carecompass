# CareCompass — functionality and code overview

*State as of 2026-11-22. Team workspace: `dbc-7caddb86…` (CLI profile `datathon`). App: https://carecompass-7474658975840995.aws.databricksapps.com (start it before use — it is stopped to save credits).*

---

## 1. What the app does

A coordinator types **what the patient needs** and **where they are**, and gets a **call plan**: three facilities to call in order, what to ask each, a government fallback, and a referral slip to send the family.

### Tabs
The app has one view, Find care. Model 1, Model 2 and Combined were built and evaluated (see Results) but are hidden in the app (`SHOW_MODEL_TABS = False` in `app.py`).

*Care deserts* and *About the data* are hidden (`SHOW_EXTRA_TABS = False` in `app.py`; code kept, not executed).

### A search, step by step (Find care)
1. **Need** — type-ahead (`streamlit-searchbox`): conditions ("broken arm", "chest pain") or services ("X-ray"). Conditions add their top services as chips; related ones are offered as "also consider". No local match → "Ask AI" (Llama picks from the catalog only). Emergencies show the red 108 banner; self-harm phrases show the Tele-MANAS 14416 helpline.
2. **Where are you?** — village, town, district, PIN, "lat, lon", or "Place, State". Ambiguous names ask (e.g. Aurangabad → Bihar or Maharashtra); a state asks which district; typos are offered, never guessed.
3. **Results**: 108 banner (emergencies) → "Showing results near X — change place" → **call plan** (call #1 with questions → #2 → #3 → government fallback) → Hindi toggle, Copy slip, WhatsApp → numbered **map** → **cards** (call first/second/third: name, red Call button, travel time, evidence label, "What to ask" dropdown; "More details" = trust checks, quoted evidence, location note) → "Nothing better nearby" → **Why not these?** (real reasons only) → **How we decided** (rules + ranking factors).
4. Classic free-text search is kept in an expander at the bottom (original `recommend()`).

**Demo setting**: `config/demo_phone.json` can make every "Call first" show one demo number (`"enabled": true`, plus `phone_display` and `tel`). It is off: real numbers from the data are shown.

---

## 2. Data (Unity Catalog `workspace.datathon_2026`)

Built by the cleaning pipeline (`00_Full_Cleaning_Pipeline`, 8 steps): raw → Bronze → Location → `03_geo` → NFHS → Specialties → Silver (with the 2-of-3 location vote) → Gold → travel times. Then `09b_Gold_Verified` loads the teammate's government-directory-verified CSV into **`gold_facility_search_verified`** (9,969 facilities; 672 located from the government directory) — this is what the app ranks. Full cleaning write-up: `CLEANING_README.md` in the shared folder.

---

## 3. Code map (local folder `/Users/rishikapachava/carecompass`, deployed to `/Workspace/Shared/Datathon 2026/app`)

| File | Role |
|---|---|
| `app.py` | Streamlit UI: tabs, inputs, results view (call plan, slip, map, cards), styling. |
| `capabilities.py` | **Pure Python.** Normalisation (x-ray = x ray = xray), type-ahead suggestions (conditions first, limited to their domains), evidence levels SPECIFIC / LISTED / PASSING with penalties, longest-match-wins term matching. |
| `catalog/capabilities.json`, `catalog/conditions.json` | The service catalog (30 services, counts = facilities that *list* it) and 46 patient-phrase conditions. Swap in the full catalog without code changes. |
| `engine/recommend_engine.py` | The team's `06_recommend` engine, **copied automatically** from the notebook by `tools/sync_engine.py` at each deploy (one location resolver, scoring, top-3 picker). Never edit here — edit the notebook. |
| `engine/find_care.py` | Capability search: matches any catalog service, multi-select = AND, keeps the 7 hand-tuned checked services (single checked service → original `recommend()`), LLM fallback. |
| `ui_results.py` | Turns any result into the call plan: cards, trust chips, evidence quotes, questions, government fallback, "why not", the slip (EN/HI), WhatsApp link. Display only. |
| `versions.py` | Model 1 / Model 2 / Combined re-ranking of Find care's search (same place and radius). |
| `data.py` | SQL-warehouse reads (care-desert data). |
| `config/` | `call_checklists.json`, `what_to_bring.json`, `slip_en.json`, `slip_hi.json`, `demo_phone.json`. |
| `.streamlit/config.toml` | Theme colours. |
| `tools/deploy.sh` | Sync engine → upload app → deploy. |
| `tools/run_model1.py`, `tools/run_model2.py` | Run the teammates' model notebooks **locally** (no credits) → `catalog/model1_levels.csv`, `catalog/model2_capability.parquet`. |
| `tools/build_stub_catalog.py` | Recount catalog `facility_count`. |
| `tests/` | `test_suggestions.py`, `test_matching.py`, `location_eval.py`, `consistency_test.py`, `model_eval.py` (accuracy test 1), `model_eval_top3.py` (accuracy test 2). |

### How ranking works (Find care)
Candidates = facilities that list every selected service within the service's distance limit (widened 1.5× / 3× if fewer than 3). Score = evidence + access (distance & location uncertainty) + readiness (type, doctors, beds) + trust (phone, website, location certainty, not a merged record), weighted by urgency (emergency → closeness counts most). Top 3 = best overall, closest credible, strongest evidence. Passing-only mentions never become top picks.

### How the models rank (hidden in the app)
| Version | Rule |
|---|---|
| Model 1 | best evidence level (Specific > Listed), then nearest, then directory-confirm % |
| Model 2 | only facilities that mention the service; best capability (Likely > Possible), then nearest, then capability % |
| Combined | best finder label (Strong > Good > Call to confirm), then nearest; Conflict / Doubtful / Not shown never top picks |

---

## 4. The models

- **Model 1 — evidence levels.** Original notebook covers 18 high-stakes services. The app uses **`Model 1 - Evidence Levels (all app services)`**, a marked copy extended to all 33 services the app needs (rules version `v2-all-app-services`; original unchanged). Output: level + quote + reasons + directory-confirm %.
- **Model 2 — facility capability.** 54 services, never reads wording; learns from the government directory which *kinds* of facility run each service. Output: LIKELY / POSSIBLE / UNLIKELY / UNKNOWN + %, drivers, and the combined finder label.
- Both run **on the laptop** from the two CSVs in the team volume (`gold_facility_search_services.csv`, `hospital_directory.csv`) — zero credits. Rerun the two `tools/run_model*.py` scripts if a model or the data changes, then deploy.

---

## 5. Accuracy (offline, against the government directory)

**Test 1 — scoring every (facility, service)** · 241 directory-matched hospitals × 27 services, held-out for Model 2:
Current 0.60 · Model 1 0.62 · Model 2 0.66 · **Combined 0.67** (average AUC; 0.5 = coin flip).

**Test 2 — the top 3 users see** · 40 random districts × 10 services:

| Version | Directory confirms (of checkable picks) | Median distance |
|---|---|---|
| **Current** | **74%** (39/53) | 66 km |
| Model 1 | 57% (30/53) | 68 km |
| Model 2 | 59% (44/75) | 68 km |
| Combined | 60% (46/77) | 76 km |

Reading: Combined scores facilities best; Current produces the most practical top 3 (it balances distance). Only ~5% of picks are checkable by the directory, so test 2 differences are suggestive, not conclusive. The directory lists coarse departments only, and both models learned from it.

---

## 6. Running it (credits)

- The team workspace is on a free/top-up allowance shared by everyone. The **app uses compute every hour it is running** — keep it stopped except for demos.
- Start + deploy + warm up: `databricks apps start carecompass -p datathon` → `tools/deploy.sh` → do one search (first search after a start takes 20–60 s).
- Everything else (editing, model runs, accuracy tests, the suggestion tests) runs on the laptop for free.

## 7. Open items
- Ranking bug: when the nearest credible facility is a "possible duplicate" record, `pick_top3` can return no top picks (seen near Secunderabad). One-line fix proposed, not applied (needs approval).
- Consistency: "560001" vs "Bangalore" give different top 3 (near-tie ranking); fix proposed, not applied.
- "dialysis + Atlantis" asks "did you mean Atanati?" instead of "unknown place" (fuzzy cutoff).
- Two suspect government-directory matches (Joshi Hospital Pune → Jalna; OM Clinical Lab → Voras).
- Model weights that look backwards (Model 1: "contradicted" raises %; Model 2: "bed count published" lowers capability) — for the model owners.
- Hand-check of ~30 recommendations (accuracy test 3) not done yet.
- Self-contained app (data bundled as files) would remove the first-search freeze and allow a backup deploy in the personal workspace.
