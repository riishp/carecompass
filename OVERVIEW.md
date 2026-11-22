# CareCompass — functionality and code overview

*State as of 2026-11-22. Team workspace: `dbc-7caddb86…` (CLI profile `datathon`). App: https://carecompass-7474658975840995.aws.databricksapps.com (needs a sign-in to the team workspace). The deployed app is **final and locked for judging**: no further deploys.*

---

## 1. What the app does

A coordinator types **what the patient needs** and **where they are**, and gets a **call plan**: three facilities to call in order, what to ask each, a government fallback, a referral slip, and a "Tell the family" WhatsApp message.

### Views
The app opens on one view, Find care. Model 1, Model 2 and Combined were built and evaluated (see Accuracy); they are hidden by default. Turn on the **Judge view** toggle (top-right corner) to show them as extra tabs, and to show the *Why not these?* and *How we decided* sections under the results.

*Care deserts* and *About the data* are hidden (`SHOW_EXTRA_TABS = False` in `app.py`; code kept, not executed).

### A search, step by step (Find care)
1. **Need**: type-ahead (`streamlit-searchbox`) for conditions ("broken arm", "chest pain") or services ("X-ray"). Conditions add their top services as chips under **Selected**; related ones are offered as "also consider". No local match → "Ask AI" (Llama picks from the catalog only). Emergencies show the red 108 banner; self-harm phrases show the Tele-MANAS 14416 helpline.
2. **Where are you?**: village, town, district, PIN, "lat, lon", or "Place, State". Ambiguous names ask (e.g. Aurangabad → Bihar or Maharashtra); a state asks which district; typos are offered as "did you mean…?" only above 0.80 similarity, otherwise "We couldn't find that place. Add the district/state or use a PIN code."
3. **Results**:
   - 108 banner (emergencies) → "Showing results near X — change place"
   - **Call plan**: call #1 with questions → #2 → #3 → **government fallback**. The fallback line is honest: "Government hospital, usually free. Lists <service>, not confirmed. Call first." if its record mentions a selected service, otherwise "… Not confirmed for <service>. Ask if they offer it or can refer you."
   - Actions: Hindi toggle, **Copy slip**, **Send slip on WhatsApp**, **📲 Tell the family** (see below)
   - Numbered **map**
   - **Cards** (call first / second / third): name, red Call button, travel time, evidence label, "What to ask". "More details" holds the trust chips, quoted evidence and location note.
   - "Nothing better nearby" options, then (Judge view only) **Why not these?** (real reasons only) and **How we decided** (rules + ranking factors).
4. **Travel times** are always labelled as estimates ("~25 min by car (estimated)", Hindi "(अनुमान)"): straight-line distance × 1.4 road detour at 40 km/h.
5. **Trust chips**: ✅ "In government hospital directory" only for STRONG directory matches; ⚠ "Government directory match unverified" only for the 2 hand-reviewed matches; no directory chip otherwise.
6. The classic free-text search expander was removed from the UI; its code (original `recommend()`) is kept.

### Tell the family
A popover in the call-plan action row: the family's WhatsApp number (normalised to `91XXXXXXXXXX`; invalid → hint and Send disabled), which facility (the 3 call-plan facilities or the government fallback), "Confirmed by phone?", and an optional "From". It builds a fixed template message (no LLM, no patient name, under 900 characters, English or Hindi) with the facility, Google Maps link, estimated travel time, need, confirmed/not-yet line, hospital phone, "Please bring" list (`config/family_bring.json`), the 108 line for emergencies only, and a footer. Send opens `wa.me`; the number and "From" are never stored or logged and are cleared after sending.

### Demo number
If `config/demo_phone.local.json` exists (**gitignored, never on GitHub**), "Call first" shows its number with a yellow **Demo number** tag; the slip and family message add "(demo number)". Otherwise the app reads the committed `config/demo_phone.json`, which is off with empty fields, and shows real numbers from the data. `tools/deploy.sh` uploads the local file explicitly (and removes it from the app if it was deleted locally). The live app currently has the demo number on.

---

## 2. Data (Unity Catalog `workspace.datathon_2026`)

Built by the cleaning pipeline (`00_Full_Cleaning_Pipeline`, 8 steps): raw → Bronze → Location → `03_geo` → NFHS → Specialties → Silver (with the 2-of-3 location vote) → Gold → travel times. Then `09b_Gold_Verified` loads the teammate's government-directory-verified CSV into **`gold_facility_search_verified`**, which is what the app ranks:

- 9,969 facilities; **670 located from the government directory** (631 moved to the directory's coordinates, 39 already matched it).
- Directory match: STRONG 2,024 · PROBABLE 594 · WEAK 1,578 · NONE 5,771 · UNVERIFIED 2.
- Two matches were marked **UNVERIFIED** by hand review and not applied (Joshi Hospital Pune → Jalna; OM Clinical Lab → Voras Clinical Laboratory): original location restored, low confidence, "confirm the address by phone".

The service tables the engine reads come from `03b_service_coverage_profile`, `04_service_evidence` and `05_service_resolver`. Full cleaning write-up: `CLEANING_README.md`.

---

## 3. Code map (this repo; deployed to `/Workspace/Shared/Datathon 2026/app`)

| File | Role |
|---|---|
| `app.py` | Streamlit UI: Judge view toggle, tabs, inputs, results view (call plan, slip, Tell the family, map, cards), styling. |
| `capabilities.py` | **Pure Python.** Normalisation (x-ray = x ray = xray), type-ahead suggestions (conditions first, limited to their domains), evidence levels SPECIFIC / LISTED / PASSING with penalties, longest-match-wins term matching. |
| `catalog/capabilities.json`, `catalog/conditions.json` | The service catalog (30 services, counts = facilities that *list* it) and 46 patient-phrase conditions. |
| `catalog/model1_levels.csv`, `catalog/model2_capability.parquet` | Precomputed model outputs used by the Judge view tabs. |
| `engine/recommend_engine.py` | The team's `06_recommend` engine, **copied automatically** from the notebook by `tools/sync_engine.py` at each deploy (location resolver, scoring, top-3 picker). Never edit here; edit the notebook. |
| `engine/find_care.py` | Capability search: matches any catalog service, multi-select = AND, keeps the 7 hand-tuned checked services (single checked service → original `recommend()`), LLM fallback. |
| `ui_results.py` | Turns any result into the call plan: cards, trust chips, evidence quotes, questions, government fallback line, "why not", the slip (EN/HI), WhatsApp links, Tell the family message. Display only. |
| `versions.py` | Model 1 / Model 2 / Combined re-ranking of Find care's search (same place and radius). |
| `data.py` | SQL-warehouse reads (care-desert data; hidden view). |
| `config/` | `call_checklists.json`, `what_to_bring.json`, `family_bring.json`, `slip_en.json`, `slip_hi.json`, `demo_phone.json` (+ local-only `demo_phone.local.json`). |
| `notebooks/` | Pipeline, engine and model notebooks (see README → Notes for what is and isn't included). |
| `.streamlit/config.toml` | Theme colours. |
| `tools/deploy.sh` | Sync engine → upload app (+ local demo file) → deploy. |
| `tools/run_model1.py`, `tools/run_model2.py` | Run the model notebooks **locally** (no credits) → the two `catalog/` model files. |
| `tools/build_stub_catalog.py` | Recount catalog `facility_count`. |
| `tests/` | pytest: `test_suggestions.py`, `test_family.py`, `test_matching.py`. Scripts: `location_eval.py`, `consistency_test.py`, `model_eval.py` (accuracy test 1), `model_eval_top3.py` (accuracy test 2). |

### How ranking works (Find care)
Candidates = facilities that list every selected service within the service's distance limit (widened 1.5× / 3× if fewer than 3). Score = evidence + access (distance & location uncertainty) + readiness (type, doctors, beds) + trust (phone, website, location certainty, not a merged record), weighted by urgency (emergency → closeness counts most). Ties are broken deterministically: score (to 0.01) → evidence level → distance → facility id. Top 3 = best overall, closest credible, strongest evidence. Passing-only mentions never become top picks, and the top 3 is never empty when candidates exist (a "possible duplicate" record can still be shown, with its warning).

### How the models rank (Judge view only)
| Version | Rule |
|---|---|
| Model 1 | best evidence level (Specific > Listed), then nearest, then directory-confirm % |
| Model 2 | only facilities that mention the service; best capability (Likely > Possible), then nearest, then capability % |
| Combined | best finder label (Strong > Good > Call to confirm), then nearest; Conflict / Doubtful / Not shown never top picks |

---

## 4. The models

- **Model 1 — evidence levels** (`notebooks/model1_evidence_levels.py`). Covers 18 high-stakes services. The app uses **`notebooks/model1_all_services.py`**, a marked copy extended to all 33 services the app needs (rules version `v2-all-app-services`; original unchanged). Output: level + quote + reasons + directory-confirm %.
- **Model 2 — facility capability** (`notebooks/model2_facility_capability.py`). 54 services, never reads wording; learns from the government directory which *kinds* of facility run each service. Output: LIKELY / POSSIBLE / UNLIKELY / UNKNOWN + %, drivers, and the combined finder label.
- Both run **on the laptop** from two CSVs in the team volume (`gold_facility_search_services.csv`, `hospital_directory.csv`), at zero credits, via the two `tools/run_model*.py` scripts.

---

## 5. Accuracy (offline, against the government directory)

**Location understanding** (`tests/location_eval.py`, 1,517 generated inputs): confidently wrong answers fell from 81 to 1 (correct 1,156 · asked to clarify 358 · wrong 1 · not found 2).

**Test 1 — scoring every (facility, service)** · 241 directory-matched hospitals × 27 services, held-out for Model 2:
Current 0.60 · Model 1 0.62 · Model 2 0.66 · **Combined 0.67** (average AUC; 0.5 = coin flip).

**Test 2 — the top 3 users see** · 40 random districts × 10 services (re-run after the ranking fixes above):

| Version | Directory confirms (of checkable picks) | Median distance |
|---|---|---|
| **Current** | **70%** (43/61) | 65 km |
| Model 1 | 57% (30/53) | 68 km |
| Model 2 | 59% (44/75) | 68 km |
| Combined | 60% (46/77) | 76 km |

Reading: Combined scores facilities best; Current produces the most practical top 3 (it balances distance), so the app ships Current. Only ~5% of picks are checkable by the directory, so test 2 differences are suggestive, not conclusive. The directory lists coarse departments only, and both models learned from it.

**Engine scenarios** (`06_recommend` benchmark): 18/18 pass. **pytest**: 57 pass.

---

## 6. Running it (credits)

- The team workspace is on a free/top-up allowance shared by everyone. The **app uses compute every hour it is running**; stop it when the judging is over.
- Normal cycle (not used while the app is locked): `databricks apps start carecompass -p datathon` → `tools/deploy.sh` → do one search to warm it up (first search after a start takes 20–60 s).
- Everything else (editing, model runs, accuracy tests, the suggestion and family tests) runs on the laptop for free.
- Run the pytest files by name (`pytest tests/test_family.py tests/test_suggestions.py tests/test_matching.py`); a plain `pytest tests/` also collects the `consistency_test.py` script, which exits with its result.

## 7. Open items

- **Consistency:** "560001" vs "Bangalore" still give different top 3s (2 of 3 shared). The 560001 centre is ~1 km from the city point; treating facilities within the location's uncertainty radius as equally close would fix it but changes ranking, so it was not applied.
- **Old-name typos:** "Bangalor" suggests "Bangar" villages, because fuzzy matching doesn't check the old-name aliases.
- **Joshi Hospital:** its own PIN (431204) is in Jalna, so the directory match marked UNVERIFIED may actually have been right. Needs a human look.
- **Model weights that look backwards** (Model 1: "contradicted" raises %; Model 2: "bed count published" lowers capability), for the model owners.
- **Hand-check** of ~30 recommendations (accuracy test 3) not done yet.
- **Self-contained app** (data bundled as files) would remove the first-search freeze and let people run it without Databricks.
- **No LICENSE** in the repo yet.

Fixed on 2026-11-22: empty top 3 near Secunderabad; "Atlantis" → "did you mean Atanati?"; the two suspect directory matches (now UNVERIFIED).
