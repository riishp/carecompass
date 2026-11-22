# CareCompass: full system overview (for ideation)

> **Context for the reader:** this describes a datathon project (Databricks Data + AI Summit 2026 datathon, Virtue Foundation dataset on Indian healthcare facilities). It covers what the product does, how every part works, the code structure, the real numbers, and the known weaknesses. The goal of sharing it is to **brainstorm improvements**. A list of open questions is at the end.
>
> Snapshot as of 2026-11-21. The service-evidence notebooks (03b/04/05/06) and the government-data comparison are still being edited by teammates. Their behaviour below is as last read.

---

## 1. What CareCompass is

Two user-facing questions:

1. **"Where can I get *this service* near *this place*?"** A patient or health worker types a need ("dialysis", "c-section", "my child needs an ICU") and a location (village, district, PIN code, or coordinates). The app returns the **3 best facilities**, each with a plain-language explanation of *why it was chosen*, *how sure we are that it offers the service*, *how sure we are of its location*, and a phone number to call ahead.
2. **"Where is care missing?" (care deserts)** For each of India's ~705 districts, health **need** (from the NFHS-5 government health survey) is compared with care **supply** (facility counts per service from our data). The districts with high need and little or no care are highlighted.

Design principle throughout: **a confident wrong answer can cost a patient hours**, so the system prefers honest uncertainty to false precision. Every location has a confidence level, every service claim has an evidence tier with quoted text, and ambiguous inputs trigger a clarifying question instead of a guess.

---

## 2. Data sources

| Source | Size | What it gives | Main problems found |
|---|---|---|---|
| `facilities` (Virtue Foundation, read-only Delta Share) | 10,088 rows, 51 cols, mostly strings | Facility name, address, PIN, city, state, lat/lon, type, operator, specialties / procedures / equipment / capabilities (scraped from facility web pages), phone, website, doctors, beds, description | Column-shifted rows, duplicates, 238 spellings of states, PINs with spaces, swapped coordinates, coordinates hundreds of km from the address, JSON fragments in list fields, self-reported (unverified) services, boilerplate specialty lists |
| `india_post_pincode_directory` | 165,627 post offices, 19,586 PINs | Office name, PIN, district, state, lat/lon: used as India's place gazetteer | Missing coords stored as the text "NA", swapped lat/lon, lat = lon, **filler coordinates** (e.g. lat 15.5934 reused across Gujarat/Kerala → Arabian Sea), whole postal divisions shifted ~300 km, wrong district labels |
| `nfhs_5_district_health_indicators` | 706 districts × ~110 indicators | Health need: insurance, institutional births, vaccination, anaemia, BP, blood sugar, stunting… | Text values with `(x)` = small sample (25–49 cases) and `*` = suppressed (< 25 cases), trailing spaces, misspelled states, a duplicated district in the wrong state, old district names |
| **Government hospital dataset** (new, teammate in progress) | — | Official hospital name → street address → coordinates | Some coordinates missing or wrong. **Team decision: where a value is present, treat it as correct (ground truth)**; missing = no comparison. Results reported as "agreement with government records". |
| Malaria Atlas / ArcGIS friction surface (teammate in progress) | — | Road- and terrain-aware travel time | Will replace straight-line travel times |

No population data is available, so there are **no per-capita figures** (supply is a count per district).

---

## 3. Architecture

Everything runs in one Databricks workspace (serverless). Data lives in Unity Catalog schema `workspace.datathon_2026`. Code is in notebooks under `/Workspace/Shared/Datathon 2026/` plus the app in `.../app/`.

```
RAW (read-only)                 CLEANING PIPELINE (00_Full_Cleaning_Pipeline, 8 steps, one button)
facilities ─────────────► 1 Raw→Bronze ─► facilities_bronze (9,983) + facilities_quarantine (93)
pincode directory ──────► 2 Location    ─► facilities_located, pin_lookup, place_lookup, pincode_clean
                          3 03_geo      ─► pin_centers (19,586), geo_lookup (183,339)
nfhs ───────────────────► 4 06 NFHS     ─► nfhs_clean (705), nfhs_value_flags, nfhs_district_crosswalk
                          5 07 Specialties ─► specialty_map (2,927 raw strings → 150 codes)
                          6 08 Silver   ─► facilities_silver (9,983, every column + flags + location vote)
                          7 09 Gold     ─► gold_facility_search (9,969)  gold_district_summary (705)
                          8 04_travel_times_v1 ─► travel_times (5.53M PIN→facility pairs ≤150 km)

SERVICE EVIDENCE + RECOMMENDER (teammates; not in the one-button pipeline)
03b coverage profile ─► service_fields_parsed (+ boilerplate flag)
04 service evidence  ─► facility_items (553k), service_requirements, facility_service_evidence (69,783)
05 service resolver  ─► service_catalog (7), service_synonyms (63), generated_mappings, query_log_service
06 recommend         ─► recommend(service_text, location_text) → top 3 + explanations; query_log_recommend

APP (Databricks App "carecompass", Streamlit)
Find care    → runs 06's recommend() via Databricks Connect (serverless Spark) + LLM (Llama 3.3 70B)
Care deserts → gold_district_summary via SQL warehouse, mapped with geo_lookup district centres
```

Conventions in every notebook:
- The first cell lists the rules in plain English.
- Source tables are never modified, and outputs are CREATE OR REPLACE.
- Every rule logs `rows_affected` to a shared `cleaning_audit` table.
- The last cell runs validation checks and **raises an error if any fail**, so the pipeline stops before building on bad data.

---

## 4. Cleaning pipeline: what each step does

### Step 1: Raw → Bronze (teammate)
- **Fake-missing values become NULL.** `"null"`, `"nan"`, `"[]"`, `"-"` and `""` all become real NULLs.
- **Duplicates removed:**
  - 11 rows with an exact duplicate `unique_id`. The most complete copy is kept.
  - 1 same-facility duplicate (same name and address under a different ID).
- **93 rows quarantined, not deleted.** A row is quarantined when its columns are shifted (the facility type holds coordinates or URLs), its name is missing, or its name is a long block of free text.
- **PIN repair.** PINs like "201 301" become "201301" (156 repaired). Invalid ones become NULL (76).
- **State = the PIN's state, with a flag on conflicts.** 238 raw state spellings are mapped to the 36 official states/UTs. The PIN's state is the source of truth; when it contradicts the written state, `state_conflict` is set.
- **List columns parsed.** Specialties, procedures and the other list columns go from JSON text to arrays.
- **Number columns cleaned.** Doctor and bed counts become integers. Implausible doctor counts are flagged.

### Step 2: Location (`03_location_cleaning_and_uncertainty`, teammate)
For each facility it produces a best point, an error estimate, an uncertainty score and a plain-English explanation.
- **Match tiers:**

  | Tier | Meaning | Facilities |
  |---|---|---|
  | 1 | PIN + city + state agree | 5,255 |
  | 2 | PIN + state agree | 4,068 |
  | 3 | PIN valid but state conflicts | 383 |
  | 4 | City + state only | 187 |
  | 5 | No match | 183 |

- **Final point:** the facility's own coordinates if valid, otherwise the PIN centroid, otherwise the place centroid.
- **Uncertainty score:** `0.30·match_penalty + 0.15·district_ambiguity + 0.40·min(error_km/50,1) + 0.15·merged_record`.
- **Warnings:** `state_conflict`, `possible_merged_record` (1,711 rows), `coords_disagree`, `ambiguous_pin` and others.

### Step 3: Geography (`03_geo`)
Turns the post office directory into a reliable gazetteer.
- **Coordinate cleaning.** An office coordinate is ignored if:
  - it's outside India's box: 14,626 offices
  - it's a **filler value**, i.e. the same lat (or lon) used by 10+ offices with 5+ different partner values within one 3-digit PIN area: 6,448 offices
  - it's **more than 200 km from its PIN area's densest cluster**: 4,207 offices. The anchor is the median of *distinct* coordinates in the densest 3×3-degree neighbourhood, so stacked identical junk can't win. Island UTs are exempt.

  PIN areas are used instead of district names because district labels are sometimes wrong, e.g. Bastar PINs labelled "Raipur".
- **`pin_centers`.** The median of each PIN's valid offices; all 19,586 PINs are located. Fallbacks are the district median, then the PIN-area anchor. 100 PINs with no state take it from the nearest PIN.
- **`geo_lookup`.** Every searchable place: PINs, 746 districts (robust median of PIN centres) and 163,007 towns (office names minus B.O/S.O/H.O/G.P.O), each with its nearest PIN (BallTree, haversine).
- **`locate(text)`.** Accepts a PIN, a place name or "lat, lon" and returns lat, lon, nearest PIN, district and state. Matching order:
  1. exact district
  2. exact town covering 2+ PINs
  3. close district spelling (≥ 0.85)
  4. exact one-PIN village
  5. fuzzy town

  Old/new name aliases are applied first (Bangalore → Bengaluru, Mangalore → Dakshina Kannada, Noida → Gautam Buddha Nagar…).
- **Validation.** Districts the raw data had badly wrong (Patan, Vadodara, Firozabad, Azamgarh, Raipur) now sit within 30 km of the real place.

### Step 4: NFHS (`06_NFHS_Cleaning`)
- Names trimmed, and states mapped to the canonical list.
- A duplicated district removed: Mizoram's "Chandel", a copy of Manipur's.
- `(56.5)` becomes 56.5 with a `small_sample` flag (5,064 values). `*` becomes NULL with a `suppressed` flag (4,120 values).
- Every NFHS district is matched to an India Post district: 613 exact, 53 fuzzy (all hand-checked) and 39 hand-written renames (Gurgaon → Gurugram, Allahabad → Prayagraj…).

### Step 5: Specialties (`07_Specialty_Standardization`)
- 2,927 raw specialty strings map to **150 standard camelCase codes**, each used by at least 50 facilities.
- Matching methods: format variants, about 110 hand-written synonyms, roll-up of narrow sub-specialties to their parent, profession names (cardiologist → cardiology), and a fuzzy match at ≥ 0.92 similarity.
- Coverage: **97.1% of facility specialty tags** map to a standard code. The rest are kept in `specialties_other`.

### Step 6: Silver (`08_Silver_Facilities`)
Joins Bronze, Location, Specialties and the NFHS crosswalk into one row per facility, then fixes:
- **State restored** when the PIN is more than 200 km from the coordinates (15 facilities). Example: Aravind Eye Hospital was labelled Delhi.
- **Cities renamed** to their current names (457). The raw value is kept.
- **Duplicates flagged:** same name + city, with one primary record per group.
- **Junk phone numbers blanked:** a number shared by 5+ facilities with different names in 2+ cities (14 numbers, 237 facilities).
- **Doctor counts over 2,000** set to NULL.
- **Operator type unified:** `government` → `public`.
- **Junk specialty items** dropped.
- **NFHS link:** 98.7% of facilities are linked, including 182 via a parent district for districts created after the survey.
- **Location vote (new).** For the 1,723 facilities whose coordinates are more than 30 km from their PIN, three clues are compared: the coordinates, the PIN centre, and the city's centre from `geo_lookup`, all within the same state.

  | Vote result | Facilities | What happens |
  |---|---|---|
  | PIN ≈ city | **800** | The coordinates were wrong. The point moves to the PIN area. |
  | Coordinates ≈ city | **367** | The PIN was wrong. The coordinates are kept and the district is re-derived. |
  | Nothing agrees, or city unknown | **556** | Flagged `location_unresolved`. |

  Resolved facilities become medium confidence. The new columns are `location_vote`, `latitude_located` / `longitude_located` (the point before the vote) and three distances.

### Step 7: Gold (`09_Gold`)
- **`gold_facility_search`** (9,969 rows) is the facility list the app and engine read.
  - It keeps primary records only: no duplicates, no non-India rows, no rows without a location.
  - Location confidence: 7,569 high, 1,558 medium, 842 low (low was 1,323 before the vote).
  - The 842 low break down as: 380 unresolved, 175 with an unknown city, 281 not checked (no PIN or no coordinates), and 6 other.
- **`gold_district_summary`** (705 rows) has one row per NFHS district.
  - **Need:** 18 readable NFHS indicators.
  - **Supply:** facility counts in total and for 10 service groups (emergency, maternity, pediatrics, cardiology, dialysis/kidney, cancer, mental health, eye, dental, diabetes), each also as a `_confident` count of high/medium-location facilities only.
  - 175 districts have **zero** facilities in the dataset.

### Step 8: Travel times (`04_travel_times_v1`)
- Every (PIN, Gold facility) pair within 150 km straight-line: **5,530,517 rows**. A BallTree radius search finds the pairs, and a brute-force check matches it exactly.
- `minutes_car = km × 1.4 / 40 × 60` and `minutes_walk = km × 1.2 / 4.5 × 60`.
- 22 PINs have no facility within 150 km.
- **v2 (teammate, ArcGIS friction surface)** keeps the same columns and keys (`pincode`, `facility_id` = Gold `unique_id`).

---

## 5. Service evidence and recommendation engine (teammates)

### 03b: service coverage profile (analysis)
- **How usable each service field is:**

  | Field | Filled | Verdict |
  |---|---|---|
  | `specialties` | 98.8% | Reliable |
  | `capability` | 98.5% | Reliable |
  | `procedure` | 87.7% | Reliable; the only place c-section appears |
  | `equipment` | 70.6% | Sparse |
  | `description` | 99.2% | Thin; median 16 words |

- **Boilerplate flag:** facilities in the top 5% by specialty count are flagged `possible_boilerplate`.
- **Mentions per service** (any match / corroborated in 2+ fields):

  | Service | Any match | Corroborated |
  |---|---|---|
  | dialysis | 2,483 | 1,456 |
  | nicu | 1,570 | 830 |
  | mri | 950 | 538 |
  | cardiac surgery | 708 | 275 |
  | c-section | 442 | 95 |
  | burn care | 64 | 26 |

### 04: service evidence scoring
- **Scope:** 7 checked services (dialysis, c_section, nicu, cardiac_surgery, angioplasty, mri, burn_care), scored for each facility × service.
- **Items:** list entries plus description. Each item is classified informative, unclear or uninformative. Addresses, amenities, doctor names and marketing count as uninformative and are ignored.
- **Matching:** each service has a **core** regex, a **supporting** regex, mapped specialty codes, and a **specificity** pattern (counts, brand names like "1.5 Tesla").
- **Tiers:**

  | Tier | Rule |
  |---|---|
  | strong | A core term in procedure/equipment/capability, plus either a second field or a specificity hit |
  | moderate | A core term in one structured field |
  | weak | Only a supporting term, a specialty, or a description mention |
  | none | Nothing matched |
  | no_data | The facility has no informative items |

- **Caps:**
  - A boilerplate facility can't be strong without a specificity hit.
  - Contradictions are capped at weak: for example a NICU at a dentist or eye-named facility, or one with ≤ 2 doctors.
- **Score:** `0.40·first core field + 0.20·second field + 0.15·specificity + 0.10·specialty + 0.05·description − 0.15·boilerplate − 0.30·contradiction`.
- **`evidence_text`** quotes the supporting items word for word.

### 05: service resolver
Turns the typed phrase into a service:
1. **Exact synonym** (63 phrases). More than one match triggers a clarification ("heart" → Heart surgery or Angioplasty).
2. **Fuzzy match** with difflib at 0.85 ("dialisis").
3. **LLM catalog pick:** the LLM (Llama 3.3 70B via `ai_query`) chooses one of the 7 services or NONE ("brain scan" → mri).
4. **LLM-generated search mapping** for real needs outside the catalog ("physiotherapy", "snakebite"):
   - it produces core terms, supporting terms and specialty codes
   - specialty codes are validated against real codes
   - terms matching more than 30% of facilities are dropped as too broad
   - the mapping is cached in `generated_mappings`
   - results are **capped at moderate** confidence
5. Otherwise "could not resolve". It never guesses.

**Benchmark** (41 queries): 73% with synonyms and fuzzy matching only, 100% with the LLM.

### 06: `recommend(service_text, location_text)`
1. **Resolve the service** using 05.
2. **Resolve the location** using its own resolver over `place_lookup` / `pin_lookup` / `pincode_clean`, with robust district centres and an uncertainty radius. It asks back when a name exists in two states ("Aurangabad").
3. **Find candidates:**
   - Matched services: strong or moderate evidence within the service's `max_km` (dialysis 100, c-section 60, NICU 100, cardiac surgery 300, angioplasty 150, MRI 200, burn care 150).
   - Generated services: term matching within 150 km.
   - If fewer than 3 qualify, the search widens to 1.5×, then 3×, then allows weak evidence, and says so.
4. **Score each candidate** (each component 0–1):

   | Component | How it's computed |
   |---|---|
   | evidence | tier base (strong 0.7 / moderate 0.45 / weak 0.15) + 0.3 × evidence_score |
   | access | `exp(−(distance + location_error + user_radius) / decay)`, decay 30 / 80 / 200 km for emergency / recurring / planned |
   | readiness | facility type, plus doctors and beds on a log scale |
   | trust | average of: has phone, has website, not a merged record, location certainty |

   **Weights by urgency:**

   | Urgency | evidence | access | readiness | trust |
   |---|---|---|---|---|
   | emergency | 0.35 | 0.40 | 0.15 | 0.10 |
   | recurring | 0.40 | 0.30 | 0.15 | 0.15 |
   | planned | 0.50 | 0.15 | 0.20 | 0.15 |

5. **Pick the top 3** with roles: "Best overall", "Closest credible option", "Strongest evidence", then "Next best". Possibly-merged records are a last resort and always carry a warning.
6. **Explanations from fixed templates**, with no LLM-written text: `headline`, `why_service` (quoted evidence), `why_location`, `why_ranked`, `contact` ("call ahead"), `warnings`, `caveat`, plus `how_we_decided`.

**Output:** `{status, service_resolution, location_resolution, results[], search_notes, how_we_decided}`. Status is one of: ok, needs_service_clarification, needs_location_clarification, unknown_service, unknown_location, no_results.

**Tests:**
- 17 of 17 benchmark scenarios pass.
- Sensitivity: when the weights shift ±20%, 73% of top-3 results stay the same on average. Emergencies are fully stable; dense metros like Mumbai MRI are volatile because many options are near-ties.

---

## 6. The app (`carecompass` Databricks App)

- **Stack:** Streamlit 1.67 and pydeck maps. Tables are read through the SQL warehouse (`databricks-sql-connector`). The engine runs through **Databricks Connect** (serverless Spark) as the app's service principal.
- **Find care:**
  - Two text boxes: what you need, and where you are.
  - It calls `recommend()` and handles each status: clarification buttons for the service or the state, honest error messages, and a "no results" note that still names the nearest evidence.
  - Results: a map of the user and the 3 results, plus a card per result with role, distance, evidence tier, quoted evidence, location explanation, why it ranked there, phone and website, warnings, and 4 score bars.
  - A "How we decided" expander summarises the process.
  - A search takes about 3–12 seconds; the first one, a cold start, takes 20–30 seconds.
- **Care deserts:**
  - You pick a service and a state. A map shows district circles: red means 0 facilities for the service, orange 1–2, green 3+. Circle size shows a need indicator matched to the service. For example, maternity uses the share of births in a facility, where lower means more need; heart care uses the share of women with high blood pressure.
  - A ranked "highest need, least care" table sits under the map.
  - It notes that a zero means "none in our data".
- **About:** data sources and how they were cleaned.
- **Deployment:**
  - `tools/sync_engine.py` copies the engine's function cells (Config → Step 6) out of the `06_recommend` notebook, with AST filtering to drop demo and test calls.
  - `tools/deploy.sh` re-syncs the engine, uploads the app and redeploys.
  - The engine's logging writes are switched off in the app, which has read-only access.
- **Permissions (app service principal):**
  - `USE SCHEMA` and `SELECT` on `workspace.datathon_2026`
  - `CAN_USE` on the SQL warehouse
  - `CAN_QUERY` on `databricks-meta-llama-3-3-70b-instruct`
- **Local app files:**
  - `app.py`: UI
  - `data.py`: warehouse queries
  - `locate.py`: the 03_geo `locate()` port, used for the state pick-list
  - `services.py`: the 10 service groups for the deserts tab
  - `engine/`: the synced recommend engine plus the Spark session

---

## 7. Known weaknesses and open issues

**Data truth**
- All service information is **self-reported** from facility web pages and unverified. There's boilerplate, and specialty lists are sometimes absurd (eye hospitals listing cardiology).
- The facility dataset is **not a census**. A district with 0 facilities may still have care that isn't in our data. The government dataset could measure coverage.
- **No population data**, so supply can't be expressed per capita.
- NFHS-5 is from 2019–21, and district boundaries have changed since.
- 842 facilities are still low-confidence on location.

**Integration gaps**
- **The engine uses pre-vote coordinates.** 06's `_base_facilities()` reads `facilities_located` instead of Gold, so the 800 corrected locations aren't used yet. The switch is planned after the Gold update.
- **The engine has its own location resolver**, built on the Location notebook's tables, separate from `03_geo.locate()`. There are two gazetteers with slightly different behaviour, and the engine doesn't accept "District, State" input.
- **Distances in the engine are straight-line.** `travel_times` isn't used in ranking yet, and the ArcGIS v2 is pending.
- **Two service taxonomies.** The engine has 7 checked services plus LLM-generated ones; the deserts tab uses 10 specialty-based service groups. They don't line up.
- The service-evidence notebooks aren't part of the one-button pipeline.

**Product**
- Only 7 services have curated evidence rules.
- Everything is English-only: no Hindi or regional-language input.
- There's no "open now" or availability information, no cost information, and no public/private filter in Find care.
- The care-desert need indicators are simple single-indicator proxies; there's no composite need score.
- Search latency is 3–12 seconds, plus a 20–30 second cold start.

**Evaluation (planned, not built)**
- **Location:** agreement with the government dataset, which also grades the location vote.
- **Service evidence:** precision per evidence tier, from about 50 hand-labelled results.
- **Input understanding:** service and place test sets of about 50 inputs each.
- **Ranking:** known-answer scenarios and a "nearest facility with the specialty" baseline.
- **Coverage:** dataset coverage versus government records.
- Separately, a hand-check of about 30 low-confidence locations is pending.

---

## 8. Constraints to respect when suggesting changes

- Short datathon timeline, with a small team working in parallel on separate notebooks.
- Serverless Databricks only, with Unity Catalog. The LLM available is Llama 3.3 70B (also Llama 4 Maverick and Llama 3.1 8B) via `ai_query` / serving endpoints.
- Source tables are read-only; new tables only.
- Teammates own Bronze, Location, 03b/04/05/06 and the ArcGIS work. Changes there should be proposals to them.
- Judges care about honesty, explainability, real-world usefulness for Indian patients and health workers, and evidence of validation.

## 9. Questions to brainstorm

1. How do we best **use the government dataset** beyond location? Options include coverage per district, adding missing facilities, a "verified" badge, or confirming a facility's type and public/private status.
2. How should we **combine the uncertainties** (service evidence, location confidence, government agreement) into something a patient understands at a glance?
3. What would make the **care-desert analysis** more convincing without population data? A composite need index, travel-time-based access (the share of PINs more than X minutes from a service), or state comparisons?
4. How can we **validate cheaply** in the time left, and which numbers would impress judges most?
5. Which **missing features** matter most to a real user in rural India: language, voice, offline/SMS, cost, public facilities, emergency mode?
6. Should the **two gazetteers and two service taxonomies** be unified, and what's the simplest way?
7. How should the **demo story** go: which 3–4 searches best show the system's honesty and value?
