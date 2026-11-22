# CareCompass — How the data was cleaned

**What this covers:** every step from the three raw shared tables to the final Gold tables the app and the analysis use. For each step: what it does, the rules, the real numbers from the last run, and what we deliberately left alone.

**Where things live**
- **Code:** the notebooks in `/Workspace/Shared/Datathon 2026/`. Each notebook's first cell repeats its rules.
- **Data:** Catalog → `workspace` → `datathon_2026`. The shared source tables are read-only and never modified.
- **Proof:** every rule logs how many rows it touched to the `cleaning_audit` table. Every notebook ends with validation checks and **stops with an error if any check fails**, so nothing downstream gets built on bad data.
- **Re-run everything:** `00_Full_Cleaning_Pipeline` (one button, about 10–15 min). Its `start_at` setting re-runs only part of the pipeline.

---

## The flow

```
RAW (read-only, shared)
  facilities (10,088)    india_post_pincode_directory (165,627 offices)    nfhs_5_district_health_indicators (706)
        │                              │                                              │
        ▼                              ▼                                              ▼
 [1] Raw → Bronze             [2] 03 Location                                  [3] 06 NFHS
 facilities_bronze (9,983)    facilities_located, pin_lookup, pincode_clean    nfhs_clean (705)
 facilities_quarantine (93)                                                    nfhs_district_crosswalk
        │                              │                                              │
        │        [4] 07 Specialties ── specialty_map (150 standard codes)             │
        ▼                              ▼                                              ▼
                  [5] 08 Silver ── facilities_silver (9,983, every column + flags)
                               │
                               ▼
                  [6] 09 Gold ── gold_facility_search (9,969)   gold_district_summary (705)

 Geography layer:  03_geo ── pin_centers (19,586), geo_lookup (183,339), locate()
                   04_travel_times_v1 ── travel_times (5,530,517 PIN → facility pairs ≤ 150 km)
                   (03_geo also feeds the Silver location vote; travel times read Gold)
```

**Row count reconciliation:** 10,088 raw facilities − 11 exact duplicates − 1 same-facility duplicate − 93 quarantined = **9,983 in Bronze and Silver**. Gold search then sets aside 14 more (listed in step 6), leaving **9,969**.

---

## 1. Raw → Bronze (`Data Cleaning: Raw to Bronze`)

General cleaning of the facilities table. It works on copies (`facilities_raw_copy`, `pincode_directory_raw_copy`).

| Rule | Rows |
|---|---|
| Pincode directory: offices with impossible coordinates (e.g. lat 620) set to NULL | 2,604 |
| Fake-missing text (`"null"`, `"nan"`, `"[]"`, `"-"`, `""`…) → real NULL; spaces trimmed | all text columns |
| Exact duplicates (same `unique_id`): kept the most complete copy | 11 removed |
| Same name + address under different IDs | 1 removed |
| `farmacy` → `pharmacy` | 10 |
| Missing facility type → `unknown` (kept) | 126 |
| **Quarantined** (moved to `facilities_quarantine` with a reason, not deleted): columns shifted so facility type holds coordinates/URLs | 19 |
| Quarantined: name missing | 54 |
| Quarantined: name is long free text (description landed in the name column) | 20 |
| Pincodes repaired (`"201 301"` → `"201301"`) | 156 |
| Pincodes still invalid after repair → NULL | 76 |
| Valid-format pincodes not in the India Post directory (kept, flagged) | 205 |
| State: 238 different raw spellings → the 36 official states/UTs; the pincode's state is the source of truth | 9,690 taken from pincode |
| State written on the record disagrees with the pincode's state (`state_conflict`) | 59 |
| `capacity`, `numberDoctors` → integers; implausible doctor counts flagged | 105 flagged |
| List columns (specialties, procedures, equipment, capabilities, phones…) parsed from JSON text into real arrays | — |
| Columns > 95% empty dropped: `acceptsVolunteers`, `countries`, `area` | 3 columns |

## 2. Location (`03_location_cleaning_and_uncertainty`)

Gives every facility the most trustworthy point, an error estimate in km, and a plain-English explanation.

- **Cleans the PIN directory:**
  - 6-digit PINs only
  - standard state and district names
  - coordinates that are NA, 0 or outside India → NULL
  - swapped lat/lon repaired
- **Matches each facility to its PIN**, best evidence first:

  | Tier | Meaning | Facilities |
  |---|---|---|
  | 1 | PIN + city + state all agree | 5,255 |
  | 2 | PIN + state agree (city missing or not matched) | 4,068 |
  | 3 | PIN valid but state contradicts it (`conflict_state`) | 383 |
  | 4 | No usable PIN; city + state found in the directory | 187 |
  | 5 | Nothing matched | 183 |

- **Final point:**
  - the facility's own coordinates if valid (9,952)
  - otherwise its PIN centre (35)
  - otherwise its place centre (1)
  - otherwise none (88)
- **Uncertainty score (0–1)** combines match tier, district ambiguity, estimated error and a "looks like two records merged" flag (1,711 rows). Every facility gets a tier: low / medium / high uncertainty.
- **Plain-English evidence** for every facility, plus warnings: `state_conflict`, `possible_merged_record`, `non_india`, `coords_disagree`, `ambiguous_pin`, `no_location`.

> The error estimates (e.g. "2 km when coordinates agree with the PIN") are judgment calls to review, not measurements.

## 3. NFHS health data (`06_NFHS_Cleaning`)

NFHS-5 is India's National Family Health Survey (2019–21): one row per district, about 110 health indicators. In CareCompass it measures **health need**.

| Rule | Rows |
|---|---|
| District names trimmed (704 of 706 had trailing spaces) | 704 |
| State names → the 36 official names (`Maharastra`, `NCT of Delhi`, `&` → `and`) | 74 |
| Removed Mizoram "Chandel", an exact copy of Manipur's Chandel (Chandel is in Manipur) | 1 |
| Text indicators → numbers. `(56.5)` means NFHS based it on 25–49 cases → kept, flagged `small_sample` | 5,064 values |
| `*` means NFHS suppressed it (< 25 cases) → NULL, flagged `suppressed` | 4,120 values |
| NFHS district → India Post district: exact name | 613 |
| … spelling variant (fuzzy, every one checked by hand) | 53 |
| … renamed districts, hand-written (Gurgaon → Gurugram, Faizabad → Ayodhya, Allahabad → Prayagraj…) | 39 |

**Result:** all 705 districts match an India Post district. Flags live in `nfhs_value_flags`, so analysis can choose whether to use small-sample values.

## 4. Specialties (`07_Specialty_Standardization`)

The raw data has **2,927 different specialty strings**. These are mapped to **150 standard codes**, each used by at least 50 facilities.

| How a string was matched | Strings | Facility tags |
|---|---|---|
| Already a standard code | 150 | 110,714 |
| Format variant (`Cardiology` vs `cardiology`) | 172 | 744 |
| Synonym (hand-written list, about 110 entries, e.g. `GI Surgery` → gastroenterology) | 185 | 1,589 |
| Narrow sub-specialty rolled up to its parent | 420 | 1,655 |
| Profession, not specialty (`Cardiologist` → cardiology) | 44 | 66 |
| Fuzzy spelling match (≥ 0.92 similarity) | 27 | 71 |
| Left as `other` (rare, kept in `specialties_other`) | 1,929 | 3,370 |

**97.1% of all facility specialty tags** end up in a standard code.

## 5. Silver (`08_Silver_Facilities`)

Joins Bronze, Location, Specialties and the NFHS crosswalk into **one row per facility with every column and flag** (9,983 rows).

| Rule | Rows |
|---|---|
| Coordinates come from Location (`final_lat/lon`) with a `location_confidence` of high / medium / low (before the vote) | 7,887 / 763 / 1,333 |
| **Location vote** (coordinates vs PIN vs city, "agree" = within 30 km), for facilities whose coordinates are > 30 km from their PIN: | 1,723 checked |
| … PIN and city agree, coordinates 75–675 km away → **coordinates were wrong, point moved to the PIN area** (e.g. Agrawal Eye Hospital, Meerut: coordinates were 675 km off) | 800 |
| … coordinates and city agree → **PIN was wrong**; coordinates kept, district re-derived from the coordinates | 367 |
| … nothing agrees, or the city isn't in the directory → kept as is, flagged `location_unresolved`, stays low confidence | 556 |
| Resolved facilities become medium confidence. The original point is kept in `latitude_located` / `longitude_located`, and the decision in `location_vote` | |
| State restored to the written state when the pincode is > 200 km away (the pincode was the error). E.g. Aravind Eye Hospital: Delhi → Telangana | 15 |
| Cities renamed to their current name (Bangalore → Bengaluru, Gurgaon → Gurugram…); raw kept in `city_raw` | 457 |
| Possible duplicates (same name + city): one primary record per group, the rest flagged | 14 in groups, 7 non-primary |
| Phone numbers shared by ≥ 5 facilities with different names in ≥ 2 cities (call centres, aggregators) blanked from `official_phone` | 14 numbers, 237 facilities |
| Doctor counts > 2,000 → NULL | 7 |
| `government` operator type unified to `public` | 2 |
| Non-India country value flagged (`country_suspect`) | 3 |
| Junk specialty items dropped (JSON fragments, no letters) | 4 |
| Facilities with at least one standard specialty | 9,946 |
| Linked to an NFHS district | 9,855 (98.7%) |
| … of which via the parent district, for districts created after NFHS-5 (NTR → Krishna, Chengalpattu → Kancheepuram…) | 182 |

## 6. Gold (`09_Gold`)

Two tables built for use. Every Gold column comes from Silver; nothing new is cleaned here.

**`gold_facility_search`** (9,969 rows) powers the "service near location" finder:
- primary records only (7 duplicates excluded)
- not `country_suspect` (3 excluded)
- has a location (4 column-shift text fragments excluded)
- includes `search_text` for keyword search
- location confidence: 7,569 high, 1,558 medium, 842 low (low was 1,323 before the location vote)

**`gold_district_summary`** (705 rows), for the care-desert analysis:
- one row per NFHS district
- **need:** 18 key NFHS indicators, renamed to readable names
- **supply:** facility counts in total and for 10 service groups: emergency, maternity, pediatrics, cardiology, dialysis/kidney, cancer, mental health, eye, dental, diabetes
- each service group also has a `_confident` count that only includes high/medium-confidence locations
- 175 districts have **zero** facilities in the dataset, which is itself a finding
- 125 facilities have no NFHS district and are not counted

## 7. Geography layer (`03_geo`, `04_travel_times_v1`)

**`03_geo`:** turns the post office directory into a lookup for any user input.
- **`pin_centers`:** one point per PIN. It ignores:
  - 14,626 office coordinates that are missing or outside India
  - **6,448 filler values**: the same coordinate reused across a whole area. For example, lat 15.5934 is used across Gujarat and Kerala, which puts those offices in the Arabian Sea.
  - **4,207 offices > 200 km from their PIN area's main cluster**: for example, 118 Firozabad offices placed 300 km south

  PIN areas (the first 3 digits) are used instead of district names because district labels are sometimes wrong (Bastar PINs labelled "Raipur"). 100 PINs with no state take it from the nearest PIN. **All 19,586 PINs are located.**
- **`geo_lookup`:** 19,586 PINs, 746 districts and 163,007 towns (office names minus `B.O` / `S.O` / `H.O` / `G.P.O`), each with its nearest PIN.
- **`locate(text)`** accepts a PIN, a place name (typos OK: "patana" → Patna) or "lat, lon", and returns lat, lon, nearest PIN, district and state.
- **Check:** districts the raw data got badly wrong (Patan, Vadodara, Firozabad, Azamgarh, Raipur…) now sit within 30 km of the real place.

**`04_travel_times_v1`:** every (PIN, Gold facility) pair within 150 km straight-line, using the vote-corrected Gold coordinates: **5,530,517 rows**. A BallTree radius search finds the pairs, and a brute-force check on 200 PINs matches it exactly.
- `minutes_car = km × 1.4 / 40 × 60` (road detour factor 1.4, 40 km/h)
- `minutes_walk = km × 1.2 / 4.5 × 60`
- 22 PINs have no facility within 150 km.
- **v2 (in progress by a teammate, ArcGIS friction mapping):** friction-surface least-cost paths. Same columns, so nothing downstream changes.

---

## What we did *not* fix (and why)

| Issue | Why it's left |
|---|---|
| 842 facilities with **low location confidence** (556 unresolved by the vote, the rest missing a PIN or coordinates) | The three clues disagree and we can't verify them without outside data, so they're flagged rather than guessed. The app should show them as approximate. A hand-check of a sample against websites is the next step. |
| **No population data** | NFHS has none, so supply is counted per district, not per 100k people. |
| Some facility **names contain full addresses** | Source data; the name is still searchable. |
| Some **eye hospitals list cardiology** | We clean the format, not the facility's own claims. |
| **Purba Bardhaman** (11 facilities) has no NFHS row | We found no matching NFHS district and didn't guess one. These facilities show as "no NFHS district". |
| About 25 facilities' **district doesn't match any NFHS district** | E.g. Hyderabad listed under Andhra Pradesh, or a misspelled Visakhapatnam under Telangana. They're listed in the audit log for review. |
| Some **district labels in the post office directory are wrong** | `03_geo` fixes the coordinates; the district name still comes from the directory. |
| **Travel times are straight-line** | v1 ignores roads and terrain; v2 replaces it. |

## Checks that run every time

| Notebook | Checks | Examples |
|---|---|---|
| Raw → Bronze | validation suite | raw = bronze + quarantine + duplicates removed |
| 06 NFHS | 8 | one row per district, every % between 0–100, every district matched |
| 07 Specialties | coverage | ≥ 96% of tags standardized |
| 08 Silver | 16 | same rows as Bronze, every specialty string mapped, every far-from-PIN facility went through the vote |
| 09 Gold | 9 | every searchable facility has coordinates |
| 03_geo | 11 | every PIN located; known districts within 60 km of the real place (actual: within 30 km); `locate()` tests |
| 04 travel times | 8 | km ≤ 150, formulas exact, BallTree = brute force |
