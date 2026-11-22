"""Location-resolution evaluation for the recommendation engine's resolve_location().

Usage (from the project root, with the engine synced into engine/):
    DATABRICKS_CONFIG_PROFILE=datathon .venv/bin/python tests/location_eval.py <label>

Builds a fixed test set once (tests/location_cases.csv, seed 7) from 03_geo's geo_lookup / pin_centers:
  district   every district name (names shared by several states appear once; any of them counts)
  typo       300 district names with one auto typo: drop / swap / double a letter
  alias      old / informal names → expected place (Bangalore, Bombay, Vizag, UP…)
  pin        200 sampled PINs
  state      36 state/UT names + common abbreviations — the right answer is a clarification
  town_state 200 "Town, State" inputs (towns covering 2+ PINs)

Outcome per input:
  correct    resolved within 30 km of the expected place
  clarified  the engine asked the user to choose (status 'ambiguous')
  wrong      resolved, but > 30 km from the expected place  ← "confidently wrong"
  not_found  the engine said it could not recognise the input
Results: tests/results/location_<label>.csv and a summary printed + saved as JSON.
"""
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = ROOT / "tests" / "location_cases.csv"
OUT = ROOT / "tests" / "results"
OK_KM = 30.0
SCHEMA = "workspace.datathon_2026"

ALIAS_CASES = [  # input, expected place (geo_lookup place), state
    ("Bangalore", "bengaluru urban", "Karnataka"), ("Bengaluru", "bengaluru urban", "Karnataka"),
    ("Blr", "bengaluru urban", "Karnataka"), ("Bombay", "mumbai", "Maharashtra"),
    ("Madras", "chennai", "Tamil Nadu"), ("Calcutta", "kolkata", "West Bengal"),
    ("Gurgaon", "gurugram", "Haryana"), ("Poona", "pune", "Maharashtra"), ("Baroda", "vadodara", "Gujarat"),
    ("Trivandrum", "thiruvananthapuram", "Kerala"), ("Cochin", "ernakulam", "Kerala"), ("Kochi", "ernakulam", "Kerala"),
    ("Calicut", "kozhikode", "Kerala"), ("Vizag", "visakhapatanam", "Andhra Pradesh"),
    ("Visakhapatnam", "visakhapatanam", "Andhra Pradesh"), ("Benares", "varanasi", "Uttar Pradesh"),
    ("Banaras", "varanasi", "Uttar Pradesh"), ("Allahabad", "prayagraj", "Uttar Pradesh"),
    ("Mysore", "mysuru", "Karnataka"), ("Mangalore", "dakshina kannada", "Karnataka"),
    ("Belgaum", "belagavi", "Karnataka"), ("Gulbarga", "kalaburagi", "Karnataka"),
    ("Trichy", "tiruchirappalli", "Tamil Nadu"), ("Faizabad", "ayodhya", "Uttar Pradesh"),
    ("Noida", "gautam buddha nagar", "Uttar Pradesh"), ("Secunderabad", "hyderabad", "Telangana"),
    ("Shimoga", "shivamogga", "Karnataka"), ("Tumkur", "tumakuru", "Karnataka"), ("Bellary", "ballari", "Karnataka"),
]
STATE_ABBR = ["UP", "MP", "TN", "WB", "AP", "HP", "J&K", "MH", "KA", "KL", "GJ", "RJ", "PB", "HR", "BR", "OD",
              "TS", "UK", "CG", "JH", "Orissa"]


def hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(math.radians, (a1, o1, a2, o2))
    h = math.sin((a2 - a1) / 2) ** 2 + math.cos(a1) * math.cos(a2) * math.sin((o2 - o1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(h))


def typo(word, rng):
    w = list(word)
    idx = [i for i, c in enumerate(w) if c.isalpha()]
    kind = rng.choice(["drop", "swap", "double"])
    i = rng.choice(idx[1:-1] or idx)
    if kind == "drop":
        del w[i]
    elif kind == "swap" and i + 1 < len(w):
        w[i], w[i + 1] = w[i + 1], w[i]
    else:
        w.insert(i, w[i])
    return "".join(w), kind


def build_cases(spark):
    rng = random.Random(7)
    gl = spark.table(f"{SCHEMA}.geo_lookup").where("lat IS NOT NULL").toPandas()
    pc = spark.table(f"{SCHEMA}.pin_centers").where("coord_source = 'offices'").toPandas()
    dist = gl[gl["type"] == "district"]
    towns = gl[gl["type"] == "town"]
    all_places = set(gl["place"])
    rows = []

    def add(cat, text, pts, note=""):
        rows.append({"category": cat, "input": text, "truth": json.dumps(pts), "note": note})

    # district + same-name town in that district both count as the district (e.g. "Pune" → Pune city)
    def district_points(name):
        d = dist[dist["place"] == name]
        pts = [[r.lat, r.lon, r.state] for r in d.itertuples()]
        t = towns[towns["place"].isin([name, name + " city"]) & towns["district"].str.lower().eq(name)]
        pts += [[r.lat, r.lon, r.state] for r in t.itertuples()]
        return pts

    names = sorted(dist["place"].unique())
    for n in names:
        add("district", dist[dist["place"] == n].iloc[0]["district"], district_points(n))
    for n in rng.sample([n for n in names if len(n) >= 6], 300):
        t, kind = typo(n, rng)
        if t in all_places:
            continue
        add("typo", t.title(), district_points(n), kind)
    for text, place, state in ALIAS_CASES:
        d = dist[(dist["place"] == place) & (dist["state"] == state)]
        t = towns[(towns["place"] == place) & (towns["state"] == state)]
        pts = [[r.lat, r.lon, r.state] for r in pd.concat([d, t]).itertuples()]
        add("alias", text, pts, f"{place}, {state}")
    for r in pc.sample(200, random_state=7).itertuples():
        add("pin", r.pincode, [[r.lat, r.lon, r.state]])
    n_dist = dist.groupby("state").size()
    for s in sorted(dist["state"].unique()) + STATE_ABBR:
        if s in n_dist.index and n_dist[s] == 1:      # one-district UT: answering directly is right
            r = dist[dist["state"] == s].iloc[0]
            add("state", s, [[r.lat, r.lon, r.state]] + district_points(r.place), "single-district state")
        else:
            add("state", s, [], "expect clarification")
    big = towns[towns["n_pins"] >= 2]
    big = big[~big.duplicated(["place", "state"], keep=False)]          # unique name within its state
    for r in big.sample(200, random_state=7).itertuples():
        add("town_state", f"{r.place.title()}, {r.state}", [[r.lat, r.lon, r.state]])
    df = pd.DataFrame(rows)
    df.to_csv(CASES, index=False)
    return df


def score(row, res):
    st = res.get("status")
    if st == "ambiguous":
        return "clarified", None
    if st != "resolved" or res.get("lat") is None:
        return "not_found", None
    pts = json.loads(row["truth"])
    if not pts:
        return "wrong", None
    d = min(hav(res["lat"], res["lon"], p[0], p[1]) for p in pts)
    return ("correct" if d <= OK_KM else "wrong"), round(d, 1)


def main(label):
    import engine  # noqa: E402  (Databricks Connect session starts here)
    spark = engine._e.spark
    cases = pd.read_csv(CASES, dtype=str) if CASES.exists() else build_cases(spark)
    print(f"{len(cases)} cases:", cases["category"].value_counts().to_dict())

    def run(i):
        row = cases.iloc[i]
        t0 = time.time()
        try:
            res = engine.resolve_location(row["input"])
            outcome, km = score(row, res)
            return dict(row, outcome=outcome, km_off=km, status=res.get("status"), label_out=res.get("label"),
                        n_options=len(res.get("options") or []), seconds=round(time.time() - t0, 2), error="")
        except Exception as e:  # recorded, never hidden
            return dict(row, outcome="error", km_off=None, status=None, label_out=None, n_options=0,
                        seconds=round(time.time() - t0, 2), error=f"{type(e).__name__}: {e}"[:300])

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = pd.DataFrame(list(ex.map(run, range(len(cases)))))
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / f"location_{label}.csv", index=False)
    tab = pd.crosstab(res["category"], res["outcome"], margins=True, margins_name="total")
    for c in ["correct", "clarified", "wrong", "not_found", "error"]:
        if c not in tab.columns:
            tab[c] = 0
    tab = tab[["correct", "clarified", "wrong", "not_found", "error", "total"]]
    print(f"\n=== {label}: {len(res)} inputs in {time.time() - t0:.0f}s ===")
    print(tab.to_string())
    (OUT / f"location_{label}.json").write_text(json.dumps(tab.to_dict(), indent=1))
    errs = res[res["outcome"] == "error"]
    if len(errs):
        print(f"\n{len(errs)} ERRORS, first 5:")
        print(errs[["input", "error"]].head().to_string())
    print("\nSample confidently-wrong:")
    print(res[res["outcome"] == "wrong"][["category", "input", "label_out", "km_off"]].head(15).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "run")
