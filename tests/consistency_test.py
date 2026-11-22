"""Same place, different spellings → same top 3.  Usage:
    DATABRICKS_CONFIG_PROFILE=datathon .venv/bin/python tests/consistency_test.py
Exits 1 (and prints the difference) if the top 3 differ."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402

GROUPS = [("dialysis", ["Bangalore", "Bengaluru", "560001"])]
failed = False
for service, places in GROUPS:
    tops = {}
    for p in places:
        out = engine.recommend(service, p)
        loc = out["location_resolution"]
        tops[p] = [r["name"] for r in out["results"]]
        print(f"{service} + {p!r}: {out['status']} | {loc.get('label')} ({loc.get('method')}, "
              f"{loc.get('lat') and round(loc['lat'], 4)}, {loc.get('lon') and round(loc['lon'], 4)})")
        for r in out["results"]:
            print(f"    {r['role']:24} {r['name'][:60]:60} {r['distance_km']:.1f} km")
    same = len({tuple(v) for v in tops.values()}) == 1
    print(f"==> {service}: top 3 {'IDENTICAL' if same else 'DIFFERENT'} across {places}\n")
    failed |= not same
sys.exit(1 if failed else 0)
