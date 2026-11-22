"""Capability matching against the real verified data (needs Databricks).
Run: DATABRICKS_CONFIG_PROFILE=datathon .venv/bin/python -m pytest tests/test_matching.py -q -s"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capabilities as caps  # noqa: E402
from engine import find_care as fc  # noqa: E402
from engine import recommend_engine as eng  # noqa: E402


def top_ids(out):
    return [r["unique_id"] for r in out["results"]]


def from_text(text, place):
    """What the app does: type-ahead → first suggestion → find_care."""
    s = caps.suggest(text)
    return fc.find_care([s["suggestions"][0]["capability_id"]], place)


@pytest.fixture(scope="module")
def fac():
    return fc.facilities().set_index("unique_id")


def test_xray_spellings_identical_results():
    outs = {t: from_text(t, "Bengaluru") for t in ["x ray", "x-ray", "xray"]}
    first = top_ids(outs["x ray"])
    assert len(first) == 3
    for t, o in outs.items():
        assert top_ids(o) == first, (t, [r["name"] for r in o["results"]])


def test_xray_bengaluru_no_dental_or_off_domain_clinic(fac):
    out = fc.find_care(["xray"], "Bengaluru")
    loc, radius = out["location_resolution"], out["how_we_decided"]["final_search_radius_km"]
    # precondition from the spec: a hospital / diagnostic centre with X-ray in its EQUIPMENT exists in range
    d = eng._hav_np(loc["lat"], loc["lon"], fac["final_lat"].to_numpy(float), fac["final_lon"].to_numpy(float))
    in_range = fac[(d <= radius)]
    good = in_range[in_range["cap_mentions"].map(lambda m: any(f == "equipment" for f, _, _ in m.get("xray", [])))
                    & ((in_range["facility_type"].str.lower() == "hospital")
                       | in_range["main_domain"].isin(["general_imaging"])
                       | in_range["name"].str.contains("diagnos|imaging|scan", case=False))]
    assert len(good) >= 3, "precondition: hospitals/diagnostic centres with X-ray equipment in range"
    for r in out["results"]:
        row = fac.loc[r["unique_id"]]
        print(f"  {r['role']:24} {r['name'][:50]:50} type={row['facility_type']} main={row['main_domain']}")
        assert row["main_domain"] != "dental", r["name"]
        assert row["main_domain"] in ("general", "general_imaging"), (r["name"], row["main_domain"])


def test_broken_arm_flow_has_no_dental(fac):
    s = caps.suggest("broken arm")
    ids = [x["capability_id"] for x in s["suggestions"]][:2]
    out = fc.find_care(ids, "Patna", condition="broken arm")
    assert out["results"], out["search_notes"]
    for r in out["results"] + out["fallback_results"]:
        assert fac.loc[r["unique_id"], "main_domain"] != "dental", r["name"]


@pytest.mark.parametrize("place", ["Darbhanga", "Bengaluru", "Patna", "Mumbai"])
def test_dialysis_same_as_old_path(place):
    new = fc.find_care(["dialysis"], place)
    old = eng.recommend("dialysis", place)
    assert new["path"] == "checked_service"
    assert top_ids(new) == top_ids(old), ([r["name"] for r in new["results"]], [r["name"] for r in old["results"]])


def test_dialysis_generic_matcher_not_worse(capsys):
    """Informational + guard: the generic matcher (no hand-tuned rules) must not return PASSING top picks."""
    for place in ["Darbhanga", "Bengaluru", "Patna"]:
        g = fc.find_care(["dialysis"], place, route_checked=False)
        for r in g["results"]:
            assert r["overall_level"] in ("LISTED", "SPECIFIC"), r


def test_multi_select_is_and_with_lowest_level():
    out = fc.find_care(["emergency_247", "ecg"], "Darbhanga", condition="chest pain")
    for r in out["results"]:
        lv = {x["capability_id"]: x["level"] for x in r["capability_levels"]}
        assert set(lv) == {"emergency_247", "ecg"}
        assert r["overall_level"] == min(lv.values(), key=caps.LEVELS.index)
        assert "NONE" not in lv.values() and "PASSING" not in lv.values()


def test_chest_pain_banner_reaches_results():
    out = fc.find_care(["emergency_247", "ecg"], "Darbhanga", condition="chest pain")
    assert out["banner"] and "108" in out["banner"]
    assert out["how_we_decided"]["urgency"] == "emergency"


def test_passing_only_never_top_pick():
    out = fc.find_care(["burn_care"], "Leh", route_checked=False)
    for r in out["results"]:
        assert r["overall_level"] in ("LISTED", "SPECIFIC")
    for r in out["fallback_results"]:
        assert r["role"] == "Nothing better nearby"
