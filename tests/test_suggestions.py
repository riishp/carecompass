"""Type-ahead suggestions — pure Python, no Databricks.   Run: .venv/bin/python -m pytest tests/test_suggestions.py -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import capabilities as caps  # noqa: E402


def ids(text):
    return [s["capability_id"] for s in caps.suggest(text)["suggestions"]]


def test_broken_arm_suggests_fracture_xray_ortho_not_dental():
    got = ids("broken arm")
    assert {"fracture_treatment", "xray", "ortho_surgery"} <= set(got), got
    assert "dental_xray" not in got, got


def test_broken_arm_while_typing():
    for partial in ["bro", "broke", "broken a", "broken ar"]:
        assert "dental_xray" not in ids(partial), partial
    assert "fracture_treatment" in ids("broken ar")


def test_xray_spellings_identical():
    variants = ["x ray", "x-ray", "xray", "X-Ray", "X RAY", "x.ray"]
    results = {v: caps.suggest(v)["suggestions"] for v in variants}
    first = results[variants[0]]
    assert first and first[0]["capability_id"] == "xray", first
    for v, r in results.items():
        assert r == first, (v, r)


def test_normalisation():
    assert caps.norm("X-Ray") == caps.norm("x ray") == caps.norm("xray") == "xray"
    assert caps.norm("C-section") == "csection"
    assert caps.norm("24x7 Emergency") == caps.norm("24/7 emergency") == "24 7 emergency"
    assert caps.norm("Orthopaedic") == "orthopedic" and caps.norm("Haemodialysis") == "hemodialysis"


def test_chest_pain_shows_108_banner():
    s = caps.suggest("chest pain")
    assert s["banner"] and "108" in s["banner"]
    assert s["urgency"] == "emergency"
    assert ids("chest pain")[:3] == ["emergency_247", "ecg", "cath_lab"]


def test_no_banner_for_planned():
    assert caps.suggest("x ray")["banner"] is None
    assert caps.suggest("cataract")["banner"] is None


def test_dental_terms_go_to_dental_xray():
    assert ids("opg")[0] == "dental_xray"
    assert ids("dental x ray")[0] == "dental_xray"


def test_condition_domains_restrict():
    # pregnancy suggests maternity things, never dental / eye / cancer capabilities
    doms = {caps.catalog()["by_id"][i]["domain"] for i in ids("pregnant")}
    assert doms <= {"maternity", "newborn", "general_imaging", "lab"}, doms


def test_under_three_chars_gives_nothing():
    assert ids("xr") == [] and ids("") == []


def test_labels_show_counts():
    s = caps.suggest("x ray")["suggestions"][0]
    assert s["label"].startswith("X-ray · ") and s["label"].endswith(" facilities")


def test_longest_match_keeps_dental_out_of_general_xray():
    found = caps.find_mentions({"equipment": ["Digital dental X-ray (RVG)", "OPG machine"]})
    assert "xray" not in found and "dental_xray" in found
    found = caps.find_mentions({"equipment": ["Digital X-ray machine 500 mA"]})
    assert "xray" in found and "dental_xray" not in found


def test_traps_from_real_data():
    f = caps.find_mentions({"procedures": ["Cast partial denture", "Gastric bypass surgery", "Full mouth rehabilitation"],
                            "equipment": ["Ultrasonic cleaner", "Dental chair with integrated delivery system"]})
    assert not ({"fracture_treatment", "cardiac_surgery", "physiotherapy", "ultrasound", "normal_delivery"} & set(f)), f


def test_llm_ids_validated_against_catalog():
    assert caps.validate_llm_ids(["xray", "made_up", "xray", "ct_scan"]) == ["xray", "ct_scan"]


def test_new_areas_reachable_from_patient_words():
    assert ids("cannot hear")[0] == "ent"
    assert ids("skin rash")[0] == "dermatology"
    assert ids("blood test")[0] == "lab_tests"
    assert ids("child doctor")[0] == "paediatrics"
    assert ids("depression")[0] == "mental_health"


def test_crisis_phrases_show_helpline_not_ambulance_text():
    s = caps.suggest("suicidal")
    assert s["banner"] and "14416" in s["banner"]
    assert "108" in caps.suggest("snake bite")["banner"]
