"""'Tell the family' WhatsApp message — pure Python, no Databricks.  Run: .venv/bin/python -m pytest tests/test_family.py -q"""
import ast
import builtins
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ui_results as ui  # noqa: E402

TARGET = {"label": "Call first: X", "name": "Ruban Memorial Hospital", "address": "Patna, Bihar 800001",
          "lat": 25.61, "lon": 85.14, "minutes": 9, "phone": "+91 61222 22222"}


@pytest.mark.parametrize("raw,expected", [
    ("9876543210", "919876543210"), ("98765 43210", "919876543210"), ("+91 98765-43210", "919876543210"),
    ("09876543210", "919876543210"), ("919876543210", "919876543210"),
    ("12345", None), ("5876543210", None), ("+1 202 555 0143", None), ("", None), ("abc", None)])
def test_number_normalisation(raw, expected):
    number, hint = ui.normalise_whatsapp(raw)
    assert number == expected
    assert (hint == "") == (expected is not None)


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_wa_link_round_trips_exactly(lang):
    msg = ui.family_message(TARGET, "Dialysis", False, "Sunita", "recurring", ["dialysis"], lang)
    url = ui.whatsapp_to("919876543210", msg)
    assert url.startswith("https://wa.me/919876543210?text=")
    assert unquote(url.split("?text=", 1)[1]) == msg
    assert " " not in url and "\n" not in url


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_names_numbers_links_unchanged_in_hindi(lang):
    msg = ui.family_message(TARGET, "x", True, "", "planned", [], lang)
    for s in [TARGET["name"], TARGET["address"], TARGET["phone"], "https://maps.google.com/?q=25.61000,85.14000"]:
        assert s in msg


def test_108_line_only_for_emergencies():
    for lang in ("en", "hi"):
        assert "108" in ui.family_message(TARGET, "x", False, "", "emergency", [], lang)
        assert "108" not in ui.family_message(TARGET, "x", False, "", "planned", [], lang)
        assert "108" not in ui.family_message(TARGET, "x", False, "", "recurring", [], lang)


def test_confirmed_lines_and_footer():
    no = ui.family_message(TARGET, "x", False, "", "planned", [], "en")
    yes = ui.family_message(TARGET, "x", True, "Sunita", "planned", [], "en")
    assert "wait for the health worker's call before travelling" in no
    assert "confirmed by phone" in yes
    assert yes.endswith("Sent by Sunita via CareCompass. Services are self-reported.")
    assert no.endswith("Sent via CareCompass. Services are self-reported.")


def test_delivery_and_surgery_extras():
    m = ui.family_message(TARGET, "C-section", False, "", "emergency", ["c_section"], "en")
    assert "MCP" in m and "sign forms" in m


def test_under_900_characters_even_worst_case():
    big = dict(TARGET, name="N" * 150, address="A" * 200)
    for lang in ("en", "hi"):
        m = ui.family_message(big, "C-section + Normal delivery + Blood bank", False, "S" * 80, "emergency",
                              ["c_section", "normal_delivery", "ortho_surgery"], lang)
        assert len(m) <= 900, len(m)


def test_number_never_in_message_and_nothing_written(monkeypatch):
    """The family number only goes into the wa.me URL: not into the message, and building it opens no files."""
    def no_open(*a, **k):
        raise AssertionError("tried to open a file")
    monkeypatch.setattr(builtins, "open", no_open)
    number, _ = ui.normalise_whatsapp("98765 43210")
    msg = ui.family_message(TARGET, "Dialysis", True, "Sunita", "planned", ["dialysis"], "en")
    url = ui.whatsapp_to(number, msg)
    assert "9876543210" not in msg and "98765" not in msg
    assert url.count("919876543210") == 1


def test_app_never_logs_or_stores_the_number():
    """Static check of app.py: the family inputs are cleared on send and never printed / logged / written."""
    src = (ROOT / "app.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "family_popover")
    body = ast.get_source_segment(src, fn)
    assert "on_click=_clear_family" in body and "[k_num, k_from]" in body
    assert "disabled=True" in body                     # no valid number → disabled button, no wa.me link
    calls = re.findall(r"\b(print|open|write|to_csv|to_parquet|saveAsTable)\s*\(|\b(logging|logger)\.|session_state\[", body)
    assert not calls, calls


def test_clear_callback_wipes_fields(monkeypatch):
    import streamlit as st
    fake = {"x_fam_num": "9876543210", "x_fam_from": "Sunita"}
    monkeypatch.setattr(st, "session_state", fake)
    src = (ROOT / "app.py").read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "_clear_family")
    ns = {"st": st}
    exec(ast.get_source_segment(src, fn), ns)
    ns["_clear_family"](["x_fam_num", "x_fam_from"])
    assert fake == {"x_fam_num": "", "x_fam_from": ""}


def test_demo_number_is_labelled_in_slip_and_family_message():
    t = {"name": "X Hospital", "address": "Patna", "lat": 25.6, "lon": 85.1, "minutes": 10, "phone": "+1 202 555 0143",
         "demo_phone": True}
    for lang in ("en", "hi"):
        msg = ui.family_message(t, "Dialysis", False, "", "recurring", ["dialysis"], lang)
        assert ui.DEMO_TAG[lang] in msg
    t["demo_phone"] = False
    assert ui.DEMO_TAG["en"] not in ui.family_message(t, "Dialysis", False, "", "recurring", ["dialysis"], "en")


def test_local_demo_file_wins_and_is_gitignored():
    cfg = Path(ui.__file__).parent / "config"
    if (cfg / "demo_phone.local.json").exists():
        assert ui._DEMO_FILE.name == "demo_phone.local.json"
    committed = json.loads((cfg / "demo_phone.json").read_text())
    assert committed["enabled"] is False and committed["phone_display"] == "" and committed["tel"] == ""
    assert "config/demo_phone.local.json" in (cfg.parent / ".gitignore").read_text().splitlines()
