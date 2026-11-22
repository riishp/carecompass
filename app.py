"""CareCompass — find care near you, and see where care is missing."""
import json
import math
import pandas as pd
import pydeck as pdk
import streamlit as st

import data
from services import SERVICE_LABELS

st.set_page_config(page_title="CareCompass", page_icon="🧭", layout="wide")

CONF_COLOR = {"high": [22, 128, 72], "medium": [214, 140, 0], "low": [140, 140, 140], "unknown": [140, 140, 140]}
CONF_TEXT = {"high": "Location confirmed", "medium": "Location approximate (PIN area)",
             "low": "Location uncertain", "unknown": "Location unknown"}


def val(x):
    """None for missing values (NULL / NaN / empty text)."""
    if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip().lower() in ("", "nan", "none")):
        return None
    return x


def link(url):
    url = val(url)
    if not url:
        return None
    return url if url.startswith(("http://", "https://")) else "https://" + url


def india_view(lat=22.5, lon=79.0, zoom=3.8):
    return pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom)


LOGO = """<svg width="58" height="58" viewBox="0 0 64 64" role="img" aria-label="CareCompass logo">
  <circle cx="32" cy="32" r="30" fill="#3B5B92"/>
  <circle cx="32" cy="32" r="23" fill="none" stroke="#FFFFFF" stroke-opacity=".35" stroke-width="2"/>
  <polygon points="32,9 38.5,32 32,55 25.5,32" fill="#FFFFFF"/>
  <polygon points="32,9 38.5,32 25.5,32" fill="#C62828"/>
  <circle cx="32" cy="32" r="7" fill="#FFFFFF"/>
  <rect x="30.4" y="27.5" width="3.2" height="9" rx="1" fill="#C62828"/>
  <rect x="27.5" y="30.4" width="9" height="3.2" rx="1" fill="#C62828"/>
</svg>"""
st.markdown(f"""<div class="cc-brand">{LOGO}<div><div class="cc-brand-name">CareCompass</div>
<div class="cc-brand-tag">Find who can treat the patient — then call in order.</div></div></div>""",
            unsafe_allow_html=True)

SHOW_EXTRA_TABS = False      # Care deserts + About the data are hidden (code kept); set True to show them again
SHOW_MODEL_TABS = False      # Model 1 / Model 2 / Combined comparison tabs are hidden (code kept)
_tab_names = (["Find care"]
              + (["Model 1", "Model 2", "Combined"] if SHOW_MODEL_TABS else [])
              + (["Care deserts", "About the data"] if SHOW_EXTRA_TABS else []))
_tabs = [st.container()] if len(_tab_names) == 1 else st.tabs(_tab_names)   # one view → no tab bar
_by_name = dict(zip(_tab_names, _tabs))
find_tab = _by_name["Find care"]
m1_tab, m2_tab, combo_tab = (_by_name.get(n) for n in ("Model 1", "Model 2", "Combined"))
desert_tab, about_tab = _by_name.get("Care deserts"), _by_name.get("About the data")

# ───────────────────────────── Find care ─────────────────────────────
# New path: type-ahead capability search (capabilities.py + engine/find_care.py, catalog/*.json).
# Classic path: the original free-text search through 06_recommend's recommend(), kept as a fallback.
# Results are shown as a call plan (ui_results.py) — display only, ranking unchanged.
import html as _html

import streamlit.components.v1 as components

import capabilities as caps
import ui_results as ui

try:
    from streamlit_searchbox import st_searchbox
    HAVE_SEARCHBOX = True
except Exception:
    HAVE_SEARCHBOX = False

st.markdown("""<style>
html {font-size: 18px;}
.block-container {max-width: 1020px; padding-top: 3.6rem;}
.cc-brand {display:flex; align-items:center; gap:.8rem; margin:.2rem 0 .6rem 0;}
.cc-brand-name {font-size:2.1rem; font-weight:800; color:#1E2A4A; line-height:1.05;}
.cc-brand-tag {font-size:.95rem; color:#55627A;}
.cc-ask {font-size:2rem; font-weight:800; color:#1E2A4A; margin:1rem 0 .35rem 0; line-height:1.2;}
.cc-help {font-size:.85rem; color:#5B6475; margin:.15rem 0 .4rem 0;}
.st-key-where_input input {font-size:1.35rem; padding:.75rem .9rem;}
.st-key-findbtn button {background:#C62828; border-color:#C62828; color:#FFFFFF; font-size:1.15rem; font-weight:700;
                        padding:.55rem 1.8rem;}
.st-key-findbtn button:hover {background:#A61F1F; border-color:#A61F1F; color:#FFFFFF;}
[data-testid="stBaseButton-pillsActive"] {background:#3B5B92 !important; color:#FFFFFF !important;
                                          border-color:#3B5B92 !important; font-weight:600;}
[data-testid="stBaseButton-pills"] {background:#E3EAF5; border-color:#B9C7DD; color:#1E2A4A;}
.cc-summary {font-size:1.3rem; font-weight:700; margin:.4rem 0 .8rem 0; line-height:1.35;}
.cc-red {background:#C62828; color:#FFFFFF; padding:.8rem 1.1rem; border-radius:10px; font-weight:700;
         font-size:1.15rem; margin:.5rem 0;}
.cc-slip {background:#FFFDF6; border:1.5px dashed #8E9AB2; border-radius:8px; padding:1rem 1.2rem; margin:.4rem 0 .8rem 0;
          font-size:1.05rem;}
.cc-slip h4 {margin:0 0 .5rem 0; font-size:1.3rem;}
.cc-slip ol {padding-left:1.3rem; margin:.3rem 0;} .cc-slip li {margin-bottom:.55rem;}
[class*="st-key-card_"] {background:#F3F7FD; border:1px solid #C9D5E8; border-left:8px solid #3B5B92;
                         border-radius:12px; padding:1rem 1.1rem .6rem 1.1rem; margin-bottom:.9rem;}
[class*="st-key-card_"][class*="_SPECIFIC"] {background:#EFF8F2; border-left-color:#1B7F4C;}
[class*="st-key-card_"][class*="_LISTED"] {background:#FCF6EA; border-left-color:#B7791F;}
[class*="st-key-card_"][class*="_PASSING"], [class*="st-key-card_"][class*="_NONE"] {background:#F3F4F6; border-left-color:#6B7280;}
[class*="st-key-card_m1_"] {background:#F4F0FB !important; border:1.5px dashed #7C5CC4 !important; border-left:8px solid #7C5CC4 !important;}
.cc-role {font-size:.95rem; color:#4A5568; font-weight:600;}
.cc-name {font-size:1.5rem; font-weight:800; color:#1E2A4A; line-height:1.25; margin:.1rem 0 .35rem 0;}
.cc-travel {font-size:1.12rem;}
.cc-level {font-size:1.12rem; font-weight:700;}
[class*="st-key-call_"] a {background:#C62828 !important; color:#FFFFFF !important; border-color:#C62828 !important;
                           font-weight:700; font-size:1.05rem;}
.cc-demo {display:inline-block; background:#FFF4D6; color:#7A5A00; border:1px solid #E8C766; border-radius:4px; padding:0 6px; font-size:.75rem; font-weight:600;}
.cc-chip {display:inline-block; border:1px solid #CBD3E1; border-radius:999px; padding:2px 10px; margin:3px 5px 3px 0;
          font-size:.9rem; background:#FFFFFF;}
.cc-quote {border-left:3px solid #9AA8C2; padding-left:.7rem; margin:.4rem 0; font-size:1.05rem;}
.cc-muted {color:#5B6475; font-size:.9rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Starting the recommendation engine…")
def engine():
    import engine as e
    return e


@st.cache_resource(show_spinner="Loading facility services… (first search only, about 20 seconds)")
def find_care_module():
    from engine import find_care as fc
    engine()
    fc.facilities()
    return fc


@st.cache_data(ttl=900, show_spinner=False)
def run_recommend(service_text: str, location_text: str) -> dict:
    return engine().recommend(service_text, location_text)


@st.cache_data(ttl=900, show_spinner=False)
def run_find_care(cap_ids: tuple, location_text: str, condition: str | None, urgency: str | None) -> dict:
    return find_care_module().find_care(list(cap_ids), location_text, condition=condition, urgency=urgency)


def start_search(service_text, location_text):            # classic path
    st.session_state["q"] = {"mode": "classic", "service": service_text.strip(), "location": location_text.strip()}


def sel():
    return st.session_state.setdefault("sel", {"ids": [], "from": None, "related": [], "urgency": None, "banner": None})


def add_caps(ids, source=None, related=(), urgency=None, banner=None):
    s = sel()
    for i in ids:
        if i not in s["ids"]:
            s["ids"].append(i)
    s["related"] = [r for r in dict.fromkeys([*s["related"], *related]) if r not in s["ids"]]
    if source:
        s["from"] = source
    if urgency:
        s["urgency"] = urgency
    if banner:
        s["banner"] = banner


def pick(value, typed=None):
    """Apply one chosen suggestion: 'cap:<id>', 'cond:<id>' or 'llm:<text>'. `typed` = what the user typed."""
    kind, _, v = value.partition(":")
    if kind == "cap":
        add_caps([v])
    elif kind == "cond":
        k = next(k for k in caps.catalog()["conditions"] if k["condition_id"] == v)
        add_caps(k["capability_ids"][:2], source=(typed or k["label"]).strip(), related=k["capability_ids"][2:],
                 urgency=k["urgency"], banner=(k.get("banner_text") or caps.DEFAULT_BANNER) if k.get("banner") else None)
    elif kind == "llm":
        ids, err = find_care_module().llm_capabilities(v)
        if err:
            st.session_state["llm_error"] = err
        elif not ids:
            st.session_state["llm_error"] = (f"We couldn't match “{v}” to a service. Try a simpler word for the problem "
                                             f"(e.g. 'fracture', 'pregnant') or the service (e.g. 'X-ray').")
        else:
            add_caps(ids[:2], source=f"{v} (AI suggestion)", related=ids[2:])


def searchbox_options(term):
    term = term or ""
    s = caps.suggest(term)
    opts = []
    for k in s["conditions"][:2]:
        names = " + ".join(caps.catalog()["by_id"][i]["name"] for i in k["capability_ids"][:2])
        opts.append((f"🩺 {k['label']} → {names}", f"cond:{k['condition_id']}"))
    opts += [(x["label"], f"cap:{x['capability_id']}") for x in s["suggestions"]]
    if not opts and len(caps.norm(term).replace(" ", "")) >= 3:
        opts.append((f"✨ Ask AI to find services for “{term}”", f"llm:{term}"))
    return opts


def red(text):
    st.markdown(f"<div class='cc-red'>🚑 {_html.escape(text)}</div>", unsafe_allow_html=True)


def numbered_map(view, loc, where_label):
    pins = view["cards"] or view["fallback_cards"][:3]
    rows = [{"lat": c["lat"], "lon": c["lon"], "num": str(c["n"]), "tip": f"{c['n']}. {c['name']}"} for c in pins
            if c.get("lat") is not None]
    if view["govt"] and view["govt"].get("lat") is not None:
        g = view["govt"]
        rows.append({"lat": g["lat"], "lon": g["lon"], "num": "G", "tip": f"Government fallback: {g['name']}"})
    if not rows:
        return
    pts = pd.DataFrame(rows)
    layers = [pdk.Layer("ScatterplotLayer", pts, get_position="[lon, lat]", get_fill_color=[30, 42, 74],
                        get_radius=120, radius_min_pixels=11, pickable=True),
              pdk.Layer("TextLayer", pts, get_position="[lon, lat]", get_text="num", get_color=[255, 255, 255],
                        get_size=13, get_alignment_baseline="'center'", get_text_anchor="'middle'")]
    if loc.get("lat") is not None:
        you = pd.DataFrame([{"lat": loc["lat"], "lon": loc["lon"], "label": "You are here", "tip": f"You: {where_label}"}])
        layers += [pdk.Layer("ScatterplotLayer", you, get_position="[lon, lat]", get_fill_color=[198, 40, 40],
                             get_radius=150, radius_min_pixels=8, pickable=True),
                   pdk.Layer("TextLayer", you, get_position="[lon, lat]", get_text="label", get_color=[198, 40, 40],
                             get_size=13, get_pixel_offset=[0, -20])]
    lat = loc.get("lat") or pts["lat"].mean()
    lon = loc.get("lon") or pts["lon"].mean()
    frame = [[r["lon"], r["lat"]] for r in rows if r["num"] != "G"] + [[lon, lat]]
    view = pdk.data_utils.compute_view(frame, view_proportion=1)      # fit every numbered pin + "You are here"
    z = view.zoom if isinstance(view.zoom, (int, float)) and math.isfinite(view.zoom) else 13.0   # all pins at one spot → NaN
    view.zoom = min(max(z - 0.6, 5), 14)
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, tooltip={"text": "{tip}"},
                             map_style=pdk.map_styles.CARTO_LIGHT), height=320)


def copy_button(text, key, label="📋 Copy slip"):
    payload = json.dumps(text)
    components.html(f"""
      <button id="b" style="width:100%;padding:.45rem .6rem;border:1px solid #1E2A4A;border-radius:8px;background:#fff;
        color:#1E2A4A;font-family:sans-serif;font-size:16px;cursor:pointer">{label}</button>
      <script>
        const t = {payload};
        document.getElementById('b').onclick = async () => {{
          try {{ await navigator.clipboard.writeText(t); document.getElementById('b').innerText = '✅ Copied'; }}
          catch (e) {{ document.getElementById('b').innerText = 'Copy blocked — use the copy icon on the slip text'; }}
        }};
      </script>""", height=54)


def call_plan(view, typed, where_label, need_label=None, ns=""):
    plan = view["plan"]
    if not plan:
        return
    steps = []
    first_qs = plan[0]["questions"]
    for i, c in enumerate(plan):
        head = f"Call #{i + 1}: <b>{_html.escape(c['name'])}</b>" + (f" — {_html.escape(c['phone'])}" + (" <span class='cc-demo'>Demo number</span>" if c.get("demo_phone") else "")
                                                         if c["phone"] else " — no phone listed")
        if i == 0:
            body = "<br><span class='cc-muted'>Ask:</span><ul>" + "".join(f"<li>{_html.escape(q)}</li>" for q in first_qs) + "</ul>"
        else:
            extra = [q for q in c["questions"] if q not in first_qs][:2]
            body = "<br><span class='cc-muted'>Ask the same questions" + (", and:" if extra else ".") + "</span>"
            body += ("<ul>" + "".join(f"<li>{_html.escape(q)}</li>" for q in extra) + "</ul>") if extra else ""
        steps.append(f"<li>{'' if i == 0 else 'If no → '}{head}{body}</li>")
    g = view["govt"]
    if g and g.get("name"):
        steps.append(f"<li>Last fallback: <b>{_html.escape(g['name'])}</b> — government, usually free; confirm on the call"
                     + (f" — {_html.escape(g['phone'])}" if g["phone"] else "") + f" ({ui.by_car(g['minutes'])})</li>")
    elif g:
        steps.append(f"<li>Last fallback: no government hospital in our data within {g['none_within_km']} km — "
                     f"in an emergency call 108.</li>")
    st.markdown(f"<div class='cc-slip'><h4>📋 Call plan</h4><ol>{''.join(steps)}</ol></div>", unsafe_allow_html=True)

    hindi = st.toggle("Translate slip to Hindi", key=f"{ns}slip_hi")
    lang = "hi" if hindi else "en"
    slip = ui.slip_text(view, typed, where_label, lang, need_label=need_label)
    if hindi:
        st.caption("Template text translated; facility details unchanged.")
    with st.expander("Referral slip (no patient details)", expanded=False):
        st.code(slip, language=None, wrap_lines=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        copy_button(slip, "copy")
    b2.link_button("💬 Send slip on WhatsApp", ui.whatsapp_url(slip), width="stretch")
    with b3:
        family_popover(view, lang, ns, need_label)


def _clear_family(keys):
    """Runs when the family message is sent: wipe the number and sender so they are never kept."""
    for k in keys:
        st.session_state[k] = ""


def family_popover(view, lang, ns, need_label=None):
    """📲 Tell the family — WhatsApp message to the patient's family. The number is only used to build the wa.me link;
    it is never stored, logged or written anywhere, and the fields are cleared when the message is sent."""
    targets = ui.family_targets(view)
    if not targets:
        return
    k_num, k_from, k_tgt, k_conf = f"{ns}fam_num", f"{ns}fam_from", f"{ns}fam_tgt", f"{ns}fam_conf"
    with st.popover("📲 Tell the family", width="stretch"):
        num_raw = st.text_input("Family's WhatsApp number", key=k_num, placeholder="98765 43210")
        number, hint = ui.normalise_whatsapp(num_raw)
        if num_raw.strip() and not number:
            st.caption(f"⚠️ {hint}")
        i = st.selectbox("Which facility?", range(len(targets)), format_func=lambda j: targets[j]["label"], key=k_tgt)
        conf = st.radio("Confirmed by phone?", ["Not yet", "Yes"], horizontal=True, key=k_conf)
        sender = st.text_input("From (optional)", key=k_from, placeholder="e.g. ASHA worker, Sunita")
        need = ui.need_text(view["cap_ids"], None, lang) if view["cap_ids"] else (need_label or "")
        msg = ui.family_message(targets[i], need, conf == "Yes", sender, view["urgency"], view["cap_ids"], lang)
        st.code(msg, language=None, wrap_lines=True)
        if number:
            st.link_button("📲 Send on WhatsApp", ui.whatsapp_to(number, msg), width="stretch", type="primary",
                           on_click=_clear_family, args=([k_num, k_from],), key=f"{ns}fam_send")
        else:                                              # no valid number → a truly disabled button, no link
            st.button("📲 Send on WhatsApp", disabled=True, width="stretch", key=f"{ns}fam_send_off")
        copy_button(msg, "fam_copy", label="📋 Copy message")
        st.caption("This number is only used to open WhatsApp. CareCompass doesn't save it.")


def result_card(c, compact=False, key=""):
    """Visible: role, name + Call, travel, evidence label, 'What to ask'. Everything else in 'More details'."""
    with st.container(key=f"card_{key}_{c['level']}"):
        left, right = st.columns([3, 1.35], vertical_alignment="top")
        with left:
            num = f"{c['n']}. " if not compact else ""
            st.markdown(f"<div class='cc-role'>{_html.escape(c['role'])} · {_html.escape(c['reason'])}</div>"
                        f"<div class='cc-name'>{num}{_html.escape(c['name'])}</div>"
                        f"<div class='cc-travel'>🚗 {ui.by_car(c['minutes'])} ({c['km']:.0f} km)"
                        + (f" · <span class='cc-muted'>{_html.escape(c['address'])}</span>" if c["address"] else "")
                        + f"</div><div class='cc-level' style='color:{c['level_colour']}; margin-top:.35rem'>● "
                        f"{c['level_label']}</div>", unsafe_allow_html=True)
        with right:
            if c["tel"] and c["phone"]:
                with st.container(key=f"call_{key}"):
                    st.link_button(f"📞 Call {c['phone']}", c["tel"], width="stretch")
                if c.get("demo_phone"):
                    st.markdown("<span class='cc-demo'>Demo number</span> <span class='cc-muted'>not this facility's real "
                                "number</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='cc-muted'>No phone number listed</span>", unsafe_allow_html=True)
            if c["questions"]:
                with st.popover("❓ What to ask", width="stretch"):
                    st.markdown("\n".join(f"- {q}" for q in c["questions"]))
        with st.expander("More details: evidence, checks, location"):
            st.markdown("".join(f"<span class='cc-chip'>{'✅' if k == 'ok' else '⚠️' if k == 'warn' else 'ℹ️'} "
                                f"{_html.escape(t)}</span>" for k, t in c["chips"]), unsafe_allow_html=True)
            ev = c["evidence"]
            st.markdown(f"<div class='cc-quote'>“{ev['main_html']}”"
                        + (f" <span class='cc-muted'>({_html.escape(ev['main_field'])})</span>" if ev["main_field"] else "")
                        + "</div>", unsafe_allow_html=True)
            if ev["bullets"]:
                st.markdown("**All evidence**")
                st.markdown("\n".join(f"- {b} <span class='cc-muted'>— {_html.escape(f)}</span>" for f, b in ev["bullets"]),
                            unsafe_allow_html=True)
            for w in c["warnings"]:
                st.caption(f"⚠️ {w}")
            if c["location_note"]:
                st.caption(f"Location: {c['location_note']}")

def model1_card(m):
    """The extra 'From Model 1' card — testing only, not part of the call plan or slip."""
    if not m:
        return
    if "note" in m:
        st.caption(f"🧪 Model 1 (testing): {m['note']}")
        return
    with st.container(key=f"card_m1_{m['level']}"):
        left, right = st.columns([3, 1.35], vertical_alignment="top")
        with left:
            pct = (f" · directory confirm {m['pct']}%" if m["pct"] is not None else "")
            st.markdown(f"<div class='cc-role'>🧪 {_html.escape(m['role'])} · {_html.escape(m['reason'])}</div>"
                        f"<div class='cc-name'>{_html.escape(m['name'])}</div>"
                        f"<div class='cc-travel'>🚗 {ui.by_car(m['minutes'])} ({m['km']:.0f} km)"
                        + (f" · <span class='cc-muted'>{_html.escape(m['address'])}</span>" if m["address"] else "")
                        + f"</div><div class='cc-level' style='color:{m['level_colour']}; margin-top:.35rem'>● "
                        f"{m['level_label']}{pct}</div>", unsafe_allow_html=True)
        with right:
            if m["tel"] and m["phone"]:
                with st.container(key="call_m1"):
                    st.link_button(f"📞 Call {m['phone']}", m["tel"], width="stretch")
            if m["questions"]:
                with st.popover("❓ What to ask", width="stretch"):
                    st.markdown("\n".join(f"- {q}" for q in m["questions"]))
        with st.expander("More details: Model 1 evidence and reasons"):
            st.markdown(f"<div class='cc-quote'>“{_html.escape(m['quote'])}” "
                        f"<span class='cc-muted'>({_html.escape(m['quote_field'])})</span></div>", unsafe_allow_html=True)
            if m["reasons"]:
                st.markdown("\n".join(f"- {_html.escape(r)}" for r in m["reasons"]))
            if m["pct"] is not None:
                st.caption("Directory confirm % = chance the government hospital directory also lists this service, "
                           "given how it is described. Not a guarantee the service is offered.")


EMPTY = {
    "unknown_location": "We couldn't find that place. Try the district name or a 6-digit PIN, and add the state if the "
                        "name is common (e.g. 'Aurangabad, Bihar').",
    "unknown_service": "We couldn't tell which service you need. Pick one of the suggestions while typing, or describe "
                       "the problem in simple words (e.g. 'broken arm', 'pregnant').",
}


def render_results(out, typed_location, typed_need, on_change_place, need_label=None, ns=""):
    svc, loc, status = out["service_resolution"], out["location_resolution"] or {}, out["status"]
    if status == "needs_service_clarification":
        st.info(ui.plain(svc["note"]))
        cols = st.columns(len(svc["options"]) or 1)
        for col, o in zip(cols, svc["options"]):
            if col.button(f"**{o['display_name']}**  \n{o.get('description', '')}", key=f"{ns}svc_{o['service_id']}",
                          width="stretch"):
                start_search(o["display_name"], typed_location)
                st.rerun()
        return
    if status == "needs_location_clarification":
        st.info(ui.plain(loc["note"]) + " Pick the right one:")
        opts = loc.get("options") or []
        if len(opts) <= 4:
            for i, o in enumerate(opts):
                if st.button(o.get("display") or o["query"], key=f"{ns}loc_opt_{i}", width="stretch"):
                    on_change_place(o["query"])
                    st.rerun()
        else:
            choice = st.selectbox("Choose a district", range(len(opts)), format_func=lambda i: opts[i]["display"], key=f"{ns}pickdist")
            if st.button("Use this place", type="primary", key=f"{ns}useplace"):
                on_change_place(opts[choice]["query"])
                st.rerun()
        return
    if status in EMPTY:
        st.warning(f"{ui.plain((svc if status == 'unknown_service' else loc).get('note') or '')} {EMPTY[status]}")
        return

    fc = find_care_module()
    view = ui.build_view(out, fc)
    where_label = loc.get("label") or typed_location
    if view["urgency"] == "emergency":
        red("If the patient gets worse, call 108 now.")
    a1, a2 = st.columns([4, 1])
    a1.markdown(f"Showing results near **{_html.escape(where_label)}** "
                f"<span class='cc-muted'>(you typed “{_html.escape(typed_location)}”)</span> — not right?",
                unsafe_allow_html=True)
    if a2.button("Change place", key=f"{ns}change_loc"):
        st.session_state["edit_loc"] = True
    if st.session_state.get("edit_loc"):
        with st.form(f"{ns}change_loc_form"):
            new_loc = st.text_input("Where are you?", value=typed_location)
            st.caption("Village, town, district or PIN. Add the state if the name is common, e.g. 'Aurangabad, Bihar'.")
            if st.form_submit_button("Search again", type="primary") and new_loc.strip():
                st.session_state["edit_loc"] = False
                on_change_place(new_loc)
                st.rerun()

    if view["plan"] and not view["cards"]:
        st.warning("No facility nearby clearly lists this — the options below are not confirmed. Ask before going.")
    if not view["plan"]:
        st.warning(f"Nothing within {view['radius']:.0f} km lists this. Try fewer services (a facility must offer all "
                   f"of them), a nearby bigger town, or call 108 in an emergency.")
    call_plan(view, typed_need, where_label, need_label=need_label, ns=ns)
    numbered_map(view, loc, where_label)
    for c in view["cards"]:
        result_card(c, key=f"{ns}{c['n']}")
    if view["fallback_cards"]:
        st.markdown("##### Nothing better nearby — not confirmed" if view["cards"] else "##### Possible options — not confirmed")
        for i, c in enumerate(view["fallback_cards"]):
            result_card(c, compact=True, key=f"{ns}fb{i}")
    if view["why_not"]:
        with st.expander("Why not these?"):
            for w in view["why_not"]:
                st.markdown(f"- **{_html.escape(w['name'])}** (~{w['minutes']} min by car, estimated) — {_html.escape(w['reason'])}"
                            + ("".join(f"; {_html.escape(m.lower())}" for m in w["more"])))
    how = out.get("how_we_decided") or {}
    with st.expander("How we decided"):
        for k in ("service", "location", "matching"):
            if how.get(k):
                st.markdown(f"- {ui.plain(how[k])}")
        urg = view["urgency"]
        st.markdown("- " + {"emergency": "Treated as an emergency: closeness counts most.",
                            "recurring": "Treated as regular, repeated care: closeness and evidence both count.",
                            "planned": "Treated as planned care: how clearly a facility offers the service counts most."}[urg])
        for w in how.get("widening") or []:
            st.markdown(f"- {ui.plain(w)}")
        st.markdown(f"- Searched within {view['radius']:.0f} km. Travel times are straight-line estimates "
                    f"(road detour × 1.4 at 40 km/h).")
        for n in out.get("search_notes") or []:
            if n:
                st.markdown(f"- {ui.plain(n)}")
        if view["cards"]:
            st.markdown("**Ranking factors (how we ordered results, not certainty)**")
            st.dataframe(pd.DataFrame([{"#": c["n"], "Facility": c["name"],
                                        "Offers the service": c["components"].get("evidence"),
                                        "Close to you": c["components"].get("access"),
                                        "Staff & capacity": c["components"].get("readiness"),
                                        "Verifiable details": c["components"].get("trust")} for c in view["cards"]]),
                         hide_index=True, width="stretch",
                         column_config={k: st.column_config.ProgressColumn(k, min_value=0, max_value=1, format="%.2f")
                                        for k in ["Offers the service", "Close to you", "Staff & capacity", "Verifiable details"]})


def searching(steps_fn):
    """st.status with the five loading steps; errors are shown, not swallowed."""
    with st.status("Understanding your request…", expanded=False) as stt:
        try:
            out = steps_fn(stt)
        except Exception as e:
            stt.update(label="Something went wrong", state="error")
            st.error(f"The search failed: {type(e).__name__}: {e}. Try again; if it keeps failing, use the classic "
                     f"search below.")
            return None
        stt.update(label="Done", state="complete")
    return out


with find_tab:
    s = sel()
    with st.expander("How to use"):
        st.markdown("1. **Describe the need** — a problem (*broken arm*, *chest pain*) or a service (*X-ray*, *dialysis*).\n"
                    "2. **Pick services** from the suggestions. A facility must offer every service you pick.\n"
                    "3. **Say where the patient is**, then **call in order** and **send the slip** to the family.")
    qp = st.query_params
    if qp.get("need") and not st.session_state.get("qp_done"):        # shareable link: ?need=xray,ct_scan&where=Patna
        st.session_state["qp_done"] = True
        ids = [i for i in qp.get("need").split(",") if i in caps.catalog()["by_id"]]
        add_caps(ids, source=qp.get("from"))
        if qp.get("where"):
            st.session_state["q"] = {"mode": "caps", "ids": ids, "from": qp.get("from"), "urgency": None,
                                     "location": qp.get("where")}

    st.markdown("<div class='cc-ask'>What does the patient need?</div>", unsafe_allow_html=True)
    if HAVE_SEARCHBOX and not st.session_state.get("plain_box"):
        big = {"fontSize": "1.3rem"}
        chosen = st_searchbox(searchbox_options, placeholder="Start typing… e.g. broken arm, dialysis", key="need_box",
                              clear_on_submit=True,
                              style_overrides={"searchbox": {"control": {"minHeight": "3.3rem"}, "input": big,
                                                             "placeholder": big, "singleValue": big,
                                                             "option": {"fontSize": "1.1rem"}}})
        if chosen and chosen != st.session_state.get("last_chosen"):
            st.session_state["last_chosen"] = chosen
            box = st.session_state.get("need_box") or {}
            pick(chosen, typed=box.get("search") if isinstance(box, dict) else None)
            st.rerun()
    else:
        typed = st.text_input("What does the patient need?", key="need_plain", label_visibility="collapsed",
                              placeholder="Type and press Enter")
        if typed.strip():
            opts = searchbox_options(typed)
            if opts:
                st.caption("Tap to add:")
                for i, (lab, value) in enumerate(opts[:8]):
                    if st.button(lab, key=f"sug_{i}_{value}"):
                        pick(value, typed=typed)
                        st.rerun()
    st.markdown("<div class='cc-help'>A condition like 'broken arm' or a service like 'dialysis'.</div>",
                unsafe_allow_html=True)
    if HAVE_SEARCHBOX and not st.session_state.get("plain_box"):
        if st.button("Search box not working? Use a simple text box", key="use_plain", type="tertiary"):
            st.session_state["plain_box"] = True
            st.rerun()

    if st.session_state.get("llm_error"):
        st.warning(st.session_state.pop("llm_error"))
    if s["banner"] and s["urgency"] == "emergency":
        red(s["banner"])

    if s["ids"]:
        names = caps.catalog()["by_id"]
        keep = st.pills("Selected — a facility must offer all of these (tap to remove)", options=s["ids"],
                        default=s["ids"], selection_mode="multi", format_func=lambda i: names[i]["name"],
                        key=f"chips_{'-'.join(s['ids'])}")
        if set(keep) != set(s["ids"]):
            s["ids"] = [i for i in s["ids"] if i in keep]
            st.rerun()
        if s["related"]:
            more = st.pills("Also consider adding", options=s["related"], selection_mode="single",
                            format_func=lambda i: "+ " + caps.label(names[i]), key=f"rel_{'-'.join(s['related'])}")
            if more:
                add_caps([more])
                s["related"] = [r for r in s["related"] if r != more]
                st.rerun()

    st.markdown("<div class='cc-ask'>Where are you?</div>", unsafe_allow_html=True)
    with st.form("where_form", border=False):
        with st.container(key="where_input"):
            where = st.text_input("Where are you?", placeholder="Village, town, district or PIN",
                                  label_visibility="collapsed")
        st.markdown("<div class='cc-help'>Village, town, district or PIN. Add the state if the name is common, "
                    "e.g. 'Aurangabad, Bihar'.</div>", unsafe_allow_html=True)
        with st.container(key="findbtn"):
            go = st.form_submit_button("Find care", type="primary")
        if go:
            if not s["ids"]:
                st.warning("Add at least one service above first — start typing the problem or the service.")
            elif not where.strip():
                st.warning("Please say where you are — a village, town, district or PIN.")
            else:
                st.session_state["q"] = {"mode": "caps", "ids": list(s["ids"]), "from": s["from"],
                                         "urgency": s["urgency"], "location": where.strip()}
                st.session_state["edit_loc"] = False

    q = st.session_state.get("q")
    if q and q.get("mode") == "caps":
        names = " + ".join(caps.catalog()["by_id"][i]["name"] for i in q["ids"])
        a1, a2 = st.columns([4, 1])
        a1.markdown(f"**Searching for:** {_html.escape(names)}" +
                    (f" (from “{_html.escape(q['from'])}”)" if q.get("from") else ""))
        if a2.button("Change", key="change_need"):
            st.session_state["sel"] = {"ids": [], "from": None, "related": [], "urgency": None, "banner": None}
            st.session_state.pop("q", None)
            st.rerun()

        def _steps(stt, q=q):
            stt.update(label="Finding the place…")
            find_care_module()
            stt.update(label="Finding facilities…")
            out = run_find_care(tuple(q["ids"]), q["location"], q.get("from"), q.get("urgency"))
            stt.update(label="Checking evidence…")
            stt.update(label="Building call plan…")
            return out
        out = searching(_steps)

        def _relocate(new_loc, q=q):
            st.session_state["q"] = dict(q, location=new_loc)
        if out is not None:
            render_results(out, q["location"], q.get("from"), _relocate)

    with st.expander("Can't find it? Describe the need in your own words (classic search)"):
        with st.form("search"):
            need = st.text_input("What does the patient need?", placeholder="e.g. dialysis, c-section, heart surgery")
            where2 = st.text_input("Where is the patient?", placeholder="Village, town, district or PIN")
            if st.form_submit_button("Search", type="primary"):
                if not need.strip() or not where2.strip():
                    st.warning("Please fill in both what the patient needs and where they are.")
                else:
                    start_search(need, where2)
    if q and q.get("mode") == "classic":
        def _steps_c(stt, q=q):
            stt.update(label="Finding the place and facilities…")
            find_care_module()
            out = run_recommend(q["service"], q["location"])
            stt.update(label="Building call plan…")
            return out
        out = searching(_steps_c)

        def _relocate_classic(new_loc, q=q):
            start_search(q["service"], new_loc)
        if out is not None:
            render_results(out, q["location"], q["service"], _relocate_classic,
                           need_label=out["service_resolution"].get("display_name") or q["service"])



# ───────────────────────────── Model 1 / Model 2 / Combined (testing) ─────────────────────────────
# Same search as Find care, re-ranked by each team model (versions.py). Find care itself is unchanged.
import versions


@st.cache_data(ttl=900, show_spinner=False)
def run_version(version: str, cap_ids: tuple, location_text: str, condition: str | None, urgency: str | None) -> dict:
    base = run_find_care(cap_ids, location_text, condition, urgency)
    return versions.rank(version, base, find_care_module())


def model_tab(tab, version):
    with tab:
        v = versions.VERSION[version]
        st.markdown(f"<div class='cc-ask' style='font-size:1.5rem'>{v['title']} <span class='cc-muted'>(testing)</span></div>"
                    f"<div class='cc-help'>{_html.escape(v['rule'])}</div>", unsafe_allow_html=True)
        if not versions.available()[version]:
            st.info(f"{v['title']} results are not loaded yet.")
            return
        q = st.session_state.get("q")
        if not q or q.get("mode") != "caps":
            st.info("Search in the **Find care** tab first — this tab re-ranks the same search with "
                    f"{v['title']}.")
            return
        names = " + ".join(caps.catalog()["by_id"][i]["name"] for i in q["ids"])
        st.markdown(f"**Same search as Find care:** {_html.escape(names)} near {_html.escape(q['location'])}")
        try:
            out = run_version(version, tuple(q["ids"]), q["location"], q.get("from"), q.get("urgency"))
        except Exception as e:
            st.error(f"{v['title']} could not rank this search: {type(e).__name__}: {e}")
            return
        if out["status"] in ("needs_location_clarification", "needs_service_clarification", "unknown_location",
                             "unknown_service"):
            st.info("Finish the search in the **Find care** tab first (pick the place or service there).")
            return

        def _relocate(new_loc, q=q):
            st.session_state["q"] = dict(q, location=new_loc)
        render_results(out, q["location"], q.get("from"), _relocate, ns=f"{version}_")


if SHOW_MODEL_TABS:
    model_tab(m1_tab, "model1")
    model_tab(m2_tab, "model2")
    model_tab(combo_tab, "combined")

# ───────────────────────────── Care deserts ─────────────────────────────
NEED_FOR = {  # service -> (NFHS indicator, True if a higher value means more need)
    "maternity": ("institutional_births_pct", False), "pediatrics": ("children_fully_vaccinated_pct", False),
    "cardiology": ("women_high_bp_pct", True), "diabetes": ("women_high_blood_sugar_pct", True),
    "dialysis_kidney": ("men_high_blood_sugar_pct", True), "cancer": ("women_cervical_screened_pct", False),
    "emergency": ("health_insurance_pct", False), "mental_health": ("health_insurance_pct", False),
    "eye": ("health_insurance_pct", False), "dental": ("health_insurance_pct", False),
}
NEED_LABEL = {
    "institutional_births_pct": "births in a facility (%)", "children_fully_vaccinated_pct": "children fully vaccinated (%)",
    "women_high_bp_pct": "women with high blood pressure (%)", "women_high_blood_sugar_pct": "women with high blood sugar (%)",
    "men_high_blood_sugar_pct": "men with high blood sugar (%)", "women_cervical_screened_pct": "women ever screened for cervical cancer (%)",
    "health_insurance_pct": "households with health insurance (%)",
}

if SHOW_EXTRA_TABS:
    with desert_tab:
        d = data.districts()
        c1, c2 = st.columns([1, 1])
        svc_d = c1.selectbox("Service", list(SERVICE_LABELS), format_func=SERVICE_LABELS.get, key="desert_svc")
        states = ["All India"] + sorted(d["state"].dropna().unique())
        st_pick = c2.selectbox("State", states)
        ind, higher_worse = NEED_FOR[svc_d]
        supply = f"n_{svc_d}_confident"
        dd = d if st_pick == "All India" else d[d["state"] == st_pick]
        dd = dd.copy()

        zero = (dd[supply] == 0).sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Districts", f"{len(dd):,}")
        m2.metric(f"With no {SERVICE_LABELS[svc_d].lower()} facility", f"{zero:,}")
        m3.metric("With no facility of any kind", f"{int(dd['has_no_facilities'].sum()):,}")

        dd["need"] = dd[ind]
        dd["color"] = dd[supply].map(lambda n: [200, 40, 40] if n == 0 else [230, 150, 30] if n <= 2 else [30, 140, 80])
        q = dd["need"].rank(pct=True, ascending=higher_worse).fillna(0.5)
        dd["radius"] = 6000 + 22000 * q
        dd["tip"] = dd.apply(lambda r: f"{r['district']}, {r['state']}\n{SERVICE_LABELS[svc_d]}: {r[supply]} facilities\n"
                                       f"{NEED_LABEL[ind]}: {r['need'] if pd.notna(r['need']) else 'n/a'}", axis=1)
        mp = dd.dropna(subset=["lat", "lon"])
        st.pydeck_chart(pdk.Deck(
            layers=[pdk.Layer("ScatterplotLayer", mp, get_position="[lon, lat]", get_fill_color="color", get_radius="radius",
                              opacity=0.7, pickable=True)],
            initial_view_state=india_view() if st_pick == "All India" else india_view(mp["lat"].mean(), mp["lon"].mean(), 5.5),
            tooltip={"text": "{tip}"}))
        st.caption(f"Red = no {SERVICE_LABELS[svc_d].lower()} facility, orange = 1–2, green = 3+ "
                   f"(counting facilities with a confirmed or approximate location). "
                   f"Bigger circle = more need: {'higher' if higher_worse else 'lower'} {NEED_LABEL[ind]} (NFHS-5, 2019–21).")

        st.subheader("Highest need, least care")
        table = (dd.sort_values([supply, "need"], ascending=[True, not higher_worse])
                   [["state", "district", supply, f"n_{svc_d}", "n_facilities", ind]]
                   .rename(columns={supply: f"{SERVICE_LABELS[svc_d]} (confirmed location)",
                                    f"n_{svc_d}": f"{SERVICE_LABELS[svc_d]} (all)", "n_facilities": "All facilities",
                                    ind: NEED_LABEL[ind]}))
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Facility counts come from the CareCompass facility dataset, which does not list every facility in India: "
                   "a zero means none in our data, not necessarily none on the ground.")

    # ───────────────────────────── About ─────────────────────────────
if SHOW_EXTRA_TABS:
    with about_tab:
        st.markdown("""
**Facilities:** 9,969 cleaned facility records with standardized specialties, from the Virtue Foundation dataset.
Every location has a confidence level. Coordinates, PIN and city were cross-checked against India Post; 800 wrong
coordinates were moved to the facility's PIN area.

**Places:** India Post directory (19,586 PINs, 163,007 towns and villages) for turning what you type into a location.

**Health need:** NFHS-5 district indicators (2019–21), 705 districts.

**Travel times:** v1 straight-line estimates from each PIN to every facility within 150 km.

Full cleaning write-up: `CLEANING_README.md` in the team's Datathon 2026 folder.
""")
