"""Capability catalog: normalisation, type-ahead suggestions, and per-facility evidence levels.

Pure Python — no Spark, no LLM — so suggestions are instant on every keystroke and testable offline.
Reads ONLY catalog/capabilities.json and catalog/conditions.json; swapping in the full catalog later
needs no code changes.

capabilities.json rows:  capability_id, name, variants[], type (equipment|procedure|department|specialty),
                          domain, facility_count
conditions.json rows:    condition_id, label, phrases[], domains[], capability_ids[], urgency, banner (bool)
"""
import difflib
import json
import re
from functools import lru_cache
from pathlib import Path

CATALOG_DIR = Path(__file__).resolve().parent / "catalog"

# ───────────────────────────── normalisation ─────────────────────────────
_SPELLING = [(r"colour", "color"), (r"haem", "hem"), (r"orthopaed", "orthoped"), (r"paed", "ped"),
             (r"caesarian|caesarean|cesarian", "cesarean"), (r"caesar", "cesar"), (r"labour", "labor"),
             (r"tumour", "tumor"), (r"anaes", "anes"), (r"centre", "center")]


def norm(s) -> str:
    """lowercase; 'X-Ray' = 'x ray' = 'xray'; '24x7' = '24/7' = '24 7'; British/US spellings unified."""
    s = str(s or "").lower()
    s = re.sub(r"(\d)\s*x\s*(\d)", r"\1 \2", s)            # 24x7 → 24 7
    s = re.sub(r"[^a-z0-9]+", " ", s)
    for a, b in _SPELLING:
        s = re.sub(a, b, s)
    toks = s.split()
    out, i = [], 0
    while i < len(toks):                                     # 'x ray' → 'xray', 'c section' → 'csection'
        t = toks[i]
        prev_single = bool(out) and len(out[-1]) == 1
        if (len(t) == 1 and t.isalpha() and not prev_single and i + 1 < len(toks)
                and len(toks[i + 1]) >= 3 and toks[i + 1].isalpha()):
            out.append(t + toks[i + 1])
            i += 2
            continue
        out.append(t)
        i += 1
    return " ".join(out)


def squash(s) -> str:
    return norm(s).replace(" ", "")


# ───────────────────────────── catalog ─────────────────────────────
@lru_cache(maxsize=1)
def catalog():
    caps = json.loads((CATALOG_DIR / "capabilities.json").read_text())
    conds = json.loads((CATALOG_DIR / "conditions.json").read_text())
    by_id = {c["capability_id"]: c for c in caps}
    for c in caps:
        c["_variants"] = sorted({norm(v) for v in [c["name"], *c.get("variants", [])] if norm(v)}, key=len, reverse=True)
    for k in conds:
        k["_phrases"] = sorted({norm(p) for p in k["phrases"]}, key=len, reverse=True)
        bad = [i for i in k["capability_ids"] if i not in by_id]
        assert not bad, f"condition {k['condition_id']} refers to unknown capabilities {bad}"
    # one regex for every variant of every capability, longest first: at any position the longest
    # variant wins, so 'dental x-ray' is Dental X-ray and never also X-ray
    variant_owner = {}
    for c in caps:
        for v in c["_variants"]:
            variant_owner.setdefault(v, c["capability_id"])
    alts = sorted(variant_owner, key=len, reverse=True)
    rx = re.compile(r"\b(" + "|".join(re.escape(v) for v in alts) + r")s?\b")
    return {"caps": caps, "by_id": by_id, "conditions": conds, "variant_owner": variant_owner, "rx": rx}


def label(cap) -> str:
    n = cap.get("facility_count")
    return f"{cap['name']} · {n:,} facilities" if isinstance(n, int) else cap["name"]


# ───────────────────────────── suggestions ─────────────────────────────
def _condition_hits(q):
    hits = []
    for k in catalog()["conditions"]:
        best = 0.0
        for p in k["_phrases"]:
            if re.search(r"\b" + re.escape(p) + r"\b", q):
                best = max(best, 1.0 + len(p) / 100)                  # whole phrase typed
            elif len(q) >= 3 and p.startswith(q):
                best = max(best, 0.9)                                  # still typing it
            elif len(q) >= 4 and difflib.SequenceMatcher(None, q, p).ratio() >= 0.85:
                best = max(best, 0.8)                                  # typo of the phrase
        if best:
            hits.append((best, k))
    return [k for _, k in sorted(hits, key=lambda x: -x[0])]


def _capability_hits(q):
    qs = q.replace(" ", "")
    scored = []
    for c in catalog()["caps"]:
        best = 0.0
        for v in c["_variants"]:
            vs = v.replace(" ", "")
            if v == q or vs == qs:
                best = max(best, 1.0)
            elif vs.startswith(qs) or any(w.startswith(q) for w in v.split()):
                best = max(best, 0.9)
            elif len(qs) >= 3 and qs in vs:
                best = max(best, 0.75)
            elif len(qs) >= 4 and difflib.SequenceMatcher(None, qs, vs).ratio() >= 0.82:
                best = max(best, 0.7)
        if best:
            scored.append((best, c.get("facility_count") or 0, c))
    return [c for _, _, c in sorted(scored, key=lambda x: (-x[0], -x[1]))]


def suggest(text, limit=8):
    """Suggestions for what the user has typed so far (from 3 characters).

    Returns {"conditions": [...], "suggestions": [...], "banner": str|None, "urgency": str|None}.
    Condition matches come first; when a condition matched, capability suggestions are limited to its
    domains (so 'broken arm' can never suggest Dental X-ray)."""
    q = norm(text)
    out = {"query": q, "conditions": [], "suggestions": [], "banner": None, "urgency": None}
    if len(q.replace(" ", "")) < 3:
        return out
    cat = catalog()
    conds = _condition_hits(q)
    seen, sugg = set(), []

    def add(cap, source):
        if cap["capability_id"] not in seen:
            seen.add(cap["capability_id"])
            sugg.append({"capability_id": cap["capability_id"], "name": cap["name"], "domain": cap["domain"],
                         "facility_count": cap.get("facility_count"), "label": label(cap), "source": source})

    allowed = None
    if conds:
        allowed = {d for k in conds for d in k["domains"]}
        for k in conds:
            for cid in k["capability_ids"]:
                add(cat["by_id"][cid], f"from '{k['label']}'")
    for cap in _capability_hits(q):
        if allowed is None or cap["domain"] in allowed:
            add(cap, "name")
    out["conditions"] = [{"condition_id": k["condition_id"], "label": k["label"], "capability_ids": k["capability_ids"],
                          "urgency": k["urgency"], "banner": bool(k.get("banner")),
                          "banner_text": (k.get("banner_text") or DEFAULT_BANNER) if k.get("banner") else None}
                         for k in conds[:3]]
    out["suggestions"] = sugg[:limit]
    flagged = [k for k in conds if k.get("banner")]
    if flagged:
        out["banner"] = flagged[0].get("banner_text") or DEFAULT_BANNER
    if conds:
        out["urgency"] = "emergency" if any(k["urgency"] == "emergency" for k in conds) else conds[0]["urgency"]
    return out


DEFAULT_BANNER = "This may be an emergency. Call 108 for a free ambulance now — don't wait for this search."


def llm_prompt(text):
    """Prompt for the Enter-with-no-local-match fallback: the model may only pick ids from the list."""
    lines = "\n".join(f"{c['capability_id']}: {c['name']} ({c['domain']})" for c in catalog()["caps"])
    return ("You map a patient's healthcare need in India to hospital capabilities. Choose 3 to 6 capability_ids "
            "from THIS LIST ONLY, most relevant first. If nothing fits, return an empty list.\n\n"
            f"LIST:\n{lines}\n\nNeed: {text}\n\nReturn JSON: {{\"capability_ids\": [\"...\"]}}")


def validate_llm_ids(ids):
    by_id = catalog()["by_id"]
    out = []
    for i in ids or []:
        i = str(i).strip()
        if i in by_id and i not in out:
            out.append(i)
    return out[:6]


# ───────────────────────────── evidence levels ─────────────────────────────
LEVELS = ["NONE", "PASSING", "LISTED", "SPECIFIC"]          # ordered weakest → strongest
LEVEL_TEXT = {"SPECIFIC": "specific evidence", "LISTED": "listed", "PASSING": "mentioned in passing",
              "NONE": "not found in its record"}
STRUCTURED = ("equipment", "procedures", "capabilities")
_DETAIL = re.compile(r"\b(\d+(\.\d+)?|siemens|philips|ge|toshiba|canon|fujifilm|carestream|wipro|hitachi|samsung|"
                     r"mindray|fresenius|nipro|b braun|tesla|slice|bedded|beds|machines?)\b")

# specialty label → domain, used to find a facility's MAIN domain (keyword based, generic labels ignored)
_SPECIALTY_DOMAIN = [
    (r"dent|orthodont|endodont|periodont|prosthodont|oral|maxillofacial|implant", "dental"),
    (r"ophthalm|retina|cataract|cornea|glaucoma|eye|optometr|oculo", "eye"),
    (r"orthoped|sports medicine|spine|joint|hand surgery|foot and ankle", "orthopaedic"),
    (r"obstet|gyn|matern|fertility|reproductive|perinat", "maternity"),
    (r"neonat|newborn", "newborn"),
    (r"cardi|heart|vascular|phlebolog|vein", "cardiac"),
    (r"nephro|renal|kidney|urolog", "kidney"),
    (r"oncolog|cancer|radiation|hematology oncology", "cancer"),
    (r"emergency|trauma", "emergency"),
    (r"critical care|intensive", "critical_care"),
    (r"neuro", "neuro"),
    (r"otolaryng|ent\b|ear|audiolog|rhinolog|laryngolog", "ent"),
    (r"dermat|cosmet|aesthet|hair|skin|trichol|plastic", "skin"),
    (r"psychiat|psycholog|mental|addiction", "mental_health"),
    (r"patholog|laborator|microbiolog|biochem|hematolog", "lab"),
    (r"radiolog|imaging|ultraso|sonolog", "general_imaging"),
]
_GENERIC = re.compile(r"internal medicine|family medicine|general medicine|general practice|general surgery|"
                      r"pediatric$|^pediatrics|anesthes|physical medicine|rehabilitation|emergency medicine|"
                      r"critical care|radiology|pathology|laborator|nursing|public health|geriatric|infectious")


def specialty_domains(labels):
    """→ (main_domain, set_of_domains). main_domain is 'general' unless one non-generic domain dominates."""
    counts = {}
    for lab in labels or []:
        n = norm(lab)
        if not n or _GENERIC.search(n):
            continue
        for rx, d in _SPECIALTY_DOMAIN:
            if re.search(rx, n):
                counts[d] = counts.get(d, 0) + 1
                break
    if not counts:
        return "general", set()
    top, n_top = max(counts.items(), key=lambda kv: kv[1])
    total = sum(counts.values())
    main = top if (n_top / total >= 0.6 and len(counts) <= 2) else "general"
    return main, set(counts)


def find_mentions(fields):
    """fields: {field_name: [item text, ...]} → {capability_id: [(field, item, variant), ...]}"""
    cat = catalog()
    found = {}
    for field, items in fields.items():
        for item in items or []:
            n = norm(item)
            if not n:
                continue
            for m in cat["rx"].finditer(n):
                cid = cat["variant_owner"][m.group(1)]
                found.setdefault(cid, []).append((field, item, m.group(1)))
    return found


def _is_generic_list(item):
    return len(re.split(r",|;|\band\b|/", str(item))) >= 6


def level_for(cap, mentions, facility_fields, main_domain, domains):
    """Evidence level of one facility for one capability, with the reasons."""
    if not mentions:
        return {"level": "NONE", "quote": None, "field": None, "penalties": []}
    structured = [m for m in mentions if m[0] in STRUCTURED and not _is_generic_list(m[1])]
    fields = {m[0] for m in mentions}
    variants = {m[2] for m in mentions}
    detail = any(_DETAIL.search(norm(m[1])) for m in structured)
    if structured:
        level = "SPECIFIC" if (len(fields) >= 2 or len(variants) >= 2 or detail) else "LISTED"
        quote = structured[0]
    else:
        level, quote = "PASSING", mentions[0]
    penalties = []
    if cap["type"] == "equipment" and facility_fields.get("equipment") and "equipment" not in fields:
        penalties.append("has an equipment list, but this is not in it")
    if main_domain not in ("general", cap["domain"]) and cap["domain"] not in domains:
        penalties.append(f"mainly a {main_domain.replace('_', ' ')} facility")
    i = LEVELS.index(level)
    i = max(1, i - len(penalties))                     # each penalty lowers one level, never below PASSING
    return {"level": LEVELS[i], "quote": quote[1], "field": quote[0], "penalties": penalties,
            "domain_mismatch": any(p.startswith("mainly") for p in penalties)}
