"""locate(text): a PIN, a place name, or "lat, lon" -> where to search from.

Same rules as the 03_geo notebook: coordinates → 6-digit PIN → exact district → exact town covering 2+ PINs
→ close district spelling (≥ 0.85) → exact one-PIN village → fuzzy town (≥ 0.8).
"""
import difflib
import re

import numpy as np
import pandas as pd

LAT_MIN, LAT_MAX, LON_MIN, LON_MAX = 6.0, 37.6, 68.0, 97.5
_COORD = re.compile(r"\s*\(?\s*(-?\d{1,2}(?:\.\d+)?)\s*[,\s]\s*(-?\d{1,3}(?:\.\d+)?)\s*\)?\s*")


# Old / common names → the name the India Post directory uses (same list as 08_Silver_Facilities, plus a few)
ALIASES = {
    "gurgaon": "gurugram", "bangalore": "bengaluru", "bombay": "mumbai", "calcutta": "kolkata",
    "madras": "chennai", "poona": "pune", "mysore": "mysuru", "trivandrum": "thiruvananthapuram",
    "cochin": "kochi", "baroda": "vadodara", "allahabad": "prayagraj", "benares": "varanasi",
    "banaras": "varanasi", "pondicherry": "puducherry", "vizag": "visakhapatnam", "calicut": "kozhikode",
    "mangalore": "mangaluru", "belgaum": "belagavi", "hubli": "hubballi", "gulbarga": "kalaburagi",
    "trichy": "tiruchirappalli", "shimoga": "shivamogga", "tumkur": "tumakuru", "bellary": "ballari",
    "faizabad": "ayodhya", "new delhi": "new delhi", "delhi": "new delhi",
    # big cities whose district has a different name
    "mangalore": "dakshina kannada", "mangaluru": "dakshina kannada", "noida": "gautam buddha nagar",
    "greater noida": "gautam buddha nagar", "navi mumbai": "thane", "secunderabad": "hyderabad",
    "kochi": "ernakulam", "vijayawada": "krishna", "vizag": "visakhapatanam", "visakhapatnam": "visakhapatanam",
}


def norm_place(s: str) -> str:
    s = str(s).lower().replace("&", " and ")
    return re.sub(r"\s+", " ", s).strip(" .,-")


_RANK = {"exact district": 0, "exact town": 1, "district name": 2, "closest district spelling": 3, "closest town spelling": 4}


class Locator:
    def __init__(self, places: pd.DataFrame, pins: pd.DataFrame):
        p = places.sort_values("n_pins", ascending=False)
        self.districts = p[p["type"] == "district"].drop_duplicates("place").set_index("place")
        self.towns = p[p["type"] == "town"].drop_duplicates("place").set_index("place")
        self._district_names = list(self.districts.index)
        self._town_names = list(self.towns.index)
        self.pins = pins.set_index("pincode")
        self._pin_ids = pins["pincode"].to_numpy()
        self._pin_rad = np.radians(pins[["lat", "lon"]].to_numpy())

    def nearest_pin(self, lat: float, lon: float) -> str:
        la, lo = np.radians(lat), np.radians(lon)
        d = (np.sin((self._pin_rad[:, 0] - la) / 2) ** 2
             + np.cos(la) * np.cos(self._pin_rad[:, 0]) * np.sin((self._pin_rad[:, 1] - lo) / 2) ** 2)
        return self._pin_ids[int(np.argmin(d))]

    @staticmethod
    def _fuzzy(q, index, names, cutoff):
        cands = difflib.get_close_matches(q, names, n=10, cutoff=cutoff)
        if not cands:
            return None, 0.0
        best = max(cands, key=lambda c: (difflib.SequenceMatcher(None, q, c).ratio(), index.loc[c, "n_pins"]))
        return best, difflib.SequenceMatcher(None, q, best).ratio()

    def _place(self, q):
        if q in self.districts.index:
            return self.districts.loc[q], "exact district"
        if q in self.towns.index and self.towns.loc[q, "n_pins"] >= 2:
            return self.towns.loc[q], "exact town"
        starts = [n for n in self._district_names if n.startswith(q + " ")]   # "bengaluru" → "bengaluru urban"
        if starts:
            return self.districts.loc[starts[0]], "district name"
        name, _ = self._fuzzy(q, self.districts, self._district_names, 0.85)
        if name:
            return self.districts.loc[name], "closest district spelling"
        if q in self.towns.index:
            return self.towns.loc[q], "exact town"
        name, _ = self._fuzzy(q, self.towns, self._town_names, 0.8)
        if name:
            return self.towns.loc[name], "closest town spelling"
        return None, None

    def locate(self, text: str):
        """Returns dict(lat, lon, pin, district, state, label, how) or None."""
        t = str(text).strip()
        if not t:
            return None
        m = _COORD.fullmatch(t)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
                return None
            pin = self.nearest_pin(lat, lon)
            r = self.pins.loc[pin]
            return dict(lat=lat, lon=lon, pin=pin, district=r["district"], state=r["state"],
                        label=f"{lat:.4f}, {lon:.4f}", how="coordinates")
        digits = t.replace(" ", "")
        if re.fullmatch(r"\d{6}", digits):
            if digits not in self.pins.index:
                return None
            r = self.pins.loc[digits]
            return dict(lat=float(r["lat"]), lon=float(r["lon"]), pin=digits, district=r["district"], state=r["state"],
                        label=f"PIN {digits}", how="PIN")
        q = norm_place(t)
        # try the name as typed and its current/old alias; keep the stronger match (then the bigger place)
        found = [x for x in (self._place(n) for n in dict.fromkeys([q, ALIASES.get(q, q)])) if x[0] is not None]
        if not found:
            return None
        r, how = min(found, key=lambda x: (_RANK[x[1]], -x[0]["n_pins"]))
        return dict(lat=float(r["lat"]), lon=float(r["lon"]), pin=r["nearest_pin"], district=r["district"],
                    state=r["state"], label=f"{r.name.title()} ({r['type']})", how=how)
