# scripts/04-stage3c-harmonize.py  (stage 3c, step 4)
#
# Harmonise the units in the merged stage3c tables:
#
#     merged-mitigation.csv
#     merged-costs.csv
#     merged-emissions.csv
#
# TARGET UNITS
#
#     power      -> MW
#     energy     -> MWh
#     CO2 mass   -> MtCO2   (CO2eq kept separate as MtCO2eq)
#     area       -> km2
#     money      -> real 2025 USD  (denominator kept, e.g. USD2025/MWh)
#
# MONEY PIPELINE  (convert-then-deflate, "pipeline B")
#
#     nominal LCU(Y)
#       -> x  FX(Y, LCU per USD)          [World Bank PA.NUS.FCRF, annual avg]
#       -> nominal USD(Y)
#       -> x  US_deflator(2025) / US_deflator(Y)   [World Bank NY.GDP.DEFL.ZS, USA]
#       -> real USD 2025
#
#     base year Y:  explicit *_Currency_Base_Year
#                   -> year embedded in the unit string ("1975$/GJ")
#                   -> Publication_Year          (flagged)
#
# REFERENCE DATA
#
#     Fetched live from the World Bank Indicators API on first run and cached
#     under inputs/reference/.  Re-fetch with --refresh.
#
# GENERIC (non-regional) ENERGY ASSUMPTIONS
#
#     1 toe  = 11.63  MWh
#     1 tce  =  8.141 MWh
#     1 bcm  = 10.6   TWh   (natural gas, gross)
#     carbon -> CO2 : x 44/12
#
# CONTEXT FILL  (fields completed from adjacent rows)
#
#     The extractor often records a field on only some rows of a paper -- most
#     commonly Area (a single-country study that labels only a few of its
#     rows, e.g. the Chile paper p436).  Before harmonising, a blank field is
#     filled from the paper's other rows when they carry exactly one distinct
#     non-blank value for it (i.e. the paper is unambiguous).  Filled fields:
#     Title, Publication_Year, DOI, Area, and the *Currency_Base_Year fields.
#     Every fill is listed per row in the Context_Filled column.
#
# SECTOR HARMONISATION
#
#     Sector_std       main sector, with Land-Use + Forestry merged to LULUCF
#                      and Sector == "Other" reclassified from Sector_Other
#                      keywords (transport / buildings -> Energy, cement /
#                      steel -> Industry, livestock / crops -> Agriculture ...)
#     Subsector_std    controlled subsector read from Sector_Other (~25 buckets)
#     Sector_std_source  landuse_merge | from_sector_other | original
#
# NET-ZERO FLAG
#
#     <output_dir>/netzero-flags.csv  -- one row per (paper, canonical scenario,
#     canonical area, basis) that carries net-zero evidence.  series_source
#     records which tier was used, most reliable first:
#         all_sectors_total / sector_sum    >=2-year trajectory, ratio test
#                                           last_year <= FRACTION * first_year
#         single_endpoint_total / ..._sector_sum   one late year, emissions <= 0
#         relative_reduction                a late "-95 % vs base" or index point
#     Net-negative end states always qualify.  FRACTION is --netzero-fraction
#     (default 0.05).  ratio_last_to_first (or, for relative rows, the remaining
#     fraction) is written so the threshold can be changed without re-running.
#
# OUTPUT
#
#     <output_dir>/harmonized-<category>.csv
#     original columns (with Context_Filled blanks completed)
#       + Context_Filled  -- ';'-joined names of fields filled from siblings
#       + Scenario_canon, Area_canon  -- canonical group/join keys
#       + Sector_std, Subsector_std, Sector_std_source
#       + per value field:
#         <F>_std, <F>_std_Unit, <F>_factor_physical,
#         <F>_currency_in, <F>_base_year_in, <F>_base_year_src,
#         <F>_fx_rate, <F>_us_deflator_factor, <F>_status
#     <output_dir>/netzero-flags.csv
#
# USAGE  (run from the repo root)
#
#     python3 scripts/04-stage3c-harmonize.py
#     python3 scripts/04-stage3c-harmonize.py outputs/output-stage3c outputs/output-stage3c \
#         --netzero-fraction 0.05 --refresh

from pathlib import Path
import argparse
import csv
import json
import re
import statistics
import sys
import time
import urllib.request
import urllib.error

import truststore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_common import canon_scenario, canon_area   # noqa: E402

truststore.inject_into_ssl()


# ============================================================================
# CONFIG
# ============================================================================

TARGET_YEAR = 2025

DEFAULT_INPUT_DIR = Path("outputs/output-stage3c")
REFERENCE_DIR = Path("inputs/reference")

WB_API = "https://api.worldbank.org/v2"
WB_FX_INDICATOR = "PA.NUS.FCRF"          # LCU per US$, period average
WB_DEFLATOR_INDICATOR = "NY.GDP.DEFL.ZS"  # GDP deflator index

HTTP_RETRIES = 3
HTTP_RETRY_SECONDS = 5

TOE_MWH = 11.63
TCE_MWH = 8.141
BCM_MWH = 10.6e6          # 10.6 TWh
CARBON_TO_CO2 = 44.0 / 12.0

# currency token (upper-case, after preclean) -> World Bank economy code
CURRENCY_TO_WB = {
    "USD": "USA",
    "EUR": "EMU",
    "CNY": "CHN",
    "AUD": "AUS",
    "CHF": "CHE",
    "JPY": "JPN",
    "THB": "THA",
    "CAD": "CAN",
    "GBP": "GBR",
    "INR": "IND",
    "SEK": "SWE",
    "NOK": "NOR",
    "DKK": "DNK",
    "KRW": "KOR",
    "BRL": "BRA",
    "ZAR": "ZAF",
}

CATEGORY_FIELDS = {
    "mitigation": [
        ("Capacity", "Capacity_Unit", None),
        ("Amount", "Amount_Unit", None),
        ("Investment_Cost", "Investment_Cost_Unit",
         "Investment_Cost_Currency_Base_Year"),
        ("Variable_Cost", "Variable_Cost_Unit",
         "Variable_Cost_Currency_Base_Year"),
    ],
    "costs": [
        ("Value", "Value_Unit", "Currency_Base_Year"),
    ],
    "emissions": [
        ("Value", "Value_Unit", None),
    ],
}

# Fields the extractor often leaves blank on some rows of a paper while
# filling them on others (Area is the canonical case: a single-country study
# that labels only a handful of its rows).  A blank is filled from the
# paper's sibling rows ONLY when they carry exactly one distinct non-blank
# value for that field, i.e. the paper is unambiguous about it.  Every fill
# is listed in the Context_Filled column so it can be audited.
CONTEXT_FILL_FIELDS = {
    "emissions": ["Title", "Publication_Year", "DOI", "Area"],
    "costs": ["Title", "Publication_Year", "DOI", "Area",
              "Currency_Base_Year"],
    "mitigation": ["Title", "Publication_Year", "DOI", "Area",
                   "Investment_Cost_Currency_Base_Year",
                   "Variable_Cost_Currency_Base_Year"],
}


def backfill_from_context(rows, fields):
    """Fill a blank `field` on a row from the other rows of the same paper
    when those rows agree on exactly one non-blank value.  Mutates `rows`,
    writes a ';'-joined Context_Filled on every row, and returns a
    {field: n_rows_filled} tally."""
    by_paper = {}
    for row in rows:
        by_paper.setdefault(row.get("paper_id", ""), []).append(row)

    # (paper_id, field) -> the single agreed non-blank value, or None
    consensus = {}
    for paper_id, group in by_paper.items():
        for field in fields:
            seen = {(r.get(field) or "").strip() for r in group}
            seen.discard("")
            consensus[(paper_id, field)] = seen.pop() if len(seen) == 1 else None

    tally = {}
    for row in rows:
        filled = []
        for field in fields:
            if (row.get(field) or "").strip():
                continue
            value = consensus.get((row.get("paper_id", ""), field))
            if value:
                row[field] = value
                filled.append(field)
                tally[field] = tally.get(field, 0) + 1
        row["Context_Filled"] = ";".join(filled)
    return tally


# ============================================================================
# REFERENCE DATA  (World Bank Indicators API)
# ============================================================================

def _http_get_json(url):
    last_error = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "stage3c-harmonize"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < HTTP_RETRIES:
                time.sleep(HTTP_RETRY_SECONDS)
    raise RuntimeError(f"failed to GET {url}: {last_error}")


def _wb_series(indicator, economies):
    """Return {economy: {year: value}} for a World Bank indicator."""
    joined = ";".join(economies)
    url = (
        f"{WB_API}/country/{joined}/indicator/{indicator}"
        f"?format=json&per_page=20000&date=1960:{TARGET_YEAR}"
    )
    payload = _http_get_json(url)

    if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
        raise RuntimeError(f"unexpected World Bank payload for {indicator}")

    series = {}
    for record in payload[1]:
        value = record.get("value")
        if value is None:
            continue
        economy = record.get("countryiso3code") or record.get("country", {}).get("id")
        year = int(record["date"])
        series.setdefault(economy, {})[year] = float(value)
    return series


def load_reference_data(refresh):
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    fx_path = REFERENCE_DIR / "wb_fx_pa_nus_fcrf.json"
    deflator_path = REFERENCE_DIR / "wb_us_gdp_deflator.json"

    economies = sorted(set(CURRENCY_TO_WB.values()))

    if refresh or not fx_path.exists():
        print("Fetching FX rates from the World Bank ...")
        fx = _wb_series(WB_FX_INDICATOR, economies)
        fx_path.write_text(json.dumps(fx, indent=1), encoding="utf-8")
    else:
        fx = json.loads(fx_path.read_text(encoding="utf-8"))

    if refresh or not deflator_path.exists():
        print("Fetching the US GDP deflator from the World Bank ...")
        deflator = _wb_series(WB_DEFLATOR_INDICATOR, ["USA"])
        deflator_path.write_text(json.dumps(deflator, indent=1), encoding="utf-8")
    else:
        deflator = json.loads(deflator_path.read_text(encoding="utf-8"))

    # normalise year keys to int (json keys are strings)
    fx = {
        economy: {int(y): v for y, v in years.items()}
        for economy, years in fx.items()
    }
    us_deflator = {
        int(y): v for y, v in deflator.get("USA", {}).items()
    }

    if not us_deflator:
        raise RuntimeError("no US GDP deflator values returned")

    return fx, us_deflator


class MoneyConverter:
    def __init__(self, fx, us_deflator):
        self.fx = fx
        self.us_deflator = us_deflator
        self.deflator_years = sorted(us_deflator)
        self._prepare_target_deflator()

    def _prepare_target_deflator(self):
        last = self.deflator_years[-1]
        if last >= TARGET_YEAR:
            self.target_deflator = self.us_deflator[TARGET_YEAR]
            self.target_extrapolated = False
        else:
            prev = self.deflator_years[-2]
            growth = self.us_deflator[last] / self.us_deflator[prev]
            factor = growth ** (TARGET_YEAR - last)
            self.target_deflator = self.us_deflator[last] * factor
            self.target_extrapolated = True

    def _deflator_for(self, year):
        if year in self.us_deflator:
            return self.us_deflator[year], False
        if year < self.deflator_years[0]:
            return self.us_deflator[self.deflator_years[0]], True
        if year > self.deflator_years[-1]:
            # base year at or past the latest data (e.g. a 2026 paper with no
            # stated price year) -> no deflation, factor ~ 1.  Not flagged:
            # the "base_year=from_publication" flag already signals this.
            return self.us_deflator[self.deflator_years[-1]], False
        # gap year within range: nearest below
        below = max(y for y in self.deflator_years if y < year)
        return self.us_deflator[below], True

    def _fx_for(self, currency, year):
        """LCU per USD for the given year (1.0 for USD)."""
        if currency == "USD":
            return 1.0, year, False
        economy = CURRENCY_TO_WB.get(currency)
        table = self.fx.get(economy, {})
        if not table:
            return None, None, True
        if year in table:
            return table[year], year, False
        nearest = min(table, key=lambda y: abs(y - year))
        return table[nearest], nearest, True

    def to_real_usd_2025(self, value, currency, base_year):
        """Return dict with converted value and provenance."""
        flags = []

        fx_rate, fx_year, fx_clamped = self._fx_for(currency, base_year)
        if fx_rate is None:
            return {
                "value": None,
                "fx_rate": "",
                "us_deflator_factor": "",
                "status": "fx_unavailable",
            }
        if fx_clamped:
            flags.append(f"fx_year={fx_year}")

        deflator_base, base_clamped = self._deflator_for(base_year)
        if base_clamped:
            flags.append("deflator_clamped")
        if self.target_extrapolated:
            flags.append("deflator_2025_extrapolated")

        deflator_factor = self.target_deflator / deflator_base
        usd_nominal = value / fx_rate
        usd_real = usd_nominal * deflator_factor

        return {
            "value": usd_real,
            "fx_rate": fx_rate,
            "us_deflator_factor": deflator_factor,
            "status": "ok" if not flags else ";".join(flags),
        }


# ============================================================================
# UNIT PARSING
# ============================================================================

MULTIPLIER_WORDS = [
    (r"\btrillion\b", 1e12), (r"\bbillion\b", 1e9), (r"\bmilliard\b", 1e9),
    (r"\bmillion\b", 1e6), (r"\bmil\.?\b", 1e6), (r"\bthousand\b", 1e3),
    (r"\bhundred\b", 1e2),
    (r"\bmrd\b", 1e9), (r"\bbn\b", 1e9), (r"\bbln\b", 1e9),
    (r"\btn\b", 1e12),
]

# leading one-letter magnitude prefixes glued to a currency token.
# The lookahead keeps the currency in place; only the prefix letter is removed.
_CUR_LA = r"(?=\s*(?:\$|€|£|¥|eur|usd|us\$|au\$|aud|cad|chf|gbp|jpy|rmb|cny|thb|sek|nok|dkk|inr))"
SYMBOL_MULT = [
    (r"^t" + _CUR_LA, 1e12),
    (r"^b" + _CUR_LA, 1e9),
    (r"^g" + _CUR_LA, 1e9),
    (r"^m" + _CUR_LA, 1e6),
    (r"^k" + _CUR_LA, 1e3),
]

CURRENCY_ALIASES = [
    (r"us\$|usd|u\.s\. dollar|dollars?|\$", "USD"),
    (r"€|eur\b|euros?", "EUR"),
    (r"yuan|renminbi|rmb|cny|¥", "CNY"),
    (r"au\$|aud|australian dollar", "AUD"),
    (r"chf|swiss franc", "CHF"),
    (r"jpy|yen", "JPY"),
    (r"thb|baht", "THB"),
    (r"cad|canadian dollar", "CAD"),
    (r"£|gbp|pound", "GBP"),
    (r"\brs\.?\b|inr|rupee", "INR"),
    (r"sek", "SEK"), (r"nok", "NOK"), (r"dkk", "DKK"),
    (r"krw|won", "KRW"), (r"brl|real", "BRL"), (r"zar|rand", "ZAR"),
]

# physical conversion factors -> target unit
POWER_TO_MW = {
    "w": 1e-6, "kw": 1e-3, "mw": 1.0, "gw": 1e3, "tw": 1e6, "pw": 1e9,
    "we": 1e-6, "kwe": 1e-3, "mwe": 1.0, "gwe": 1e3,
    "wel": 1e-6, "kwel": 1e-3, "mwel": 1.0, "gwel": 1e3,
    "wp": 1e-6, "kwp": 1e-3, "mwp": 1.0, "gwp": 1e3,
}
ENERGY_TO_MWH = {
    "wh": 1e-6, "kwh": 1e-3, "mwh": 1.0, "gwh": 1e3, "twh": 1e6, "pwh": 1e9,
    "j": 2.777778e-10, "kj": 2.777778e-7, "mj": 2.777778e-4,
    "gj": 2.777778e-1, "tj": 2.777778e2, "pj": 2.777778e5, "ej": 2.777778e8,
    "wa": 8760e-6, "kwa": 8760e-3, "mwa": 8760.0, "gwa": 8.76e6, "twa": 8.76e9,
    "cal": 1.163e-9, "kcal": 1.163e-6, "mcal": 1.163e-3, "gcal": 1.163,
    "pcal": 1.163e9,
    "btu": 2.930711e-10, "mmbtu": 0.2930711, "mbtu": 0.2930711e-3,
    "quad": 2.930711e8,
}
ENERGY_SPECIAL = {          # (factor_to_MWh, flag)
    "toe": (TOE_MWH, "generic_calorific"),
    "ktoe": (TOE_MWH * 1e3, "generic_calorific"),
    "mtoe": (TOE_MWH * 1e6, "generic_calorific"),
    "tce": (TCE_MWH, "generic_calorific"),
    "ktce": (TCE_MWH * 1e3, "generic_calorific"),
    "mtce": (TCE_MWH * 1e6, "generic_calorific"),
    "sce": (TCE_MWH, "generic_calorific"),
    "tsce": (TCE_MWH, "generic_calorific"),
    "bcm": (BCM_MWH, "generic_calorific"),
    "mcm": (BCM_MWH / 1e3, "generic_calorific"),
    "bcom": (BCM_MWH, "generic_calorific"),
}
CO2_TO_MT = {
    "g": 1e-12, "kg": 1e-9, "t": 1e-6, "kt": 1e-3, "mt": 1.0, "gt": 1e3,
    "pt": 1e6, "tg": 1.0, "pg": 1e3, "eg": 1e6, "mg": 1e-6, "gg": 1e-3,
}
AREA_TO_KM2 = {
    "m2": 1e-6, "m²": 1e-6, "km2": 1.0, "km²": 1.0, "ha": 1e-2,
    "mha": 1e4, "kha": 1e1, "mkm2": 1e6, "acre": 4.046856e-3,
    "acres": 4.046856e-3, "hectare": 1e-2, "hectares": 1e-2,
}

# denominator token -> (canonical token, factor applied to the VALUE)
# value is "numerator per denominator"; e.g. $/kWh -> $/MWh multiplies the
# value by 1000 because 1 MWh = 1000 kWh.  factor = 1 / (denom_unit in target).
DENOM_CANON = {
    "wh": ("MWh", 1e6), "kwh": ("MWh", 1e3), "mwh": ("MWh", 1.0),
    "gwh": ("MWh", 1e-3), "twh": ("MWh", 1e-6),
    "j": ("MWh", 3.6e9), "kj": ("MWh", 3.6e6), "mj": ("MWh", 3600.0),
    "gj": ("MWh", 3.6), "tj": ("MWh", 3.6e-3), "pj": ("MWh", 3.6e-6),
    "mmbtu": ("MWh", 3.412142),
    "w": ("MW", 1e6), "kw": ("MW", 1e3), "mw": ("MW", 1.0), "gw": ("MW", 1e-3),
    "kwyr": ("MW.yr", 1e3), "mwyr": ("MW.yr", 1.0), "kwyr": ("MW.yr", 1e3),
    "tco2": ("tCO2", 1.0), "tco2e": ("tCO2eq", 1.0), "tco2eq": ("tCO2eq", 1.0),
    "tc": ("tCO2", 1.0 / CARBON_TO_CO2),
    "t": ("t", 1.0), "ton": ("t", 1.0), "tonne": ("t", 1.0), "kg": ("kg", 1.0),
    "yr": ("yr", 1.0), "day": ("day", 1.0),
    "cap": ("capita", 1.0), "capita": ("capita", 1.0), "person": ("capita", 1.0),
}

# numerator strings that are explicitly NOT convertible to a target unit
OTHER_MARKERS = re.compile(
    r"(^|[\s(])("
    r"%|percent|percentage points?|pp|share|index|ratio|factor|fold|"
    r"times|log|dimensionless|au|a\.u\.|relative|"
    r"vehicles?|turbines?|trees?|units?|plants?|households?|coach|"
    r"people|persons?|pkm|passenger|vehicle-km|"
    r"years?|days?"
    r")($|[\s)/])",
    re.IGNORECASE,
)


def preclean_unit(raw):
    u = (raw or "").strip()
    if not u:
        return ""
    u = u.replace(" ", " ")
    # unicode -> ascii
    u = (u.replace("²", "2").replace("³", "3")
           .replace("₂", "2").replace("·", ".").replace("•", ".")
           .replace("−", "-").replace("–", "-").replace("—", "-")
           .replace("’", "'").replace("×", "x")
           .replace("⁻¹", "-1").replace("⁻", "-").replace("¹", "1"))
    u = re.sub(r"\b(kilo|mega|giga|tera)\s*ton(ne)?s?\b",
               lambda m: {"kilo": "kt", "mega": "Mt", "giga": "Gt",
                          "tera": "Tt"}[m.group(1).lower()], u, flags=re.IGNORECASE)
    u = re.sub(r"\bCO\s*2\b", "CO2", u, flags=re.IGNORECASE)
    u = re.sub(r"CO2[\s\-().]*?(eqv|eq|equiv\.?|equivalent|e)\b", "CO2eq", u,
               flags=re.IGNORECASE)
    u = re.sub(r"\bMt\s*CO2", "MtCO2", u, flags=re.IGNORECASE)
    # coal-equivalent phrases (consume the preceding mass word) -> "tce"
    u = re.sub(
        r"\b(?:t|tons?|tonnes?|metric tons?)\s+(?:of\s+)?"
        r"(?:standard coal(?:\s+equivalent)?|sce)\b",
        "tce", u, flags=re.IGNORECASE,
    )
    u = re.sub(r"\bstandard coal(?:\s+equivalent)?\b", "tce", u,
               flags=re.IGNORECASE)
    u = re.sub(r"\b(tonnes?|tons?|metric tons?|mt\.)\b", "t", u,
               flags=re.IGNORECASE)
    u = re.sub(r"\bof\b", " ", u, flags=re.IGNORECASE)
    u = re.sub(r"\bper\b", "/", u, flags=re.IGNORECASE)
    # inverse-power time notation:  "yr-1", "a-1", "yr^-1"  ->  "/yr"
    u = re.sub(r"\b(yr|year|a)\s*\^?\s*-\s*1\b", "/yr", u, flags=re.IGNORECASE)
    u = re.sub(r"\bp\.?\s*a\.?\b", "/yr", u, flags=re.IGNORECASE)
    u = re.sub(r"\b(yr|y|a|annum|annually|year)\b(?![a-z])", "yr", u,
               flags=re.IGNORECASE)
    u = re.sub(r"\s*/\s*", "/", u)
    u = re.sub(r"\s+", " ", u).strip()
    return u


def extract_multiplier(u):
    """Pull magnitude words / 10^n / xN out of the string. Returns (mult, rest)."""
    mult = 1.0
    rest = u

    for pattern, mval in SYMBOL_MULT:
        if re.search(pattern, rest, flags=re.IGNORECASE):
            mult *= mval
            rest = re.sub(pattern, "", rest, count=1, flags=re.IGNORECASE)

    # magnitude glued *after* a currency symbol:  "$B", "$ bn", "€ bn"
    for pattern, mval in [(r"[\$€£¥]\s*bn?\b", 1e9), (r"[\$€£¥]\s*bln\b", 1e9),
                          (r"[\$€£¥]\s*tn?\b", 1e12), (r"[\$€£¥]\s*m\b", 1e6)]:
        if re.search(pattern, rest, flags=re.IGNORECASE):
            mult *= mval
            rest = re.sub(r"(?<=[\$€£¥])\s*(bn?|bln|tn?|m)\b", "", rest,
                          count=1, flags=re.IGNORECASE)

    for pattern, mval in MULTIPLIER_WORDS:
        for _ in re.findall(pattern, rest, flags=re.IGNORECASE):
            mult *= mval
        rest = re.sub(pattern, " ", rest, flags=re.IGNORECASE)

    # 10^n , 10^-n , x10^n   (the caret is required -- do NOT match bare "1000")
    for m in re.finditer(r"(?:x\s*)?10\s*\^\s*(-?\d+)", rest):
        mult *= 10.0 ** int(m.group(1))
    rest = re.sub(r"(?:x\s*)?10\s*\^\s*-?\d+", " ", rest)

    # a bare leading number ("1000 JPY/kW", "10 thousand tce", "6.6 MW turbines")
    m = re.match(r"\s*([\d.]+)\s+(?=[a-zA-Z€$£¥])", rest)
    if m:
        try:
            mult *= float(m.group(1))
            rest = rest[m.end():]
        except ValueError:
            pass

    rest = re.sub(r"\s+", " ", rest).strip(" .")
    return mult, rest


def extract_base_year(u):
    m = re.search(r"(?<!\d)(19[6-9]\d|20[0-3]\d)(?!\d)", u or "")
    return int(m.group(1)) if m else None


def strip_year(u):
    return re.sub(r"(?<!\d)(19[6-9]\d|20[0-3]\d)(?!\d)", " ", u or "").strip()


def split_rate(u):
    parts = [p.strip() for p in u.split("/") if p.strip()]
    if not parts:
        return "", []
    return parts[0], parts[1:]


def _norm_token(tok):
    t = tok.lower().strip()
    t = t.replace(" ", "").replace(".", "").replace("-", "")
    t = t.replace("(", "").replace(")", "")
    return t


def detect_currency(u):
    low = u.lower()
    for pattern, code in CURRENCY_ALIASES:
        if re.search(pattern, low):
            return code
    return None


_DESCRIPTOR_WORDS = re.compile(
    r"\b(biomass|bioenergy|electricity|electrical|electric|heat|thermal|"
    r"primary|final|gross|net|total|nuclear|renewable|renewables|solar|pv|"
    r"wind|hydro|hydrogen|fossil|natural gas|gas|oil|coal|clinker|cement|"
    r"ammonia|steel|captured|sequestered|abated|avoided|reduced|removed|"
    r"annual|cumulative|installed|new|added|of)\b",
    re.IGNORECASE,
)

_CO2_MEASURES = {
    "carbon dioxide removal",
    "carbon capture and storage (ccs)",
    "carbon capture and utilization (ccu)",
}


def classify_numerator(numerator, category, measure=""):
    """Return (dimension, std_unit, factor_to_std, flags[list])."""
    flags = []

    numerator = _DESCRIPTOR_WORDS.sub(" ", numerator).strip()
    core = _norm_token(numerator)

    if not core:
        return ("empty", "", None, ["unit_missing"])

    if OTHER_MARKERS.search(numerator):
        return ("other", numerator.strip(), None, ["not_convertible"])

    # ---- money ---------------------------------------------------------
    currency = detect_currency(numerator)
    if currency is not None:
        return ("money", currency, 1.0, flags)

    # ---- CO2 / carbon -----------------------------------------------
    # core looks like  co2 | mtco2 | gtco2eq | tc | tgc | mtco  (last = not CO2)
    m = re.match(r"^([a-z]*)(co2eq|co2|c)$", core)
    if m:
        prefix = m.group(1) or "mt"
        base = m.group(2)
        if prefix in CO2_TO_MT:
            factor = CO2_TO_MT[prefix]
            if base == "c":
                factor *= CARBON_TO_CO2
                flags.append("carbon_to_co2")
                std = "MtCO2"
            else:
                std = "MtCO2eq" if base == "co2eq" else "MtCO2"
            return ("co2", std, factor, flags)

    # bare mass token -> assume CO2 in an emissions row, or in a mitigation
    # row whose measure is inherently a CO2 quantity (CDR / CCS / CCU)
    if core in CO2_TO_MT and (
        category == "emissions"
        or measure.strip().lower() in _CO2_MEASURES
    ):
        flags.append("assumed_co2")
        return ("co2", "MtCO2", CO2_TO_MT[core], flags)

    # ---- energy special (toe / tce / bcm) ---------------------------
    for key, (factor, flag) in ENERGY_SPECIAL.items():
        if core == key or core == key + "e":
            flags.append(flag)
            return ("energy", "MWh", factor, flags)

    # ---- power (before energy: 'w' vs 'wh') -----------------------
    if core in POWER_TO_MW:
        return ("power", "MW", POWER_TO_MW[core], flags)

    # ---- energy -----------------------------------------------------
    if core in ENERGY_TO_MWH:
        return ("energy", "MWh", ENERGY_TO_MWH[core], flags)

    # ---- area -----------------------------------------------------
    if core in AREA_TO_KM2:
        return ("area", "km2", AREA_TO_KM2[core], flags)
    m = re.match(r"^(m|k|g|t)?(km2|m2|ha)$", core)
    if m and core in AREA_TO_KM2:
        return ("area", "km2", AREA_TO_KM2[core], flags)

    # ---- generic prefixed fallbacks ------------------------------
    m = re.match(r"^(k|m|g|t|p)(wh|w|j|toe|tce)$", core)
    if m:
        prefix, base = m.groups()
        pmap = {"k": 1e3, "m": 1e6, "g": 1e9, "t": 1e12, "p": 1e15}
        if base == "w":
            return ("power", "MW", pmap[prefix] * 1e-6, flags)
        if base == "wh":
            return ("energy", "MWh", pmap[prefix] * 1e-6, flags)
        if base == "j":
            return ("energy", "MWh", pmap[prefix] * 2.777778e-10, flags)
        if base in ("toe", "tce"):
            flags.append("generic_calorific")
            unit_mwh = TOE_MWH if base == "toe" else TCE_MWH
            return ("energy", "MWh", pmap[prefix] * unit_mwh, flags)

    return ("unknown", numerator.strip(), None, ["unit_unparsed"])


def canon_denominators(denominators):
    """Return (canonical '/a/b' string, combined value factor, flags)."""
    if not denominators:
        return "", 1.0, []
    factor = 1.0
    canon_parts = []
    flags = []
    for denom in denominators:
        key = _norm_token(denom)
        if key in DENOM_CANON:
            canon, dfactor = DENOM_CANON[key]
            canon_parts.append(canon)
            factor *= dfactor
        else:
            canon_parts.append(denom.strip())
            if key:
                flags.append(f"denom_kept:{denom.strip()}")
    return "/" + "/".join(canon_parts), factor, flags


# ============================================================================
# SECTOR HARMONISATION
# ============================================================================

# Land use and forestry -> the IPCC land-use aggregate.
SECTOR_ALIASES = {
    "Land-Use": "LULUCF", "Land Use": "LULUCF", "Land use": "LULUCF",
    "Forestry": "LULUCF", "LULUCF": "LULUCF",
}

# Sector_Other keyword -> (canonical subsector, implied main sector or None).
# First match wins; specific rules before general ones.
SECTOR_OTHER_RULES = [
    (r"\b(dac|daccs)\b|direct air capture", "CDR: DACCS", "Energy"),
    (r"\bbeccs\b|bioenergy with (carbon|ccs)", "CDR: BECCS", "Energy"),
    (r"afforest|reforest|forest carbon|forest (products|uptake)|"
     r"land restoration|restored land|harvested wood|urban green|grassland",
     "CDR: land & forestry", "LULUCF"),
    (r"carbon dioxide removal|carbon removal|negative emission|"
     r"co2 (transport|storage|capture)|carbon capture|\bccs\b|\bccus\b",
     "CDR: unspecified", None),

    (r"electricity storage|battery|storage system|pumped hydro|"
     r"demand response|flexibility", "Power: storage & flexibility", "Energy"),
    (r"transmission|electricity network|interconnect|\bt&d\b|\bgrid\b",
     "Power: transmission & distribution", "Energy"),
    (r"electric|power sector|power system|power generation|\bpower\b|"
     r"coal.?fired power|clean electricity|solar (electric|power)|"
     r"wind (electric|power)|renewable electric|heat and power|onshore wind|"
     r"offshore wind|hydropower|nuclear|thermal (production|power)|"
     r"energy transformation", "Power: generation", "Energy"),
    (r"district heating|heat network|process heat|heat (production|supply)|"
     r"\bheating\b|thermal$|^heat$", "Heat", "Energy"),
    (r"hydrogen|synthetic (fuel|methane|natural gas)|e-fuel|synfuel|"
     r"power-to-|methanol|\bmethane\b", "Hydrogen & synthetic fuels", "Energy"),
    (r"bioenerg|biofuel|biomass|biorefiner|biogas|biomethane",
     "Bioenergy", "Energy"),
    (r"(oil|gas|coal|petroleum|fossil|lng)\b.*(supply|import|export|"
     r"production|extraction|end.?use)|refined oil|natural gas (fuel|end use)|"
     r"coal end use|energy (import|export)|\bmining\b", "Fossil fuel supply", "Energy"),

    (r"transport|aviation|shipping|maritime|mobility|\bfreight\b",
     "Transport", "Energy"),
    (r"residential|\bbuildings?\b|\bhousing\b|\bdwelling\b|"
     r"construction material|in-use concrete|household", "Buildings", "Energy"),
    (r"commercial|tertiary|\bservices?\b", "Services", "Energy"),
    (r"energy.?related|energy system|stationary energy|energy demand|"
     r"final energy|primary energy", "Energy: unspecified", "Energy"),

    (r"cement|clinker|concrete", "Industry: cement", "Industry"),
    (r"\bsteel\b|iron and steel|blast furnace|\biron\b",
     "Industry: iron & steel", "Industry"),
    (r"ammonia|nitrogen fertili|\bchemical|petrochemical|plastics",
     "Industry: chemicals", "Industry"),
    (r"\bpulp\b|\bpaper\b", "Industry: pulp & paper", "Industry"),
    (r"food processing|food supply|food consumption|food and beverage",
     "Industry: food processing", "Industry"),
    (r"alumin|\bglass\b|ceramics|refiner|non.?metallic",
     "Industry: other", "Industry"),
    (r"industrial process|\bindustry\b|manufactur", "Industry: other", "Industry"),

    (r"agricultural waste|waste burning|\bsewage|\bsludge|landfill|"
     r"wastewater|solid waste", "Waste", "Waste"),

    (r"land use, land.?use change|\blulucf\b|\bafolu\b|land.?use change|"
     r"agriculture, forestry|forestry and (other )?land", "LULUCF / AFOLU", "LULUCF"),
    (r"beef|sheep|cattle|livestock|dairy|\banimal|\bmeat\b|fisheries|poultry",
     "Agriculture: livestock", "Agriculture"),
    (r"\brice\b|\bwheat\b|cereal|maize|\bcrop|vegetable|fruit|oilseed|"
     r"sugar crop|fibre crop|\bgrain|arable", "Agriculture: crops", "Agriculture"),
    (r"agricultur|farming", "Agriculture: unspecified", "Agriculture"),
    (r"\bforest|wetland|mangrove|peatland|ecological land|impervious land|"
     r"green (open )?space", "Forestry & other land", "LULUCF"),

    (r"whole economy|economy.?wide|^economy$|cross.?sector|all sectors|"
     r"final demand|energy.?using sectors|end.?use sectors",
     "Whole economy", "All sectors"),
    (r"emission trading|carbon market|carbon (price|revenue)|"
     r"opportunity cost|\brevenue\b", "Cross-cutting / market", None),
    (r"\bwater\b", "Water", None),
    (r"\bland (requirement|footprint)\b|land[- ]use footprint",
     "Land footprint", None),
    (r"employment|\bjobs?\b|\bgdp\b", "Macroeconomic", None),
]

_COMPILED_SECTOR_RULES = [
    (re.compile(p, re.IGNORECASE), sub, main)
    for p, sub, main in SECTOR_OTHER_RULES
]


def classify_subsector(sector_other):
    """Return (canonical subsector, implied main sector or None)."""
    text = (sector_other or "").strip().lower()
    if not text:
        return "", None
    for pattern, subsector, implied_main in _COMPILED_SECTOR_RULES:
        if pattern.search(text):
            return subsector, implied_main
    return "", None


def harmonize_sector(sector, sector_other):
    """Return (sector_std, subsector_std, source).

    source: landuse_merge | from_sector_other | original | unmapped_other
    """
    sector = (sector or "").strip()
    subsector, implied_main = classify_subsector(sector_other)

    if sector in SECTOR_ALIASES and SECTOR_ALIASES[sector] != sector:
        return SECTOR_ALIASES[sector], subsector, "landuse_merge"

    if sector in ("", "Other") and implied_main:
        return implied_main, subsector, "from_sector_other"

    if sector == "":
        return "Other", subsector, "unmapped_other"

    return sector, subsector, "original"


# ============================================================================
# VALUE PARSING
# ============================================================================

def parse_value(raw):
    s = (raw or "").strip()
    if not s:
        return None
    s = s.replace("−", "-").replace("–", "-")
    s = s.replace("~", "").replace("≈", "").replace("about", "").strip()
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        pass
    m = re.search(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


# ============================================================================
# ROW HARMONISATION
# ============================================================================

_NOT_REPORTED = {"not reported", "notreported", "na", "n/a", "none", "unknown",
                 "-", "--", "?", "tbd"}


def fmt_num(x):
    """10 significant figures, plain (not scientific unless tiny/huge)."""
    return float(f"{x:.10g}")


def harmonize_field(raw_value, raw_unit, raw_base_year, publication_year,
                    category, money, measure=""):
    """Return the dict of <F>_* output columns for one value field."""

    out = {
        "std": "", "std_unit": "", "factor_physical": "",
        "currency_in": "", "base_year_in": "", "base_year_src": "",
        "fx_rate": "", "us_deflator_factor": "", "status": "",
    }

    value = parse_value(raw_value)
    unit_clean = preclean_unit(raw_unit)

    if value is None and not (raw_value or "").strip():
        out["status"] = "no_value"
        return out
    if not unit_clean or unit_clean.lower() in _NOT_REPORTED:
        out["status"] = "unit_not_reported"
        return out
    if value is None:
        out["status"] = "value_unparsed"
        return out

    embedded_year = extract_base_year(unit_clean)
    mult, rest = extract_multiplier(strip_year(unit_clean))
    numerator, denominators = split_rate(rest)

    dimension, std_unit, factor, flags = classify_numerator(
        numerator, category, measure
    )
    denom_str, denom_factor, denom_flags = canon_denominators(denominators)
    flags = list(flags) + denom_flags

    if dimension in ("unknown", "other", "empty"):
        out["status"] = ";".join(flags) or dimension
        return out

    if dimension == "money":
        currency = std_unit  # classify_numerator returns the currency code here

        base_year, base_src = None, ""
        if (raw_base_year or "").strip().isdigit():
            base_year = int(raw_base_year.strip())
            base_src = "explicit"
        elif embedded_year is not None:
            base_year = embedded_year
            base_src = "unit_string"
        elif (publication_year or "").strip().isdigit():
            base_year = int(publication_year.strip())
            base_src = "publication_year"

        out["currency_in"] = currency
        out["base_year_in"] = base_year if base_year is not None else ""
        out["base_year_src"] = base_src

        if base_year is None:
            out["status"] = "no_base_year"
            return out

        nominal = value * mult * denom_factor
        conv = money.to_real_usd_2025(nominal, currency, base_year)
        out["fx_rate"] = conv["fx_rate"]
        out["us_deflator_factor"] = conv["us_deflator_factor"]

        if conv["value"] is None:
            out["status"] = conv["status"]
            return out

        out["std"] = fmt_num(conv["value"])
        out["std_unit"] = f"USD{TARGET_YEAR}{denom_str}"
        status_bits = [b for b in [conv["status"]] if b and b != "ok"]
        status_bits += [f for f in flags if not f.startswith("denom_kept")]
        if base_src == "publication_year":
            status_bits.append("base_year=from_publication")
        out["status"] = "ok" if not status_bits else ";".join(status_bits)
        return out

    # physical
    total_factor = mult * factor * denom_factor
    out["std"] = fmt_num(value * total_factor)
    out["std_unit"] = std_unit + denom_str
    out["factor_physical"] = total_factor
    out["status"] = "ok" if not flags else ";".join(flags)
    return out


def harmonize_category(category, input_dir, output_dir, money):
    in_path = input_dir / f"merged-{category}.csv"
    out_path = output_dir / f"harmonized-{category}.csv"

    with open(in_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
        base_fields = list(reader.fieldnames or [])

    specs = CATEGORY_FIELDS[category]
    suffixes = [
        "std", "std_Unit", "factor_physical", "currency_in",
        "base_year_in", "base_year_src", "fx_rate",
        "us_deflator_factor", "status",
    ]
    new_fields = ["Context_Filled", "Scenario_canon", "Area_canon",
                  "Sector_std", "Subsector_std", "Sector_std_source"]
    for value_field, _unit_field, _by in specs:
        for suffix in suffixes:
            new_fields.append(f"{value_field}_{suffix}")

    # fill blanks that the paper's other rows resolve unambiguously
    # (before harmonising, so e.g. a back-filled base year feeds the money
    # pipeline and a back-filled Area feeds the net-zero grouping)
    fill_tally = backfill_from_context(rows, CONTEXT_FILL_FIELDS[category])

    status_counter = {}
    sector_counter = {}

    for row in rows:
        publication_year = row.get("Publication_Year", "")
        measure = row.get("Mitigation_Measure", "")

        # canonical scenario / area keys (see _pipeline_common) -- every
        # downstream group-by and netzero-flags join uses these, not the raw
        # strings, so surface-form drift no longer fragments a scenario
        row["Scenario_canon"] = canon_scenario(row.get("Scenario_Name", ""))
        row["Area_canon"] = canon_area(row.get("Area", ""))

        sector_std, subsector_std, sector_src = harmonize_sector(
            row.get("Sector", ""), row.get("Sector_Other", "")
        )
        row["Sector_std"] = sector_std
        row["Subsector_std"] = subsector_std
        row["Sector_std_source"] = sector_src
        sector_counter[sector_src] = sector_counter.get(sector_src, 0) + 1

        for value_field, unit_field, base_year_field in specs:
            raw_by = row.get(base_year_field, "") if base_year_field else ""
            result = harmonize_field(
                raw_value=row.get(value_field, ""),
                raw_unit=row.get(unit_field, ""),
                raw_base_year=raw_by,
                publication_year=publication_year,
                category=category,
                money=money,
                measure=measure,
            )
            row[f"{value_field}_std"] = result["std"]
            row[f"{value_field}_std_Unit"] = result["std_unit"]
            row[f"{value_field}_factor_physical"] = result["factor_physical"]
            row[f"{value_field}_currency_in"] = result["currency_in"]
            row[f"{value_field}_base_year_in"] = result["base_year_in"]
            row[f"{value_field}_base_year_src"] = result["base_year_src"]
            row[f"{value_field}_fx_rate"] = result["fx_rate"]
            row[f"{value_field}_us_deflator_factor"] = result["us_deflator_factor"]
            row[f"{value_field}_status"] = result["status"]

            head = (result["status"] or "").split(";")[0] or "empty"
            key = f"{value_field}:{head}"
            status_counter[key] = status_counter.get(key, 0) + 1

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=base_fields + new_fields, delimiter=";",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{category}: {len(rows)} rows -> {out_path}")
    for key in sorted(status_counter):
        print(f"    {status_counter[key]:5d}  {key}")
    print("    -- Sector_std source --")
    for key in sorted(sector_counter):
        print(f"    {sector_counter[key]:5d}  {key}")
    if fill_tally:
        print("    -- context-filled from sibling rows --")
        for key in sorted(fill_tally):
            print(f"    {fill_tally[key]:5d}  {key}")


# ============================================================================
# NET-ZERO SCENARIO FLAG
# ============================================================================

_ABS_CO2_UNITS = {"MtCO2", "MtCO2/yr", "MtCO2eq", "MtCO2eq/yr"}

# earliest year at which a lone "emissions are ~0" or "-95 % vs base" point is
# credible as a net-zero end state rather than a mid-pathway reading
MIN_NETZERO_YEAR = 2035

NETZERO_FIELDS = [
    "paper_id", "Title", "DOI", "Publication_Year", "Scenario_Name", "Area",
    "scenario_canon", "area_canon",
    "basis", "series_source", "n_years", "first_year", "first_emissions_Mt",
    "last_year", "last_emissions_Mt", "ratio_last_to_first",
    "n_sectors_first", "n_sectors_last", "is_net_negative", "is_netzero",
    "netzero_fraction",
]

# raw Value_Unit / detail wording that is NOT an emissions level (so a "%" or
# "index" row carrying it must not be read as a reduction trajectory)
_NOT_A_LEVEL = re.compile(
    r"rate|share|penetration|installation|capacity|per[\s-]?capita|/capita|"
    r"person|/kwh|/mwh|intensity|\bg\s?co2|gco2|/gdp|efficiency|"
    r"utili[sz]ation|deployment|adoption|coverage", re.I)
_REDUCTION_WORD = re.compile(
    r"reduction|below|decrease|cut|abatement|lower than|less than|"
    r"mitigat|avoided|saving", re.I)


def _year_int(raw):
    s = (raw or "").strip()
    if re.fullmatch(r"(?:19|20)\d{2}", s):
        return int(s)
    found = re.findall(r"(?:19|20)\d{2}", s)
    return int(found[-1]) if len(found) == 1 else None


def _relative_remaining(value, unit, detail):
    """Fraction of the reference emissions still remaining (0..1+, or negative
    for a net-negative end state), read from a percentage / index row.
    Returns None when the row is not an emissions level relative to a base."""
    u = (unit or "").lower()
    text = f"{u} {(detail or '').lower()}"
    if _NOT_A_LEVEL.search(text):
        return None
    is_reduction = bool(_REDUCTION_WORD.search(text)) or value < 0
    if ("index" in u or re.search(r"=\s*100", u)) and not is_reduction:
        return value / 100.0                       # Index (YYYY=100)
    if "fraction" in u and not is_reduction:
        return value if abs(value) <= 1.5 else value / 100.0
    return 1.0 - abs(value) / 100.0                 # a reduction percentage


def _classify_series(recs, rel_recs, fraction):
    """Decide the net-zero status of one (paper, scenario, area, basis) group.

    recs      : (year, MtCO2 value, Sector_std) for absolute CO2/CO2e rows
    rel_recs  : (year, remaining_fraction) for percentage / index rows

    Returns a dict of the series_source / year / emission / ratio / flag
    fields, or None when the group carries no usable net-zero evidence.
    Tiers, most reliable first:

      1. >= 2 years of an economy-wide "All sectors" total   -> ratio test
      2. >= 2 years of a >=2-sector sum                       -> ratio test
      3. one late "All sectors" total <= 0                    -> net zero
      4. one late >=2-sector sum <= 0                         -> net zero
      5. a late percentage / index point <= fraction          -> net zero
    """
    totals = [(y, v) for y, v, s in recs if s == "All sectors"]
    sectors = [(y, v, s) for y, v, s in recs if s and s != "All sectors"]

    def _pack(source, series, n_first="", n_last=""):
        yrs = sorted(series)
        fe, le = series[yrs[0]], series[yrs[-1]]
        ratio = le / fe if fe > 0 else ""
        return {
            "series_source": source, "n_years": len(yrs),
            "first_year": yrs[0], "first_emissions_Mt": round(fe, 3),
            "last_year": yrs[-1], "last_emissions_Mt": round(le, 3),
            "ratio_last_to_first": "" if ratio == "" else round(ratio, 4),
            "n_sectors_first": n_first, "n_sectors_last": n_last,
            "is_net_negative": le < 0,
            "is_netzero": (le <= fraction * fe) if fe > 0 else "",
        }

    if len({y for y, _ in totals}) >= 2:
        by = {}
        for y, v in totals:
            by.setdefault(y, []).append(v)
        return _pack("all_sectors_total",
                     {y: statistics.median(v) for y, v in by.items()})

    if len({y for y, _, _ in sectors}) >= 2:
        by = {}
        for y, v, s in sectors:
            by.setdefault(y, {}).setdefault(s, []).append(v)
        series = {y: sum(statistics.median(vs) for vs in sd.values())
                  for y, sd in by.items()}
        ys = sorted(series)
        return _pack("sector_sum", series,
                     len(by[ys[0]]), len(by[ys[-1]]))

    # --- single-endpoint absolute (no first year to normalise against) ---
    def _single(source, year, value, n_sec=""):
        return {
            "series_source": source, "n_years": 1,
            "first_year": "", "first_emissions_Mt": "",
            "last_year": year, "last_emissions_Mt": round(value, 3),
            "ratio_last_to_first": "", "n_sectors_first": "",
            "n_sectors_last": n_sec,
            "is_net_negative": value < 0, "is_netzero": value <= 1e-9,
        }

    if totals:
        y = max(y for y, _ in totals)
        v = statistics.median([vv for yy, vv in totals if yy == y])
        if y >= MIN_NETZERO_YEAR and v <= 1e-9:
            return _single("single_endpoint_total", y, v)

    if sectors:
        y = max(y for y, _, _ in sectors)
        at_y = {}
        for yy, vv, s in sectors:
            if yy == y:
                at_y.setdefault(s, []).append(vv)
        if len(at_y) >= 2:
            v = sum(statistics.median(vs) for vs in at_y.values())
            if y >= MIN_NETZERO_YEAR and v <= 1e-9:
                return _single("single_endpoint_sector_sum", y, v, len(at_y))

    # --- relative trajectory (% below / of a base year, or an index) ---
    if rel_recs:
        by = {}
        for y, rem in rel_recs:
            by.setdefault(y, []).append(rem)
        series = {y: statistics.median(v) for y, v in by.items()}
        ys = sorted(series)
        y, rem = ys[-1], series[ys[-1]]
        if y >= MIN_NETZERO_YEAR:
            return {
                "series_source": "relative_reduction", "n_years": len(ys),
                "first_year": ys[0] if len(ys) > 1 else "",
                "first_emissions_Mt": "",
                "last_year": y, "last_emissions_Mt": "",
                "ratio_last_to_first": round(rem, 4),
                "n_sectors_first": "", "n_sectors_last": "",
                "is_net_negative": rem < 0, "is_netzero": rem <= fraction,
            }
    return None


def flag_netzero_scenarios(output_dir, fraction):
    """Read harmonized-emissions.csv and write netzero-flags.csv.

    One row per (paper, canonical scenario, canonical area, basis) that carries
    any net-zero evidence.  See _classify_series for the tiers -- an absolute
    CO2/CO2e trajectory (ratio of last year to first), a lone late "~0"
    reading, or a late "-95 % vs base" / index point.
    """
    src = output_dir / "harmonized-emissions.csv"
    with open(src, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=";"))

    groups = {}
    for row in rows:
        year = _year_int(row.get("Year"))
        if year is None:
            continue
        etype = (row.get("Emission_Type") or "")
        unit_std = row.get("Value_std_Unit", "")
        scen_c = (row.get("Scenario_canon") or "").strip()
        area_c = (row.get("Area_canon") or "").strip()

        abs_val = None
        if unit_std in _ABS_CO2_UNITS:
            try:
                abs_val = float(row["Value_std"])
            except (TypeError, ValueError, KeyError):
                abs_val = None

        rel_rem = None
        if abs_val is None and "CO2" in etype:
            raw_unit = row.get("Value_Unit", "")
            if re.search(r"%|index|fraction|percent", raw_unit, re.I):
                try:
                    rv = float((row.get("Value") or "").replace("%", "").strip())
                except (TypeError, ValueError):
                    rv = None
                if rv is not None:
                    rel_rem = _relative_remaining(
                        rv, raw_unit, row.get("Emission_Type_Detail", ""))

        if abs_val is None and rel_rem is None:
            continue

        basis = "CO2eq" if ("eq" in unit_std or "eq" in etype.lower()) else "CO2"
        key = (row["paper_id"], scen_c, area_c, basis)
        g = groups.setdefault(
            key, {"recs": [], "rel": [], "meta": row,
                  "names": {}, "areas": {}})
        if abs_val is not None:
            g["recs"].append((year, abs_val, row.get("Sector_std", "")))
        if rel_rem is not None:
            g["rel"].append((year, rel_rem))
        g["names"][(row.get("Scenario_Name") or "").strip()] = \
            g["names"].get((row.get("Scenario_Name") or "").strip(), 0) + 1
        g["areas"][(row.get("Area") or "").strip()] = \
            g["areas"].get((row.get("Area") or "").strip(), 0) + 1

    out_rows = []
    for (paper_id, scen_c, area_c, basis), grp in sorted(groups.items()):
        info = _classify_series(grp["recs"], grp["rel"], fraction)
        if info is None:
            continue
        meta = grp["meta"]
        disp_name = max(grp["names"], key=grp["names"].get) if grp["names"] else ""
        disp_area = max(grp["areas"], key=grp["areas"].get) if grp["areas"] else ""
        out_rows.append({
            "paper_id": paper_id,
            "Title": meta.get("Title", ""),
            "DOI": meta.get("DOI", ""),
            "Publication_Year": meta.get("Publication_Year", ""),
            "Scenario_Name": disp_name,
            "Area": disp_area,
            "scenario_canon": scen_c,
            "area_canon": area_c,
            "basis": basis,
            "netzero_fraction": fraction,
            **info,
        })

    out_path = output_dir / "netzero-flags.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=NETZERO_FIELDS, delimiter=";", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(out_rows)

    nz = sum(1 for r in out_rows if r["is_netzero"] is True)
    nn = sum(1 for r in out_rows if r["is_net_negative"] is True and r["is_netzero"] is True)
    src_counts = {}
    for r in out_rows:
        if r["is_netzero"] is True:
            src_counts[r["series_source"]] = src_counts.get(r["series_source"], 0) + 1

    print(f"\nnetzero-flags.csv: {len(out_rows)} scenarios evaluated "
          f"(fraction={fraction}) -> {out_path}")
    print(f"    {nz:5d}  flagged net zero ({nn} of them net-negative), "
          f"{len({r['paper_id'] for r in out_rows if r['is_netzero'] is True})} papers")
    for key in sorted(src_counts):
        print(f"    {src_counts[key]:5d}  {key} (flagged)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Harmonise units in the merged stage3c tables"
    )
    parser.add_argument(
        "input_dir", nargs="?", default=str(DEFAULT_INPUT_DIR),
        help="folder with merged-*.csv (default: %(default)s)",
    )
    parser.add_argument(
        "output_dir", nargs="?", default=None,
        help="where to write harmonized-*.csv (default: same as input_dir)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="re-download the World Bank reference data",
    )
    parser.add_argument(
        "--netzero-fraction", type=float, default=0.05,
        help="last-year emissions below this fraction of first-year emissions "
             "=> net zero (default: %(default)s)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for category in CATEGORY_FIELDS:
        if not (input_dir / f"merged-{category}.csv").is_file():
            sys.exit(f"missing input: {input_dir / f'merged-{category}.csv'}")

    fx, us_deflator = load_reference_data(args.refresh)
    money = MoneyConverter(fx, us_deflator)

    deflator_last = max(us_deflator)
    print(
        f"US GDP deflator: {min(us_deflator)}-{deflator_last}"
        f"{' (2025 extrapolated)' if money.target_extrapolated else ''}"
    )

    for category in CATEGORY_FIELDS:
        harmonize_category(category, input_dir, output_dir, money)

    flag_netzero_scenarios(output_dir, args.netzero_fraction)


if __name__ == "__main__":
    main()
