# scripts/_pipeline_common.py
#
# Shared key handling for the stage-3c harmonize + analysis scripts (04-10).
#
# The extractor records the SAME scenario or area with different surface forms
# in different parts of a paper (and across the two extraction passes):
#
#     "NZ_2050"  "NZ 2050"  "NZ-2050"          -> one scenario
#     "NZE"      "2050 NZE"                     -> one scenario
#     "USA"      "United States"  "U.S."       -> one area
#     "All scenarios"  "All modeled scenarios" -> a paper-wide value, no scenario
#
# Grouping / joining on the raw strings therefore splits one scenario into
# several sub-threshold groups.  Every script that groups by scenario or area,
# or joins a cost/mitigation row to netzero-flags.csv, funnels the key through
# here first.
#
# Import (scripts/ is added to sys.path by each caller):
#
#     from _pipeline_common import canon_scenario, canon_area, \
#         is_catchall_scenario, resolve_scenario

import re


# ============================================================================
# SCENARIO NAME
# ============================================================================

_SCEN_SYNONYMS = (
    (re.compile(r"\bnet[\s_-]*zero\b", re.I), "net zero"),
    (re.compile(r"\bnetzero\b", re.I), "net zero"),
    (re.compile(r"\bclimate[\s_-]*neutral(?:ity)?\b", re.I), "climate neutral"),
    (re.compile(r"\bcarbon[\s_-]*neutral(?:ity)?\b", re.I), "carbon neutral"),
    (re.compile(r"\bnz[\s_-]*(\d)", re.I), r"nz \1"),   # NZ50 -> nz 50
)
_SCEN_SEP = re.compile(r"[_\-/.,:;()\[\]{}]+")
_WS = re.compile(r"\s+")


def canon_scenario(name):
    """Formatting-only canonical form of a scenario name: lower-case, unified
    separators / whitespace, net-zero spelling variants merged.  Years and any
    other distinguishing token are KEPT, so "NZ 2040" and "NZ 2050" stay
    distinct.  Returns "" for an empty / missing name."""
    s = (name or "").strip()
    if not s:
        return ""
    s = s.replace("&", " and ")
    s = _SCEN_SEP.sub(" ", s)
    for pattern, repl in _SCEN_SYNONYMS:
        s = pattern.sub(repl, s)
    s = _WS.sub(" ", s).strip().lower()
    return s


_CATCHALL_EXACT = {
    "", "model assumption", "model assumptions", "common assumption",
    "common assumptions", "not reported", "n/a", "na", "none", "unspecified",
    "general", "all", "overall", "aggregate",
}
_CATCHALL_RE = re.compile(
    r"^(all|every|each|both)\b.*\b("
    r"scenario|scenarios|pathway|pathways|case|cases|run|runs|"
    r"model|models|modell?ed|variant|variants|simulation|simulations)\b",
    re.I,
)


def is_catchall_scenario(name):
    """True when the scenario field is not a real scenario but a note that a
    value applies to the whole paper ("All scenarios", "All modeled
    scenarios", "Model assumption", "", ...).  Used to spread a paper-wide
    cost onto every flagged scenario of that paper."""
    s = _WS.sub(" ", (name or "").strip().lower())
    return s in _CATCHALL_EXACT or bool(_CATCHALL_RE.match(s))


# ============================================================================
# AREA
# ============================================================================

# Only unambiguous aliases.  Regional labels that are genuinely broader than a
# single country (e.g. "Europe" vs "EU") are left untouched.
_AREA_ALIASES = {
    "us": "United States", "usa": "United States", "u.s.": "United States",
    "u.s.a.": "United States", "u.s.a": "United States", "u.s": "United States",
    "united states of america": "United States", "the united states": "United States",
    "uk": "United Kingdom", "u.k.": "United Kingdom", "u.k": "United Kingdom",
    "great britain": "United Kingdom",
    "prc": "China", "p.r. china": "China", "mainland china": "China",
    "people's republic of china": "China", "china (mainland)": "China",
    "eu": "EU", "eu27": "EU", "eu-27": "EU", "eu 27": "EU", "eu28": "EU",
    "eu-28": "EU", "eu 28": "EU", "european union": "EU", "eu27+uk": "EU",
    "world": "Global", "worldwide": "Global", "globe": "Global",
    "global total": "Global", "the world": "Global",
    "the netherlands": "Netherlands", "holland": "Netherlands",
    "republic of korea": "South Korea", "korea, rep.": "South Korea",
    "s. korea": "South Korea", "south korea": "South Korea",
    "republic of ireland": "Ireland",
}


def canon_area(area):
    """Canonical area label: well-known aliases folded to one spelling, other
    values passed through with whitespace tidied.  Returns "" for empty."""
    s = (area or "").strip()
    if not s:
        return ""
    key = _WS.sub(" ", s.lower().strip()).rstrip(".")
    return _AREA_ALIASES.get(key, s)


# ============================================================================
# SCENARIO JOIN
# ============================================================================

_STOP_TOKENS = {"scenario", "scenarios", "case", "pathway", "the", "a", "of"}


def resolve_scenario(scenario, per_paper):
    """Look a scenario up in `per_paper` (a {canon_scenario: payload} dict built
    from ONE paper's flagged scenarios).  Returns the payload or None.

        1. exact canonical match
        2. token-subset match, when exactly one flagged scenario's tokens are a
           subset (or superset) of this scenario's tokens -- recovers
           "NZE" <-> "2050 NZE"

    Deliberately conservative: an ambiguous subset match (>1 candidate) yields
    None, and a match on a single stop-word token ("scenario") is ignored."""
    c = canon_scenario(scenario)
    if not c:
        return None
    if c in per_paper:
        return per_paper[c]
    toks = set(c.split()) - _STOP_TOKENS
    if not toks:
        return None
    hits = []
    for name, payload in per_paper.items():
        other = set(name.split()) - _STOP_TOKENS
        if other and (other <= toks or toks <= other):
            hits.append(payload)
    return hits[0] if len(hits) == 1 else None
