# 07-stage3c-mitigation-contribution.py
#
# Exploratory analysis: what is the relative contribution of the different
# mitigation options to reaching climate neutrality?
#
# Mitigation options are reported in incommensurable units (MW, MWh, MtCO2/yr,
# %, km2), so no single ratio spans all of them.  Two harmonised measures,
# one per panel:
#
#   PANEL A  carbon-management options (CCS, BECCS, DACCS, afforestation & land,
#            other CDR, CCU) -- contribution = the CO2 the option captures or
#            removes in the net-zero year, as a share of the scenario's
#            baseline (first-year) gross emissions from netzero-flags.csv.
#
#   PANEL B  low-carbon supply mix -- for scenarios that report generation by
#            source in MWh, each source's share of total low-carbon generation
#            in the net-zero year.
#
# Both measures are within-scenario fractions, so country size drops out.
# Every number in the figure and the generated doc is computed here.
#
# INPUT   outputs/output-stage3c/harmonized-mitigation.csv
#         outputs/output-stage3c/netzero-flags.csv
# OUTPUT  outputs/analysis/mitigation-contribution-carbon.csv
#         outputs/analysis/mitigation-contribution-energy.csv
#         figures/mitigation-contribution.png
#         docs/mitigation-contribution.html
#
#   python3 scripts/07-stage3c-mitigation-contribution.py

from pathlib import Path
import argparse
import base64
import csv
import html
import re
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_common import (canon_scenario, canon_area,   # noqa: E402
                              resolve_scenario)

INK, INK2, MUTED, GRID = "#1a1a1a", "#55534e", "#8a8a84", "#dedcd5"
PANEL_A = "#2f7d54"   # green  -- carbon management
PANEL_B = "#2a78d6"   # blue   -- energy supply

CARBON_MEASURES = {
    "Carbon capture and storage (CCS)": "CCS",
    "Carbon dioxide removal": None,       # split by subtype below
    "Carbon capture and utilization (CCU)": "CCU",
}
STRICT_CO2 = {"MtCO2/yr", "MtCO2eq/yr", "MtCO2", "MtCO2eq"}
MWH_UNITS = {"MWh", "MWh/yr"}

# a resulting contribution above this multiple of baseline emissions means the
# baseline denominator is wrong (a mis-matched netzero flag) -- drop, count
MAX_PLAUSIBLE_SHARE = 3.0

DEFAULT_MIT = Path("outputs/output-stage3c/harmonized-mitigation.csv")
DEFAULT_FLAGS = Path("outputs/output-stage3c/netzero-flags.csv")
DEFAULT_OUTDIR = Path("outputs/analysis")
DEFAULT_FIGDIR = Path("figures")
DEFAULT_DOC = Path("docs/mitigation-contribution.html")


# ============================================================================
# HELPERS
# ============================================================================

def parse_year(raw):
    s = (raw or "").strip()
    if re.fullmatch(r"(?:19|20)\d{2}", s):
        return int(s)
    years = re.findall(r"(?:19|20)\d{2}", s)
    return int(years[-1]) if len(years) == 1 else None


def load_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=";"))


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def build_flag_index(flags):
    """psa: {(paper, scenario_canon, area_canon): rec}
       pp : {paper: {scenario_canon: rec}}  (for the token-subset fallback)"""
    psa, pp = {}, {}
    for r in flags:
        y = _to_float(r.get("last_year"))
        if y is None:
            continue
        rec = {"last_year": int(y),
               "first_em": _to_float(r.get("first_emissions_Mt"))}
        sc = r.get("scenario_canon") or canon_scenario(r.get("Scenario_Name"))
        ac = r.get("area_canon") or canon_area(r.get("Area"))
        psa[(r["paper_id"], sc, ac)] = rec
        pp.setdefault(r["paper_id"], {})[sc] = rec
    return psa, pp


def flag_lookup(row, psa, pp):
    pid = row["paper_id"]
    sc = row.get("Scenario_canon") or canon_scenario(row.get("Scenario_Name"))
    ac = row.get("Area_canon") or canon_area(row.get("Area"))
    if (pid, sc, ac) in psa:
        return psa[(pid, sc, ac)]
    per_paper = pp.get(pid)
    if not per_paper:
        return None
    return per_paper.get(sc) or resolve_scenario(sc, per_paper)


def std_value(row, units):
    """(std value, std unit, raw value, raw unit) for the first Amount/Capacity
    field whose standardised unit is in `units`; (None, None, "", "") if none."""
    for field, ufield, raw_f, raw_uf in (
        ("Amount_std", "Amount_std_Unit", "Amount", "Amount_Unit"),
        ("Capacity_std", "Capacity_std_Unit", "Capacity", "Capacity_Unit"),
    ):
        u = (row.get(ufield) or "").strip()
        if u in units and row.get(field):
            try:
                return (float(row[field]), u,
                        (row.get(raw_f) or "").strip(),
                        (row.get(raw_uf) or "").strip())
            except ValueError:
                pass
    return None, None, "", ""


def source_ref(row):
    """'Figure 3' / 'Table S5' / 'Section 4.2' style reference for one row."""
    st = (row.get("Source_Type") or "").strip()
    sn = (row.get("Source_Number") or "").strip()
    if st and sn:
        return f"{st} {sn}"
    return st or sn or "Not reported"


def is_cumulative(row):
    text = (row.get("Mitigation_Measure_Detail", "") + " "
            + row.get("Comment", "")).lower()
    return "cumulat" in text


def cdr_subtype(row):
    t = (row.get("Subsector_std", "") + " "
         + row.get("Mitigation_Measure_Detail", "")).lower()
    if "beccs" in t or ("bioenergy" in t and ("captur" in t or "ccs" in t)):
        return "BECCS"
    if "direct air" in t or "daccs" in t or re.search(r"\bdac\b", t):
        return "DACCS"
    if any(k in t for k in ("afforest", "reforest", "forest", "land-use",
                            "land use", "restoration", "vegetation",
                            "soil carbon", "peat", "wetland", "weathering",
                            "olivine", "biochar", "tree planting")):
        return "Afforestation & land"
    return "Other CDR"


def energy_source(row):
    m = row["Mitigation_Measure"]
    sub = (row.get("Subsector_std") or "").lower()
    det = (row.get("Mitigation_Measure_Detail") or "").lower()
    if m == "Nuclear Energy":
        return "Nuclear"
    if m == "Renewable Energy":
        if "hydro" in det:
            return "Hydro"
        if "wind" in det:
            return "Wind"
        if "solar" in det or re.search(r"\bpv\b", det):
            return "Solar"
        if "geotherm" in det:
            return "Geothermal"
        if "bio" in det:
            return "Bioenergy"
        return "Other renewables"
    if m == "Other measure":
        if "bioenergy" in sub or "biomass" in det or "biofuel" in det:
            return "Bioenergy"
        if "hydrogen" in sub or "hydrogen" in det or "synthetic" in det:
            return "Hydrogen / synfuel"
        if ("power: generation" in sub or "power generation" in det
                or "electricity generation" in det):
            return "Other generation"
    return None


# ============================================================================
# BUILD  --  PANEL A : carbon-management contribution
# ============================================================================

def build_carbon(mit_rows, psa, ps):
    scen = {}
    for r in mit_rows:
        if r["Mitigation_Measure"] not in CARBON_MEASURES:
            continue
        value, unit, raw_v, raw_u = std_value(r, STRICT_CO2)
        if value is None:
            continue
        flag = flag_lookup(r, psa, ps)
        if flag is None:
            continue
        cat = CARBON_MEASURES[r["Mitigation_Measure"]] or cdr_subtype(r)
        sc = r.get("Scenario_canon") or canon_scenario(r.get("Scenario_Name"))
        ac = r.get("Area_canon") or canon_area(r.get("Area"))
        scen.setdefault(
            (r["paper_id"], sc, ac),
            {"flag": flag, "title": r.get("Title", ""),
             "doi": r.get("DOI", ""),
             "scenario": (r.get("Scenario_Name") or "").strip(),
             "area": (r.get("Area") or "").strip(), "items": []}
        )["items"].append({
            "cat": cat, "year": parse_year(r.get("Year")),
            "value": abs(value), "flow": unit.endswith("/yr"),
            "cumulative": is_cumulative(r),
            "raw": f"{raw_v} {raw_u}".strip(), "src": source_ref(r),
        })

    rows, suspect = [], 0
    for (paper_id, _sc, _ac), g in sorted(scen.items()):
        scenario, area = g["scenario"], g["area"]
        baseline = g["flag"]["first_em"]
        if not baseline or baseline <= 0:
            continue
        last_year = g["flag"]["last_year"]
        years = sorted({it["year"] for it in g["items"] if it["year"]})
        if not years:
            continue
        endpoint = max([y for y in years if y <= last_year] or years)

        by_cat = {}
        for it in g["items"]:
            if it["year"] != endpoint or it["cumulative"]:
                continue
            by_cat.setdefault(it["cat"], {"flow": [], "stock": []})
            bucket = "flow" if it["flow"] else "stock"
            by_cat[it["cat"]][bucket].append(it)

        for cat, buckets in by_cat.items():
            used = buckets["flow"] or buckets["stock"]
            if not used:
                continue
            vals = [it["value"] for it in used]
            share = statistics.median(vals) / baseline
            if share > MAX_PLAUSIBLE_SHARE:
                suspect += 1
                continue
            rows.append({
                "paper_id": paper_id, "scenario": scenario, "area": area,
                "title": g["title"], "doi": g["doi"], "category": cat,
                "endpoint_year": endpoint, "baseline_Mt": round(baseline, 1),
                "option_MtCO2_per_yr": round(statistics.median(vals), 2),
                "used_stock": not buckets["flow"],
                "contribution_pct_of_baseline": round(share * 100, 2),
                "n_source_rows": len(used),
                "raw_values": " | ".join(sorted({it["raw"] for it in used})),
                "source": " | ".join(sorted({it["src"] for it in used})),
            })
    return rows, suspect


# ============================================================================
# BUILD  --  PANEL B : low-carbon supply mix
# ============================================================================

def build_energy(mit_rows, psa, ps):
    scen = {}
    for r in mit_rows:
        value, _, raw_v, raw_u = std_value(r, MWH_UNITS)
        if value is None or value <= 0:
            continue
        src = energy_source(r)
        if src is None:
            continue
        flag = flag_lookup(r, psa, ps)
        if flag is None:
            continue
        sc = r.get("Scenario_canon") or canon_scenario(r.get("Scenario_Name"))
        ac = r.get("Area_canon") or canon_area(r.get("Area"))
        scen.setdefault(
            (r["paper_id"], sc, ac),
            {"flag": flag, "title": r.get("Title", ""),
             "doi": r.get("DOI", ""),
             "scenario": (r.get("Scenario_Name") or "").strip(),
             "area": (r.get("Area") or "").strip(), "items": []}
        )["items"].append({
            "src": src, "year": parse_year(r.get("Year")), "value": value,
            "raw": f"{raw_v} {raw_u}".strip(), "srcref": source_ref(r),
        })

    rows = []
    for (paper_id, _sc, _ac), g in sorted(scen.items()):
        scenario, area = g["scenario"], g["area"]
        last_year = g["flag"]["last_year"]
        years = sorted({it["year"] for it in g["items"] if it["year"]})
        if not years:
            continue
        endpoint = max([y for y in years if y <= last_year] or years)
        by_src = {}
        for it in g["items"]:
            if it["year"] == endpoint:
                by_src.setdefault(it["src"], []).append(it)
        agg = {s: statistics.median(it["value"] for it in v)
               for s, v in by_src.items()}
        total = sum(v for v in agg.values() if v > 0)
        if total <= 0 or sum(1 for v in agg.values() if v > 0) < 2:
            continue
        for src, v in agg.items():
            if v <= 0:
                continue
            used = by_src[src]
            rows.append({
                "paper_id": paper_id, "scenario": scenario, "area": area,
                "title": g["title"], "doi": g["doi"], "source": src,
                "endpoint_year": endpoint,
                "generation_MWh_per_yr": round(v, 1),
                "share_of_low_carbon_pct": round(v / total * 100, 2),
                "n_source_rows": len(used),
                "raw_values": " | ".join(sorted({it["raw"] for it in used})),
                "source_ref": " | ".join(sorted({it["srcref"] for it in used})),
            })
    return rows


# ============================================================================
# STATISTICS
# ============================================================================

def _summary(rows, key, value):
    out, papers = {}, {}
    for r in rows:
        out.setdefault(r[key], []).append(r[value])
        papers.setdefault(r[key], set()).add(r["paper_id"])
    summ = []
    for k, vals in out.items():
        vals = sorted(vals)
        if len(vals) >= 4:
            q1, _, q3 = statistics.quantiles(vals, n=4)
        else:
            q1, q3 = vals[0], vals[-1]
        summ.append({
            "name": k, "n": len(vals), "n_papers": len(papers[k]),
            "min": round(vals[0], 1),
            "q1": round(q1, 1), "median": round(statistics.median(vals), 1),
            "q3": round(q3, 1), "max": round(vals[-1], 1),
        })
    # order by median, but rows with n < 3 (a point cloud, not a distribution)
    # sink to the bottom so they don't out-rank real distributions
    summ.sort(key=lambda s: (s["n"] < 3, -s["median"]))
    return summ


def compute_stats(carbon_rows, energy_rows, suspect, n_flags):
    ca = _summary(carbon_rows, "category", "contribution_pct_of_baseline")
    en = _summary(energy_rows, "source", "share_of_low_carbon_pct")
    ca_top = ca[0] if ca else None
    return {
        "n_flags": n_flags,
        "carbon_summary": ca,
        "energy_summary": en,
        "carbon_scenarios": len({(r["paper_id"], r["scenario"], r["area"])
                                 for r in carbon_rows}),
        "carbon_papers": len({r["paper_id"] for r in carbon_rows}),
        "energy_scenarios": len({(r["paper_id"], r["scenario"], r["area"])
                                 for r in energy_rows}),
        "energy_papers": len({r["paper_id"] for r in energy_rows}),
        "suspect_dropped": suspect,
        "carbon_top": ca_top,
        "carbon_year_min": min((r["endpoint_year"] for r in carbon_rows),
                               default=None),
        "carbon_year_max": max((r["endpoint_year"] for r in carbon_rows),
                               default=None),
        "energy_top": en[0] if en else None,
        "energy_solar_wind": [s for s in en if s["name"] in ("Solar", "Wind")],
    }


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter=";",
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ============================================================================
# FIGURE
# ============================================================================

def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": INK,
        "axes.edgecolor": GRID, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.dpi": 160,
    })


def _jitter(n, spread=0.26):
    if n <= 1:
        return [0.0]
    step = (2 * spread) / (n - 1)
    return [-spread + i * step for i in range(n)]


def _panel(ax, summary, rows, key, value, colour, xlabel, xmax):
    order = [s["name"] for s in summary]
    lab_x = -xmax * 0.02
    for i, s in enumerate(summary):
        y = len(order) - 1 - i
        vals = sorted(r[value] for r in rows if r[key] == s["name"])
        if s["n"] >= 4:
            ax.plot([s["q1"], s["q3"]], [y - 0.30, y - 0.30], color=INK2,
                    lw=1.4, zorder=3)
        if s["n"] >= 3:
            ax.plot([s["median"], s["median"]], [y - 0.38, y - 0.22],
                    color=INK, lw=2.4, zorder=4)
        clipped = 0
        for dy, v in zip(_jitter(len(vals)), vals):
            if v > xmax:
                clipped += 1
                ax.scatter([xmax], [y + 0.08 + dy], s=34, c=colour, marker=">",
                           edgecolors="white", linewidths=0.6, alpha=0.9,
                           zorder=5)
            else:
                ax.scatter([max(v, -6)], [y + 0.08 + dy], s=34, c=colour,
                           edgecolors="white", linewidths=0.7, alpha=0.9,
                           zorder=5)
        ax.text(lab_x, y + 0.06, s["name"], ha="right", va="center",
                fontsize=9.5, color=INK)
        ax.text(lab_x, y - 0.30, f"n={s['n']} ({s['n_papers']})", ha="right",
                va="center", fontsize=7.5, color=MUTED)
        tag = f"{s['median']:.0f}%" if s["n"] >= 3 else "—"
        if clipped:
            tag += f"   ({clipped}›{xmax:.0f}%)"
        ax.text(xmax + xmax * 0.03, y + 0.02, tag, ha="left", va="center",
                fontsize=8.2, fontweight="bold", color=INK)
        if i:
            ax.axhline(y + 0.5, color=GRID, lw=0.8, zorder=1)
    ax.axvline(0, color="#b4b2aa", lw=1.2, zorder=2)
    ax.set_xlim(-xmax * 0.03, xmax * 1.18)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontsize=8.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="x", color=GRID, lw=0.7)


def figure_caption(stats):
    d = stats["suspect_dropped"]
    dropped = "" if not d else (
        f"; {d} row{'s' if d != 1 else ''} dropped for an implausible denominator")
    return (
        f"Exploratory. Panel A: carbon captured/removed in the net-zero year as "
        f"a share of the scenario's baseline emissions ({stats['carbon_scenarios']} "
        f"scenarios, {stats['carbon_papers']} papers{dropped}). Panel B: each "
        f"source's share of reported low-carbon generation "
        f"({stats['energy_scenarios']} scenarios that give a source split, "
        f"{stats['energy_papers']} papers). Renewables and efficiency are mostly "
        f"reported in MW/MWh not MtCO2, so they are under-represented in panel A. "
        f"Dot = scenario, bar = median, whisker = IQR; n = scenarios (papers), "
        f"n < 3 shown without a median."
    )


def _p90_xmax(vals, lo, hi):
    v = sorted(vals)
    p90 = v[min(len(v) - 1, int(0.9 * len(v)))]
    import math
    return float(min(hi, max(lo, math.ceil(p90 / 10) * 10)))


def make_figure(carbon_rows, energy_rows, stats, path):
    ca, en = stats["carbon_summary"], stats["energy_summary"]
    ca_xmax = _p90_xmax(
        [r["contribution_pct_of_baseline"] for r in carbon_rows], 45, 120)
    en_xmax = _p90_xmax(
        [r["share_of_low_carbon_pct"] for r in energy_rows], 60, 100)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.6, 0.52 * (len(ca) + len(en)) + 2.4),
        gridspec_kw={"height_ratios": [len(ca), len(en)]})

    _panel(ax1, ca, carbon_rows, "category", "contribution_pct_of_baseline",
           PANEL_A,
           "carbon captured / removed  ·  % of baseline (first-year) emissions",
           ca_xmax)
    ax1.set_title("A  ·  Carbon-management options", fontsize=10.5, loc="left",
                  color=INK, pad=10)

    _panel(ax2, en, energy_rows, "source", "share_of_low_carbon_pct",
           PANEL_B, "share of reported low-carbon generation  ·  %", en_xmax)
    ax2.set_title("B  ·  Low-carbon supply mix", fontsize=10.5, loc="left",
                  color=INK, pad=10)

    fig.suptitle(
        "Relative contribution of mitigation options to net zero",
        fontsize=12, x=0.02, ha="left", y=0.995)
    fig.text(0.5, -0.11 / (len(ca) + len(en)) * 6, figure_caption(stats),
             ha="center", va="top", fontsize=7.3, color=MUTED, wrap=True)

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# DOCUMENT
# ============================================================================

DOC_CSS = """
  :root{--ground:#f5f5f2;--panel:#efeeea;--panel-line:#dddcd4;--ink:#232427;
    --ink-soft:#4c4d4e;--muted:#7c7b73;--accent:#2f7d54;--flag:#9a6a1c;
    --rule:#cfcec6;--code-bg:#ecebe5}
  @media(prefers-color-scheme:dark){:root:not([data-theme="light"]){
    --ground:#16171a;--panel:#1d1e21;--panel-line:#2c2d31;--ink:#e9e7e1;
    --ink-soft:#c3c1b9;--muted:#8f8e85;--accent:#5cb98a;--flag:#d0a558;
    --rule:#2f3034;--code-bg:#232428}}
  :root[data-theme="dark"]{--ground:#16171a;--panel:#1d1e21;--panel-line:#2c2d31;
    --ink:#e9e7e1;--ink-soft:#c3c1b9;--muted:#8f8e85;--accent:#5cb98a;
    --flag:#d0a558;--rule:#2f3034;--code-bg:#232428}
  *{box-sizing:border-box}
  body{margin:0;background:var(--ground);color:var(--ink);
    font-family:"Newsreader",Georgia,serif;font-size:18px;line-height:1.62;
    font-variant-numeric:oldstyle-nums}
  .topbar{position:sticky;top:0;z-index:10;display:flex;
    justify-content:space-between;align-items:baseline;gap:1rem;
    padding:.7rem clamp(1rem,4vw,2.5rem);
    background:color-mix(in srgb,var(--ground) 88%,transparent);
    backdrop-filter:blur(8px);border-bottom:1px solid var(--rule);
    font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.74rem;
    letter-spacing:.04em;text-transform:uppercase}
  .topbar b{color:var(--accent);font-weight:600}.topbar span{color:var(--muted)}
  main{max-width:46rem;margin:0 auto;padding:0 clamp(1rem,4vw,2rem) 6rem}
  section{margin-top:4.5rem}
  .hero{margin-top:3.2rem}
  .kicker{font-family:"IBM Plex Mono",monospace;font-size:.76rem;
    letter-spacing:.14em;text-transform:uppercase;color:var(--accent);
    margin:0 0 1.1rem}
  h1{font-family:"IBM Plex Mono",monospace;font-weight:600;
    font-size:clamp(2rem,6vw,2.9rem);line-height:1.08;letter-spacing:-.01em;
    text-wrap:balance;margin:0 0 1rem}
  .lede{font-size:1.26rem;line-height:1.5;color:var(--ink-soft);
    max-width:35rem;margin:0 0 2.4rem}
  .lede em{font-style:italic;color:var(--ink)}
  h2{font-family:"IBM Plex Mono",monospace;font-weight:600;font-size:1.32rem;
    letter-spacing:-.005em;margin:0 0 1.1rem;padding-bottom:.5rem;
    border-bottom:1px solid var(--rule);text-wrap:balance}
  h3{font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:.95rem;
    letter-spacing:.02em;text-transform:uppercase;color:var(--ink-soft);
    margin:2.2rem 0 .8rem}
  p{margin:0 0 1.05rem;max-width:40rem}
  a{color:var(--accent)}
  strong{font-weight:600}
  ul{padding-left:1.2rem;margin:0 0 1.05rem;max-width:40rem}
  li{margin:.35rem 0}
  code{font-family:"IBM Plex Mono",monospace;font-size:.86em;
    background:var(--code-bg);padding:.06em .38em;border-radius:2px;
    white-space:nowrap}
  figure{margin:1.6rem 0 .6rem;padding:.9rem;background:var(--panel);
    border:1px solid var(--panel-line)}
  figure img{width:100%;height:auto;display:block}
  figcaption{font-family:"IBM Plex Mono",monospace;font-size:.74rem;
    color:var(--muted);margin-top:.7rem;line-height:1.5}
  .flow{display:flex;flex-wrap:wrap;gap:.5rem;align-items:stretch;
    margin:1.4rem 0 1.8rem;font-family:"IBM Plex Mono",monospace;font-size:.8rem}
  .flow .step{flex:1 1 8rem;background:var(--panel);
    border:1px solid var(--panel-line);padding:.7rem .8rem}
  .flow .step b{color:var(--accent);display:block;font-weight:600;
    margin-bottom:.2rem}
  .flow .arr{align-self:center;color:var(--muted)}
  .scroll{overflow-x:auto;margin:1.4rem 0 1.6rem}
  table{width:100%;border-collapse:collapse;font-family:"IBM Plex Mono",monospace;
    font-size:.8rem;font-variant-numeric:tabular-nums}
  th{text-align:left;font-weight:600;color:var(--ink-soft);
    text-transform:uppercase;letter-spacing:.04em;font-size:.7rem;
    padding:.5rem .8rem .5rem 0;border-bottom:2px solid var(--accent);
    white-space:nowrap}
  td{padding:.45rem .8rem .45rem 0;border-bottom:1px solid var(--rule);
    vertical-align:top}
  td.n,th.n{text-align:right;padding-right:1.2rem}
  tr:last-child td{border-bottom:none}
  .t-note{color:var(--muted);font-size:.78rem}
  .callout{background:var(--panel);border:1px solid var(--panel-line);
    border-left:3px solid var(--flag);padding:1rem 1.2rem;margin:1.4rem 0;
    font-size:.96rem}
  .callout ul{margin:0}
  footer{max-width:46rem;margin:5rem auto 0;padding:2rem clamp(1rem,4vw,2rem) 3rem;
    border-top:1px solid var(--rule);font-family:"IBM Plex Mono",monospace;
    font-size:.76rem;color:var(--muted)}
  footer code{background:none;color:var(--ink-soft)}
  @media(max-width:640px){body{font-size:16.5px}.lede{font-size:1.12rem}}
  @media(prefers-reduced-motion:reduce){*{transition:none!important}}
"""


def _rows_html(summary, unit_word):
    return "\n".join(
        f"          <tr><td>{html.escape(s['name'])}</td>"
        f"<td class=\"n\">{s['n']}</td>"
        f"<td class=\"n\">{s['n_papers']}</td>"
        f"<td class=\"n\">{s['median']:.0f}%</td>"
        f"<td class=\"n\">{s['q1']:.0f}&ndash;{s['q3']:.0f}%</td>"
        f"<td class=\"n\">{s['min']:.0f}&ndash;{s['max']:.0f}%</td></tr>"
        for s in summary)


def build_doc(stats, png_b64):
    s = stats
    ct, et = s["carbon_top"], s["energy_top"]
    sw = s["energy_solar_wind"]
    sw_txt = ""
    if len(sw) == 2:
        a, b = sw
        sw_txt = (f" {html.escape(a['name'])} and {html.escape(b['name'])} "
                  f"together are a median "
                  f"{a['median'] + b['median']:.0f}% of it.")

    return f"""<title>Mitigation Contribution at Net Zero</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500;6..72,600&family=Newsreader:ital@1&display=swap">

<style>{DOC_CSS}</style>

<div class="topbar">
  <b>mitigation&#8202;·&#8202;contribution</b>
  <span>07-stage3c-mitigation-contribution.py</span>
</div>

<main>
  <div class="hero">
    <p class="kicker">Data extraction pipeline &nbsp;/&nbsp; analysis</p>
    <h1>Mitigation Contribution at Net Zero</h1>
    <p class="lede">
      Mitigation options are reported in units that don't compare &mdash; MW,
      MWh, MtCO&#8322;/yr, percentages. <em>Two within-scenario fractions</em>
      make them comparable: carbon handled as a share of baseline emissions,
      and each energy source as a share of low-carbon supply.
    </p>
  </div>

  <figure>
    <img src="data:image/png;base64,{png_b64}" alt="Two-panel strip plot: carbon-management options as a percent of baseline emissions, and low-carbon energy sources as a share of generation, across net-zero scenarios.">
    <figcaption>{html.escape(figure_caption(stats))}</figcaption>
  </figure>

  <section>
    <h2>What it shows</h2>
    <ul>
      <li><strong>Carbon management is modest per route.</strong> The largest
        single carbon-management contribution
        (<strong>{html.escape(ct['name'])}</strong>) is a median
        <strong>{ct['median']:.0f}%</strong> of baseline emissions
        ({ct['n']} scenarios); the routes below it sit between
        {s['carbon_summary'][-1]['median']:.0f}% and
        {s['carbon_summary'][1]['median']:.0f}%. No single removal or capture
        route carries the transition on its own.</li>
      <li><strong>The low-carbon supply mix is solar-led.</strong>
        <strong>{html.escape(et['name'])}</strong> is a median
        <strong>{et['median']:.0f}%</strong> of reported low-carbon
        generation.{sw_txt}</li>
      <li>The spread within every row is wide &mdash; scenarios disagree far
        more about <em>how much</em> each option contributes than about
        <em>which</em> options appear.</li>
    </ul>
    <div class="scroll">
      <table>
        <caption>Panel A &mdash; carbon captured / removed, % of baseline emissions</caption>
        <thead><tr><th>Option</th><th class="n">scenarios</th>
          <th class="n">papers</th><th class="n">median</th>
          <th class="n">IQR</th><th class="n">range</th></tr></thead>
        <tbody>
{_rows_html(s['carbon_summary'], 'baseline')}
        </tbody>
      </table>
    </div>
    <div class="scroll">
      <table>
        <caption>Panel B &mdash; source share of reported low-carbon generation</caption>
        <thead><tr><th>Source</th><th class="n">scenarios</th>
          <th class="n">papers</th><th class="n">median</th>
          <th class="n">IQR</th><th class="n">range</th></tr></thead>
        <tbody>
{_rows_html(s['energy_summary'], 'generation')}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Method</h2>
    <div class="flow">
      <div class="step"><b>1</b> net-zero filter</div>
      <div class="arr">&rarr;</div>
      <div class="step"><b>2</b> option rows at the end year</div>
      <div class="arr">&rarr;</div>
      <div class="step"><b>3</b> normalise per scenario</div>
    </div>
    <ul>
      <li><strong>Net-zero membership</strong> from <code>netzero-flags.csv</code>
        (<code>is_netzero&nbsp;==&nbsp;True</code>) &mdash; the same filter as
        the other figures. {s['n_flags']} scenarios qualify.</li>
      <li><strong>Panel A.</strong> <code>harmonized-mitigation.csv</code> rows
        for CCS / CDR / CCU with a clean <code>MtCO2[eq][/yr]</code> value.
        CDR is split by <code>Subsector_std</code> / detail into BECCS, DACCS,
        afforestation &amp; land, other. Cumulative-stock rows are dropped;
        annual-flow rows win where both exist. Contribution =
        <code>|option MtCO2/yr|&nbsp;/&nbsp;first_emissions_Mt</code> from the
        flag &mdash; a size-free share of the emissions the pathway had to
        eliminate. Shares above {MAX_PLAUSIBLE_SHARE * 100:.0f}% of baseline
        ({s['suspect_dropped']} rows) are dropped as a mis-matched
        denominator.</li>
      <li><strong>Panel B.</strong> Rows reporting generation in
        <code>MWh[/yr]</code> by source (Solar, Wind, Hydro, Bioenergy,
        Geothermal, Nuclear, Hydrogen / synfuel, other). A scenario is kept
        only if it names &ge; 2 sources at the end year; each source's share is
        of the scenario's summed low-carbon generation.</li>
      <li><strong>End year</strong> = the latest year the scenario reports the
        option, capped at the flag's net-zero year.</li>
    </ul>
  </section>

  <section>
    <h2>Caveats</h2>
    <div class="callout">
      <ul>
        <li><strong>Panel A leans toward carbon management.</strong> Renewables,
          efficiency and electrification are mostly reported in MW / MWh, not
          MtCO&#8322;, so their CO&#8322; contribution is rarely stated and they
          barely appear here.</li>
        <li>Panel A n = {s['carbon_scenarios']} scenarios /
          {s['carbon_papers']} papers; panel B n = {s['energy_scenarios']} /
          {s['energy_papers']}. Small rows (a source or route with n &lt; 3)
          are a point cloud, not a distribution.</li>
        <li>CO&#8322; and CO&#8322;e are pooled; a few absolute
          <code>MtCO2</code> rows may be cumulative despite no &ldquo;cumulative&rdquo;
          label, which would inflate their share.</li>
        <li>Panel B is a share of the <em>reported</em> sources, not of total
          final energy &mdash; a scenario that only lists solar and wind reads
          as 100% those two even if it also runs nuclear.</li>
        <li>End years span {s['carbon_year_min']}&ndash;{s['carbon_year_max']};
          the panels are not additive with each other (MWh &ne; MtCO&#8322;).</li>
      </ul>
    </div>
  </section>
</main>

<footer>
  reproduce &nbsp;&rarr;&nbsp;
  <code>python3 scripts/07-stage3c-mitigation-contribution.py</code><br>
  inputs: <code>harmonized-mitigation.csv</code>, <code>netzero-flags.csv</code>
  &nbsp;&middot;&nbsp; outputs: <code>figures/mitigation-contribution.png</code>,
  <code>outputs/analysis/mitigation-contribution-{{carbon,energy}}.csv</code>
</footer>
"""


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mitigation", nargs="?", default=str(DEFAULT_MIT))
    ap.add_argument("--flags", default=str(DEFAULT_FLAGS))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--figures-dir", default=str(DEFAULT_FIGDIR))
    ap.add_argument("--doc", default=str(DEFAULT_DOC))
    args = ap.parse_args()

    flags = [r for r in load_csv(Path(args.flags))
             if r.get("is_netzero") == "True"]
    psa, ps = build_flag_index(flags)
    mit_rows = load_csv(Path(args.mitigation))

    carbon_rows, suspect = build_carbon(mit_rows, psa, ps)
    energy_rows = build_energy(mit_rows, psa, ps)
    if not carbon_rows or not energy_rows:
        raise SystemExit("not enough data for one of the panels")

    stats = compute_stats(carbon_rows, energy_rows, suspect, len(flags))

    outdir = Path(args.outdir)
    write_csv(outdir / "mitigation-contribution-carbon.csv", carbon_rows, [
        "paper_id", "title", "doi", "scenario", "area", "category",
        "endpoint_year", "baseline_Mt", "option_MtCO2_per_yr", "used_stock",
        "contribution_pct_of_baseline", "n_source_rows", "raw_values", "source",
    ])
    write_csv(outdir / "mitigation-contribution-energy.csv", energy_rows, [
        "paper_id", "title", "doi", "scenario", "area", "source",
        "endpoint_year", "generation_MWh_per_yr", "share_of_low_carbon_pct",
        "n_source_rows", "raw_values", "source_ref",
    ])

    fig_path = Path(args.figures_dir) / "mitigation-contribution.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _style()
    make_figure(carbon_rows, energy_rows, stats, fig_path)

    doc_path = Path(args.doc)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(
        build_doc(stats, base64.b64encode(fig_path.read_bytes()).decode()),
        encoding="utf-8")

    print(f"net-zero scenarios flagged : {stats['n_flags']}")
    print(f"  panel A (carbon)  : {stats['carbon_scenarios']} scenarios, "
          f"{stats['carbon_papers']} papers, {stats['suspect_dropped']} dropped")
    for s in stats["carbon_summary"]:
        print(f"      {s['name']:<22} n={s['n']:>2}  median {s['median']:>5.0f}%"
              f"  IQR [{s['q1']:.0f},{s['q3']:.0f}]%")
    print(f"  panel B (energy)  : {stats['energy_scenarios']} scenarios, "
          f"{stats['energy_papers']} papers")
    for s in stats["energy_summary"]:
        print(f"      {s['name']:<22} n={s['n']:>2}  median {s['median']:>5.0f}%")
    print(f"\nfigure -> {fig_path}\ndoc    -> {doc_path}")


if __name__ == "__main__":
    main()
